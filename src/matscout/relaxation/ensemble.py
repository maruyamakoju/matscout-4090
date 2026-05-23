"""Relaxation ensemble: CHGNet (primary) + MACE (secondary) with uncertainty.

Stage 1: CHGNet fast relax.
Stage 2: MACE cross-check (energy + geometry).
Stage 3: disagreement -> uncertainty score.
Then apply KEEP / MAYBE / KILL (technical spec §5.2).

Degrades gracefully: with no ML backend installed it runs a dry pass (structures
flow through unrelaxed, flagged ml_relaxed=False) so the rest of the pipeline works.
"""

from __future__ import annotations

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from rich.console import Console
from rich.progress import track

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord, structure_hash
from ..validation.phase_diagram import compute_e_above_hull
from ..validation.structure_checks import validate_structure
from .base import RelaxResult
from .chgnet_relaxer import CHGNetRelaxer
from .mace_relaxer import MACERelaxer
from .matgl_relaxer import M3GNetRelaxer

console = Console()

_MODEL_REGISTRY = {
    "chgnet": lambda cfg: CHGNetRelaxer(device=cfg.device, fmax=cfg.relaxation.fmax_ev_a,
                                        max_steps=cfg.relaxation.max_steps,
                                        relax_cell=cfg.relaxation.cell_relax),
    "mace_mp": lambda cfg: MACERelaxer(variant="mp", device=cfg.device,
                                       fmax=cfg.relaxation.fmax_ev_a,
                                       max_steps=cfg.relaxation.max_steps,
                                       relax_cell=cfg.relaxation.cell_relax),
    "mace_omat": lambda cfg: MACERelaxer(variant="omat", device=cfg.device,
                                         fmax=cfg.relaxation.fmax_ev_a,
                                         max_steps=cfg.relaxation.max_steps,
                                         relax_cell=cfg.relaxation.cell_relax),
    "m3gnet": lambda cfg: M3GNetRelaxer(device=cfg.device, fmax=cfg.relaxation.fmax_ev_a,
                                        max_steps=cfg.relaxation.max_steps,
                                        relax_cell=cfg.relaxation.cell_relax),
}


def build_relaxers(global_cfg: GlobalConfig):
    rc = global_cfg.relaxation
    primary = _MODEL_REGISTRY.get(rc.primary_model, _MODEL_REGISTRY["chgnet"])(global_cfg)
    secondaries = []
    for name in rc.secondary_models:
        if name in _MODEL_REGISTRY:
            secondaries.append(_MODEL_REGISTRY[name](global_cfg))
    return primary, secondaries


def _geometry_rmsd(s1, s2) -> float | None:
    try:
        matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
        res = matcher.get_rms_dist(s1, s2)
        return float(res[0]) if res else None
    except Exception:
        return None


def _apply_relaxed(record: CandidateRecord, result: RelaxResult, model_name: str) -> None:
    relaxed = result.relaxed_structure
    record.structure_cif = relaxed.to(fmt="cif")
    record.structure_hash = structure_hash(relaxed)
    record.nsites = len(relaxed)
    try:
        sym = relaxed.get_space_group_info()
        record.spacegroup_symbol, record.spacegroup_number = sym[0], sym[1]
    except Exception:
        pass
    record.ml_relaxed = True
    record.ml_model = model_name
    record.ml_energy_per_atom = result.energy_per_atom
    record.relaxation_converged = result.converged
    record.max_force_ev_a = result.max_force_ev_a
    record.volume_change_pct = result.volume_change_pct


def _classify(record: CandidateRecord, campaign: CampaignConfig) -> str:
    """Return KEEP / MAYBE / KILL and set rejection_reason on KILL."""
    f = campaign.filters
    # post-relax geometry sanity
    try:
        chk = validate_structure(record.get_structure())
        if not chk.ok:
            record.rejection_reason = f"post_relax_{chk.reason}"
            return "KILL"
    except Exception:
        record.rejection_reason = "post_relax_parse_error"
        return "KILL"

    if record.ml_relaxed and not record.relaxation_converged:
        record.rejection_reason = "not_converged"
        return "KILL"
    vc = abs(record.volume_change_pct) if record.volume_change_pct is not None else 0.0
    if vc > 30.0:  # absolute cell-collapse / explosion cap
        record.rejection_reason = f"huge_volume_change({vc:.0f}%)"
        return "KILL"

    eh = record.ml_e_above_hull
    if eh is not None:
        if eh > f.max_predicted_e_hull + 0.0:
            record.rejection_reason = f"e_above_hull_high({eh:.3f})"
            return "KILL" if eh > 0.10 else "MAYBE"
        if eh > 0.05 or vc > 20:
            return "MAYBE"
    if record.uncertainty_score > 0.08:
        return "MAYBE"
    return "KEEP"


_RELAXED_FIELDS = (
    "structure_cif", "structure_hash", "nsites", "spacegroup_symbol", "spacegroup_number",
    "ml_relaxed", "ml_model", "ml_energy_per_atom", "ml_e_above_hull",
    "relaxation_converged", "max_force_ev_a", "volume_change_pct",
    "uncertainty_score", "rejection_reason", "tags",
)


def _copy_relaxed_state(dst: CandidateRecord, src: CandidateRecord) -> None:
    for f in _RELAXED_FIELDS:
        setattr(dst, f, getattr(src, f))


def _load_checkpoint(checkpoint_path) -> dict[str, CandidateRecord]:
    from pathlib import Path

    from ..data.store import load_parquet

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {}
    try:
        prior = load_parquet(checkpoint_path)
    except Exception:
        return {}
    # only treat fully-processed records as done (relaxed or explicitly rejected)
    return {r.candidate_id: r for r in prior if r.ml_relaxed or r.rejection_reason}


def relax_records(
    records: list[CandidateRecord],
    global_cfg: GlobalConfig,
    campaign: CampaignConfig,
    limit: int | None = None,
    compute_hull: bool = True,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> list[CandidateRecord]:
    from ..data.store import save_parquet

    primary, secondaries = build_relaxers(global_cfg)
    subset = records[:limit] if limit else records

    if not primary.available():
        console.print(
            "[yellow]WARNING:[/] No ML potential available (torch/chgnet/mace). "
            "Running DRY relaxation: structures pass through unrelaxed (ml_relaxed=false)."
        )
        for r in subset:
            r.ml_relaxed = False
            r.tags.append("dry_relax")
        return subset

    done = _load_checkpoint(checkpoint_path) if resume else {}
    if done:
        console.print(f"[cyan]Resume:[/] {len(done)} candidates already relaxed (from checkpoint)")

    sec_names = [s.name for s in secondaries if s.available()]
    console.print(
        f"[cyan]Relaxing[/] {len(subset)} candidates on [bold]{primary.device}[/] "
        f"(primary={primary.name}, secondary={sec_names or 'none'})"
    )

    processed = 0
    for r in track(subset, description=f"relax/{campaign.name}"):
        if r.candidate_id in done:
            _copy_relaxed_state(r, done[r.candidate_id])
            continue
        res = primary.relax(r.get_structure())
        if not res.ok:
            r.rejection_reason = f"relax_failed:{res.error}"
            r.tags.append("relax_failed")
            continue
        _apply_relaxed(r, res, primary.name)
        energies = [res.energy_per_atom]

        for sec in secondaries:
            if not sec.available():
                continue
            sres = sec.relax(r.get_structure())
            if sres.ok and sres.energy_per_atom is not None:
                energies.append(sres.energy_per_atom)
                rmsd = _geometry_rmsd(res.relaxed_structure, sres.relaxed_structure)
                if rmsd is not None:
                    r.tags.append(f"rmsd_{sec.name}:{rmsd:.2f}")

        # uncertainty = std of per-atom energies across models
        if len(energies) >= 2:
            r.uncertainty_score = float(np.std(energies))
        if compute_hull:
            r.ml_e_above_hull = compute_e_above_hull(
                r.get_structure(), r.ml_energy_per_atom, global_cfg
            )

        verdict = _classify(r, campaign)
        r.tags.append(f"verdict:{verdict}")

        processed += 1
        if checkpoint_path and processed % checkpoint_every == 0:
            save_parquet(subset, checkpoint_path)

    if checkpoint_path:
        save_parquet(subset, checkpoint_path)
    return subset
