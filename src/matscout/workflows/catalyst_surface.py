"""Catalyst surface + adsorbate workflow (the step beyond bulk pre-screening).

For a bulk candidate: generate low-index slabs, place reaction-relevant adsorbates,
ML-relax (cell fixed), compute adsorption energies against CHE gas references, and map
to HER / OER / CO2RR activity proxies (Sabatier optimality / overpotential).

This is expensive, so it runs only on the top-K bulk catalyst candidates. Energies are
CHGNet estimates of free-energy diagrams — directional proxies, not DFT.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

from rich.console import Console
from rich.progress import track

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from ..properties.adsorbates import (
    ADSORBATE_DG_CORRECTION,
    ADSORBATES,
    GAS_MOLECULES,
    reference_energy,
)
from ..properties.common import peak_score
from ..relaxation.chgnet_relaxer import CHGNetRelaxer

console = Console()

# adsorbates needed per reaction
REACTION_ADSORBATES = {
    "HER": ["H"],
    "OER": ["OH", "O", "OOH"],
    "CO2RR": ["CO", "COOH"],
}


@dataclass
class SurfaceResult:
    candidate_id: str
    formula: str
    miller: tuple[int, int, int] | None = None
    adsorption_energies: dict[str, float] = field(default_factory=dict)  # *X -> E_ads (eV)
    free_energies: dict[str, float] = field(default_factory=dict)  # *X -> dG (eV)
    her_score: float | None = None
    oer_overpotential: float | None = None
    oer_score: float | None = None
    co2rr_score: float | None = None
    error: str | None = None


def _slab_relaxer(g: GlobalConfig) -> CHGNetRelaxer:
    # cell fixed for slabs/molecules (preserve vacuum); fewer steps (bigger cells)
    return CHGNetRelaxer(device=g.device, fmax=0.1, max_steps=120, relax_cell=False)


@lru_cache(maxsize=1)
def _gas_reference_energies_cached(device: str) -> tuple:
    relaxer = CHGNetRelaxer(device=device, fmax=0.1, max_steps=200, relax_cell=False)
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mol in GAS_MOLECULES.items():
            try:
                boxed = mol.get_boxed_structure(14, 14, 14)
                res = relaxer.relax(boxed)
                if res.ok and res.energy_per_atom is not None:
                    out[name] = res.energy_per_atom * len(boxed)
            except Exception:
                continue
    return tuple(sorted(out.items()))


def gas_reference_energies(g: GlobalConfig) -> dict[str, float]:
    return dict(_gas_reference_energies_cached(g.device))


def _make_slab(structure, miller, min_slab=6.0, min_vac=12.0):
    from pymatgen.core.surface import SlabGenerator

    sg = SlabGenerator(structure, miller, min_slab_size=min_slab, min_vacuum_size=min_vac,
                       center_slab=True, primitive=True, max_normal_search=2)
    slabs = sg.get_slabs(symmetrize=False)
    return slabs[0] if slabs else None


def _adsorption_energy(slab, e_slab, adsorbate_name, relaxer, e_ref, max_sites=2):
    """Strongest (most negative) adsorption energy over a few sites; None on failure."""
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder

    mol = ADSORBATES[adsorbate_name]
    try:
        asf = AdsorbateSiteFinder(slab)
        ads_structs = asf.generate_adsorption_structures(mol, repeat=[1, 1, 1], min_lw=4.0)
    except Exception:
        return None
    if not ads_structs:
        return None
    best = None
    for ads in ads_structs[:max_sites]:
        try:
            res = relaxer.relax(ads)
            if res.ok and res.energy_per_atom is not None:
                e_total = res.energy_per_atom * len(ads)
                e_ads = e_total - e_slab - e_ref
                if best is None or e_ads < best:
                    best = e_ads
        except Exception:
            continue
    return best


def evaluate_surface(record: CandidateRecord, reactions, g: GlobalConfig,
                     miller=(1, 1, 1)) -> SurfaceResult:
    res = SurfaceResult(candidate_id=record.candidate_id, formula=record.reduced_formula)
    relaxer = _slab_relaxer(g)
    gas = gas_reference_energies(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slab = _make_slab(record.get_structure(), miller)
        except Exception as exc:
            res.error = f"slab_gen_failed:{str(exc)[:80]}"
            return res
        if slab is None:
            res.error = "no_slab"
            return res
        res.miller = miller
        slab_relax = relaxer.relax(slab)
        if not slab_relax.ok:
            res.error = "slab_relax_failed"
            return res
        e_slab = slab_relax.energy_per_atom * len(slab)

        needed = sorted({a for r in reactions for a in REACTION_ADSORBATES.get(r, [])})
        for ads_name in needed:
            e_ref = reference_energy(ads_name, gas)
            if e_ref is None:
                continue
            e_ads = _adsorption_energy(slab, e_slab, ads_name, relaxer, e_ref)
            if e_ads is not None:
                res.adsorption_energies[ads_name] = round(e_ads, 3)
                res.free_energies[ads_name] = round(
                    e_ads + ADSORBATE_DG_CORRECTION.get(ads_name, 0.0), 3
                )

    _score_reactions(res, reactions)
    return res


def _score_reactions(res: SurfaceResult, reactions) -> None:
    dg = res.free_energies
    if "HER" in reactions and "H" in dg:
        # Sabatier: ideal dG_H = 0; good window +/-0.2, decays over 0.4
        res.her_score = round(peak_score(dg["H"], 0.0, 0.4), 3)
    if "OER" in reactions and all(k in dg for k in ("OH", "O", "OOH")):
        steps = [dg["OH"], dg["O"] - dg["OH"], dg["OOH"] - dg["O"], 4.92 - dg["OOH"]]
        eta = max(steps) - 1.23
        res.oer_overpotential = round(eta, 3)
        res.oer_score = round(max(0.0, 1.0 - eta / 0.8), 3)  # eta 0 ->1, 0.8 ->0
    if "CO2RR" in reactions and "CO" in dg:
        # moderate CO binding desirable (Cu-like ~ -0.5 eV); too strong/weak penalized
        res.co2rr_score = round(peak_score(dg["CO"], -0.5, 0.8), 3)


def run_catalyst_surfaces(
    ranked: list[CandidateRecord],
    campaign: CampaignConfig,
    g: GlobalConfig,
    top_k: int = 8,
    miller=(1, 1, 1),
) -> list[SurfaceResult]:
    reactions = campaign.reaction_targets or ["HER", "OER", "CO2RR"]
    subset = [r for r in ranked if not r.rejection_reason][:top_k]
    console.print(f"[cyan]Surface analysis[/] on top {len(subset)} catalysts, "
                  f"Miller={miller}, reactions={reactions}")
    results = []
    for r in track(subset, description="surfaces"):
        results.append(evaluate_surface(r, reactions, g, miller=miller))
    ok = [x for x in results if not x.error]
    console.print(f"[green]Surface analysis done:[/] {len(ok)}/{len(results)} succeeded")
    return results


def write_surface_report(results: list[SurfaceResult], g: GlobalConfig) -> str:
    from datetime import date

    from .paths import Paths

    lines = [
        "# Catalyst surface / adsorbate analysis",
        "",
        f"_Generated {date.today().isoformat()}. CHGNet slab + adsorbate relaxation; "
        "adsorption free energies vs CHE gas references._",
        "",
        "| formula | Miller | dG(*H) | dG(*OH) | dG(*O) | dG(*OOH) | dG(*CO) | "
        "HER | OER η(V) | OER | CO2RR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(d, k):
        return f"{d[k]:.2f}" if k in d else "—"

    for r in results:
        if r.error:
            lines.append(f"| {r.formula} | — | _{r.error}_ |||||||||")
            continue
        dg = r.free_energies
        lines.append(
            f"| {r.formula} | {r.miller} | {f(dg,'H')} | {f(dg,'OH')} | {f(dg,'O')} | "
            f"{f(dg,'OOH')} | {f(dg,'CO')} | "
            f"{r.her_score if r.her_score is not None else '—'} | "
            f"{r.oer_overpotential if r.oer_overpotential is not None else '—'} | "
            f"{r.oer_score if r.oer_score is not None else '—'} | "
            f"{r.co2rr_score if r.co2rr_score is not None else '—'} |"
        )

    lines += [
        "",
        "## Interpretation & caveats",
        "- **HER**: best when dG(*H) ≈ 0 (Sabatier). **OER**: lower overpotential η is better "
        "(thermodynamic limiting potential from the 4 proton-coupled steps). **CO2RR**: moderate "
        "dG(*CO) (~ -0.5 eV, Cu-like) is desirable.",
        "- These are **CHGNet estimates on a single low-index termination** — directional only. "
        "Oxide surfaces are polar and termination-sensitive; quantitative activity requires DFT "
        "(with dipole correction, solvation, applied potential / pH) or an OCP/fairchem model.",
        "- Use this to *prioritize* which bulk catalysts merit a full surface DFT study, not as a "
        "final activity ranking.",
    ]
    out = Paths(g).reports() / "catalyst_surface.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Surface report:[/] {out}")
    return str(out)
