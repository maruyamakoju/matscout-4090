"""Ingest completed DFT runs and compare against the ML predictions (loop closure).

Scans `outputs/dft_queue/<campaign>/<deck>/` for finished VASP (vasprun.xml / OUTCAR)
or Quantum ESPRESSO outputs, extracts the DFT energy / bandgap / convergence, optionally
recomputes a real e_above_hull (needs MP key), and writes a Markdown report comparing
ML surrogate predictions to DFT. This is the feedback signal that closes the
active-learning loop: ML filters, DFT verifies, deltas inform the next round.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from ..config.schema import GlobalConfig
from .paths import Paths

console = Console()


@dataclass
class DFTResult:
    deck: str
    formula: str
    source: str  # "vasp" | "qe"
    candidate_id: str | None = None
    energy_per_atom: float | None = None
    bandgap_ev: float | None = None
    is_direct: bool | None = None
    converged: bool | None = None
    error: str | None = None
    ml: dict = field(default_factory=dict)  # ML predictions from metadata.json


def parse_vasp(vasp_dir: Path) -> DFTResult | None:
    """Parse a VASP deck via pymatgen (prefers static vasprun.xml, falls back to OUTCAR)."""
    xmls = [vasp_dir / "vasprun.xml.static", vasp_dir / "vasprun.xml"]
    xml = next((p for p in xmls if p.exists()), None)
    if xml is None and not (vasp_dir / "OUTCAR").exists():
        return None
    res = DFTResult(deck=vasp_dir.parent.name, formula="", source="vasp")
    try:
        if xml is not None:
            from pymatgen.io.vasp.outputs import Vasprun

            vr = Vasprun(str(xml), parse_dos=False, parse_eigen=True, parse_potcar_file=False)
            res.converged = bool(vr.converged)
            res.energy_per_atom = float(vr.final_energy) / len(vr.final_structure)
            gap, _cbm, _vbm, is_direct = vr.eigenvalue_band_properties
            res.bandgap_ev = float(gap)  # type: ignore[arg-type]
            res.is_direct = bool(is_direct)
            res.formula = vr.final_structure.composition.reduced_formula
        else:
            from pymatgen.io.vasp.outputs import Outcar

            oc = Outcar(str(vasp_dir / "OUTCAR"))
            res.energy_per_atom = float(oc.final_energy)  # type: ignore[arg-type]  # total, not per-atom
    except Exception as exc:
        res.error = f"vasp_parse:{str(exc)[:100]}"
    return res


def parse_qe(qe_dir: Path) -> DFTResult | None:
    """Best-effort Quantum ESPRESSO output parse (total energy + estimated gap)."""
    out = next((p for p in [qe_dir / "relax.out", qe_dir / "scf.out"] if p.exists()), None)
    if out is None:
        return None
    res = DFTResult(deck=qe_dir.parent.name, formula="", source="qe")
    try:
        text = out.read_text(encoding="utf-8", errors="ignore")
        # final total energy in Ry -> eV
        matches = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text)
        if matches:
            res.energy_per_atom = float(matches[-1]) * 13.605693  # Ry -> eV (total, not per-atom)
        res.converged = "JOB DONE" in text
        gap = re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", text)
        if gap:
            homo, lumo = float(gap[-1][0]), float(gap[-1][1])
            res.bandgap_ev = max(0.0, lumo - homo)
    except Exception as exc:
        res.error = f"qe_parse:{str(exc)[:100]}"
    return res


def _attach_ml(result: DFTResult, deck_dir: Path) -> None:
    meta = deck_dir / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            result.candidate_id = m.get("candidate_id")
            result.ml = {
                "formula": m.get("formula"),
                "ml_energy_per_atom": None,  # not in deck metadata; reported via parquet if needed
                "ml_e_above_hull": m.get("ml_e_above_hull"),
                "predicted_bandgap_ev": m.get("predicted_bandgap_ev"),
                "final_score": m.get("final_score"),
                "priority": m.get("priority"),
            }
            if not result.formula:
                result.formula = m.get("formula", "")
        except Exception:
            pass


def ingest_campaign(campaign: str, g: GlobalConfig) -> list[DFTResult]:
    paths = Paths(g)
    qdir = paths.dft_queue(campaign)
    results: list[DFTResult] = []
    for deck in sorted(p for p in qdir.iterdir() if p.is_dir()):
        r = parse_vasp(deck / "vasp")
        if r is None:
            r = parse_qe(deck / "qe")
        if r is None:
            continue  # not run yet
        _attach_ml(r, deck)
        # optional real e_hull from DFT energy
        if r.energy_per_atom is not None and g.mp_api_key:
            try:
                from pymatgen.core import Structure

                from ..validation.phase_diagram import compute_e_above_hull

                pos = deck / "vasp" / "CONTCAR"
                if pos.exists():
                    s = Structure.from_file(pos)
                    r.ml["dft_e_above_hull"] = compute_e_above_hull(s, r.energy_per_atom, g)
            except Exception:
                pass
        results.append(r)
    return results


def write_comparison_report(results: list[DFTResult], g: GlobalConfig) -> str:
    from datetime import date

    paths = Paths(g)
    lines = [
        "# DFT verification — ML vs DFT",
        "",
        f"_Generated {date.today().isoformat()}. {len(results)} completed DFT decks ingested._",
        "",
        "| deck | formula | DFT E/atom (eV) | DFT gap (eV) | ML gap (eV) | Δgap | conv |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ml_gap = r.ml.get("predicted_bandgap_ev")
        dgap = (f"{abs(r.bandgap_ev - ml_gap):.2f}"
                if (r.bandgap_ev is not None and ml_gap is not None) else "—")
        e = f"{r.energy_per_atom:.3f}" if r.energy_per_atom is not None else "—"
        dft_gap = f"{r.bandgap_ev:.2f}" if r.bandgap_ev is not None else "—"
        ml_gap_s = f"{ml_gap:.2f}" if ml_gap is not None else "—"
        conv = "yes" if r.converged else ("no" if r.converged is not None else "?")
        lines.append(
            f"| {r.deck} | {r.formula or r.ml.get('formula', '?')} | {e} | "
            f"{dft_gap} | {ml_gap_s} | {dgap} | {conv} |"
        )
    lines += [
        "",
        "_DFT gap is PBE-level from these decks (underestimates); escalate survivors to HSE06/GW. "
        "Energy/atom comparisons to ML are meaningful once on a common reference (MP corrections)._",
    ]
    out = paths.reports() / "dft_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def update_records_with_dft(results: list[DFTResult], campaign: str, g: GlobalConfig) -> int:
    """Write DFT energy/gap back into the ranked records (loop closure). Returns # updated."""
    from ..data.store import load_parquet, save_parquet

    paths = Paths(g)
    src = paths.ranked(campaign)
    if not src.exists():
        return 0
    records = load_parquet(src)
    by_id = {r.candidate_id: r for r in records}
    n = 0
    for res in results:
        rec = by_id.get(res.candidate_id) if res.candidate_id else None
        if rec is None:
            continue
        rec.dft_verified = True
        rec.dft_energy_per_atom = res.energy_per_atom
        rec.dft_bandgap_ev = res.bandgap_ev
        if "dft:verified" not in rec.tags:
            rec.tags.append("dft:verified")
        n += 1
    if n:
        out = paths.data / "processed" / campaign / "dft_verified.parquet"
        save_parquet(records, out)
    return n


def ingest_dft(campaign: str, g: GlobalConfig) -> list[DFTResult]:
    results = ingest_campaign(campaign, g)
    if not results:
        console.print(f"[yellow]No completed DFT decks found[/] in outputs/dft_queue/{campaign}/. "
                      "Run the decks on a cluster first (see docs/DFT_VERIFICATION.md).")
        return results
    report = write_comparison_report(results, g)
    updated = update_records_with_dft(results, campaign, g)
    n_conv = sum(1 for r in results if r.converged)
    console.print(f"[green]Ingested {len(results)} DFT decks[/] ({n_conv} converged, "
                  f"{updated} records updated) -> {report}")
    return results
