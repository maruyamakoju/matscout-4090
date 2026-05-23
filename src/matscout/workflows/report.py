"""Markdown reports + top-200 CIF/POSCAR export + executive summary."""

from __future__ import annotations

import json
from datetime import date

from pymatgen.io.vasp.inputs import Poscar
from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from ..scoring.objectives import APP_SCORERS, SCORE_FIELD
from ..scoring.ranker import dedup_by_formula, select_wildcards
from .paths import Paths

console = Console()


def _explain(record: CandidateRecord, campaign: CampaignConfig) -> list[str]:
    module = APP_SCORERS.get(campaign.name)
    if module is None:
        return [f"{record.reduced_formula}: score={record.final_score}"]
    try:
        result = module.score_candidate(record, campaign, {"cohort_min_energy": {}})
        return module.explain_score(record, result)
    except Exception:
        return [f"{record.reduced_formula}: score={record.final_score}"]


def write_campaign_report(ranked: list[CandidateRecord], campaign: CampaignConfig, g: GlobalConfig, top_n: int = 20) -> None:
    paths = Paths(g)
    field = SCORE_FIELD[campaign.name]
    top = dedup_by_formula(ranked)[:top_n]
    lines = [
        f"# {campaign.name.replace('_', ' ').title()} shortlist",
        "",
        f"_Generated {date.today().isoformat()} by MatScout-4090. "
        f"{len(ranked)} surviving candidates; showing top {len(top)}._",
        "",
        "| # | formula | final | app | e_hull | gap(eV) | novelty | synth | unc | priority |",
        "|---|---------|-------|-----|--------|---------|---------|-------|-----|----------|",
    ]
    from .export_dft import _priority

    for i, r in enumerate(top, 1):
        app = getattr(r, field) or 0.0
        eh = f"{r.ml_e_above_hull:.3f}" if r.ml_e_above_hull is not None else "n/a"
        gap = f"{r.predicted_bandgap_ev:.2f}" if r.predicted_bandgap_ev is not None else "n/a"
        lines.append(
            f"| {i} | {r.reduced_formula} | {r.final_score:.3f} | {app:.3f} | {eh} | {gap} | "
            f"{r.novelty_score:.2f} | {r.synthesizability_score:.2f} | {r.uncertainty_score:.3f} | {_priority(r)} |"
        )

    lines += ["", "## Top candidate rationale", ""]
    for i, r in enumerate(top[:5], 1):
        lines.append(f"### {i}. {r.reduced_formula}  (`{r.candidate_id}`)")
        for ex in _explain(r, campaign):
            lines.append(f"- {ex}")
        lines.append(f"- provenance: source={r.source}, parent={r.parent_source_id or r.parent_mp_id}")
        lines.append("")

    out = paths.reports() / f"{campaign.name}_shortlist.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Report:[/] {out}")


def write_top200_and_outputs(
    all_ranked: dict[str, list[CandidateRecord]],
    g: GlobalConfig,
    campaigns: dict[str, CampaignConfig],
) -> None:
    """Write combined parquet, top-200 CIF/POSCAR, executive summary."""
    from ..data.store import save_parquet

    paths = Paths(g)

    # combined pools
    all_records: list[CandidateRecord] = []
    for recs in all_ranked.values():
        all_records.extend(recs)
    save_parquet(all_records, paths.candidates_all())

    combined = sorted(all_records, key=lambda r: (-(r.final_score or 0.0), r.candidate_id))
    save_parquet(combined, paths.candidates_ranked())

    # top 200 across all campaigns (deduped by formula) -> CIF + POSCAR + metadata
    top200 = dedup_by_formula(combined)[:200]
    tdir = paths.top_200()
    for stale in list(tdir.glob("*.cif")) + list(tdir.glob("*.poscar")) + list(tdir.glob("*.json")):
        stale.unlink()  # idempotent: clear previous run's files
    meta = []
    for i, r in enumerate(top200, 1):
        struct = r.get_structure()
        base = f"{i:03d}_{r.reduced_formula}_{r.candidate_id.replace('::', '-')}"
        struct.to(filename=str(tdir / f"{base}.cif"), fmt="cif")
        Poscar(struct).write_file(tdir / f"{base}.poscar")
        meta.append({
            "rank": i, "candidate_id": r.candidate_id, "formula": r.reduced_formula,
            "campaign_scores": {k: getattr(r, v) for k, v in SCORE_FIELD.items()},
            "final_score": r.final_score, "ml_e_above_hull": r.ml_e_above_hull,
            "predicted_bandgap_ev": r.predicted_bandgap_ev, "novelty_score": r.novelty_score,
            "uncertainty_score": r.uncertainty_score, "source": r.source,
            "tags": r.tags,
        })
    (tdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # executive summary
    _write_executive_summary(all_ranked, paths, campaigns)
    console.print(f"[green]Top-200[/] CIF/POSCAR + metadata -> {tdir}")


def _write_executive_summary(all_ranked, paths: Paths, campaigns) -> None:
    lines = [
        "# MatScout-4090 — Executive Summary",
        "",
        f"_Generated {date.today().isoformat()}._",
        "",
        "Autonomous high-throughput screening across battery, semiconductor, catalyst, "
        "solar and CO₂-capture applications. Local generation + CHGNet/MACE relaxation + "
        "application scoring + novelty/synthesizability. Top candidates exported as "
        "DFT-ready VASP/QE decks.",
        "",
        "## Per-campaign top 3",
        "",
    ]
    for name, recs in all_ranked.items():
        field = SCORE_FIELD.get(name, "final_score")
        lines.append(f"### {name.replace('_', ' ').title()}  ({len(recs)} survivors)")
        for i, r in enumerate(dedup_by_formula(recs)[:3], 1):
            eh = f"{r.ml_e_above_hull:.3f}" if r.ml_e_above_hull is not None else "n/a"
            lines.append(
                f"{i}. **{r.reduced_formula}** — final={r.final_score:.3f}, "
                f"{name}_score={getattr(r, field) or 0:.3f}, e_hull={eh}, "
                f"novelty={r.novelty_score:.2f}, gap={r.predicted_bandgap_ev}"
            )
        lines.append("")

    # wildcards across all campaigns
    everything = [r for recs in all_ranked.values() for r in recs]
    wc = dedup_by_formula(select_wildcards(everything, 60))[:20]
    lines += ["## Wildcards (high novelty + interesting uncertainty)", ""]
    for i, r in enumerate(wc, 1):
        lines.append(
            f"{i}. **{r.reduced_formula}** — novelty={r.novelty_score:.2f}, "
            f"uncertainty={r.uncertainty_score:.3f}, final={r.final_score:.3f}"
        )
    if not wc:
        lines.append("_(none met the wildcard criteria in this run)_")

    lines += [
        "",
        "## Caveats",
        "- ML-relaxed energies and e_above_hull are surrogate estimates (CHGNet/MACE), "
        "not DFT. Predicted bandgaps are heuristic first-pass filters.",
        "- Catalyst scores are bulk pre-screens; real activity requires surface + adsorbate DFT.",
        "- CO₂-capture scoring requires pore descriptors (CoRE MOF/QMOF/Zeo++); "
        "candidates without them carry low confidence.",
        "- Generated structures are hypotheses for DFT/experimental verification, not "
        "confirmed materials.",
    ]
    out = paths.reports() / "executive_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Executive summary:[/] {out}")
