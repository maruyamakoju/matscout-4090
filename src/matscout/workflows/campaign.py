"""Campaign orchestration: per-stage functions + full pipeline runner."""

from __future__ import annotations

from rich.console import Console

from ..config.loader import load_campaign_config, load_global_config
from ..config.schema import CampaignConfig, GlobalConfig
from ..data.mp_client import fetch_mp_seeds
from ..data.store import save_parquet
from ..generation.generate import generate_candidates
from ..relaxation.ensemble import relax_records
from ..scoring.objectives import score_records
from ..scoring.ranker import rank_records
from ..validation.novelty import NoveltyIndex
from ..validation.structure_checks import validate_structure
from .paths import Paths

console = Console()


def stage_fetch(campaign: CampaignConfig, g: GlobalConfig, limit: int) -> list:
    seeds = fetch_mp_seeds(campaign, g, limit=limit)
    return seeds


def stage_generate(seeds, campaign, g, n=None):
    cands = generate_candidates(seeds, campaign, g, n=n)
    # pre-relax structure validity filter
    kept = []
    for r in cands:
        try:
            chk = validate_structure(r.get_structure(),
                                     min_dist=g.relaxation.min_interatomic_distance_a)
        except Exception:
            r.rejection_reason = "parse_error"
            kept.append(r)
            continue
        if not chk.ok:
            r.rejection_reason = f"pre_relax_{chk.reason}"
        kept.append(r)
    n_bad = sum(1 for r in kept if r.rejection_reason)
    console.print(f"[cyan]Pre-relax validation:[/] {len(kept) - n_bad}/{len(kept)} pass")
    return kept


def stage_relax(records, campaign, g, limit=None, checkpoint_path=None):
    # only relax structurally-valid candidates
    valid = [r for r in records if not r.rejection_reason]
    limit = limit if limit is not None else campaign.n_relax
    relax_records(valid, g, campaign, limit=limit, checkpoint_path=checkpoint_path)
    return records


def stage_score(records, campaign, g, seeds=None):
    seeds = seeds or []
    idx = NoveltyIndex.from_records(seeds) if seeds else NoveltyIndex()
    score_records(records, campaign, g, novelty_index=idx)
    return records


def run_campaign(
    name: str,
    g: GlobalConfig | None = None,
    n_generate: int | None = None,
    relax_limit: int | None = None,
    export_top: int = 50,
) -> list:
    """Run the full pipeline for one campaign and persist all artifacts."""
    g = g or load_global_config()
    campaign = load_campaign_config(name)
    paths = Paths(g)
    console.rule(f"[bold]Campaign: {name}")

    seeds = stage_fetch(campaign, g, limit=max(1000, (n_generate or campaign.n_generate)))
    cands = stage_generate(seeds, campaign, g, n=n_generate)
    save_parquet(cands, paths.generated(name))

    cands = stage_relax(cands, campaign, g, limit=relax_limit,
                        checkpoint_path=str(paths.relaxed(name)))
    save_parquet(cands, paths.relaxed(name))

    cands = stage_score(cands, campaign, g, seeds=seeds)
    save_parquet(cands, paths.scored(name))

    ranked = rank_records(cands, campaign)
    save_parquet(ranked[: campaign.top_k], paths.ranked(name))
    console.print(
        f"[green]{name}:[/] {len(cands)} scored, {len(ranked)} survivors, "
        f"top saved to {paths.ranked(name)}"
    )

    # downstream artifacts
    from .export_dft import export_dft_queue
    from .report import write_campaign_report

    export_dft_queue(ranked, campaign, g, top_k=export_top)
    write_campaign_report(ranked, campaign, g)
    return ranked
