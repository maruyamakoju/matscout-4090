"""Active-learning loop (technical spec §8.1).

Rounds of: select (exploit / explore / diversity / random) -> mutate/expand ->
relax -> score -> accumulate. Selection balances exploitation against novelty,
diversity and moderate uncertainty.
"""

from __future__ import annotations

import random

from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from ..generation.mutate import expand_candidates
from ..relaxation.ensemble import relax_records
from ..scoring.objectives import score_records
from ..scoring.ranker import alive
from ..scoring.uncertainty import acquisition_score
from ..validation.novelty import NoveltyIndex

console = Console()


def _select_parents(records: list[CandidateRecord], n: int, rng: random.Random) -> list[CandidateRecord]:
    survivors = alive(records)
    if not survivors:
        return []
    by_acq = sorted(survivors, key=lambda r: -acquisition_score(r, r.final_score or 0.0))
    n_exploit = int(n * 0.4)
    n_explore = int(n * 0.4)
    pool = by_acq[:n_exploit]  # exploitation
    # exploration: high novelty
    explore = sorted(survivors, key=lambda r: -(r.novelty_score or 0))[:n_explore * 2]
    pool += rng.sample(explore, min(n_explore, len(explore)))
    # random control
    pool += rng.sample(survivors, min(n - len(pool), len(survivors)))
    # dedupe
    seen, out = set(), []
    for r in pool:
        if r.candidate_id not in seen:
            seen.add(r.candidate_id); out.append(r)
    return out


def run_active_loop(
    initial: list[CandidateRecord],
    seeds: list[CandidateRecord],
    campaign: CampaignConfig,
    g: GlobalConfig,
    rounds: int = 3,
    expand_per_round: int = 200,
    relax_per_round: int = 100,
) -> list[CandidateRecord]:
    rng = random.Random(g.seed)
    idx = NoveltyIndex.from_records(seeds) if seeds else NoveltyIndex()
    pool = list(initial)
    score_records(pool, campaign, g, novelty_index=idx)

    for rd in range(1, rounds + 1):
        console.rule(f"[bold]{campaign.name} active round {rd}/{rounds}")
        parents = _select_parents(pool, expand_per_round // 2, rng)
        children = expand_candidates(parents, campaign, g, expand_per_round, rng)
        relax_records(children, g, campaign, limit=relax_per_round)
        score_records(children, campaign, g, novelty_index=idx)
        pool.extend(children)
        best = max((r.final_score or 0) for r in pool)
        console.print(f"  +{len(children)} children, pool={len(pool)}, best final={best:.3f}")
    return pool
