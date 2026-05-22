"""Orchestrate candidate generation from all sources up to a target count."""

from __future__ import annotations

import random

from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from .mattergen_runner import load_mattergen_outputs
from .prototype_enum import generate_prototypes
from .substitution import generate_substitutions

console = Console()


def generate_candidates(
    seeds: list[CandidateRecord],
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    n: int | None = None,
) -> list[CandidateRecord]:
    """Combine MatterGen + substitution + prototype enumeration up to n candidates."""
    n = n or campaign.n_generate
    rng = random.Random(global_cfg.seed)

    pool: list[CandidateRecord] = list(seeds)  # keep seeds as candidates too

    mg = load_mattergen_outputs(campaign, global_cfg)
    pool.extend(mg)

    # split remaining budget ~60% substitution / 40% prototype
    remaining = max(n - len(pool), 0)
    n_sub = int(remaining * 0.6)
    n_proto = remaining - n_sub

    subs = generate_substitutions(seeds, campaign, global_cfg, n_sub, rng)
    pool.extend(subs)
    protos = generate_prototypes(seeds, campaign, global_cfg, n_proto, rng)
    pool.extend(protos)

    console.print(
        f"[cyan]Generated[/] {len(pool)} candidates for {campaign.name} "
        f"(seeds={len(seeds)}, mattergen={len(mg)}, sub={len(subs)}, proto={len(protos)})"
    )
    return pool[:n] if n else pool
