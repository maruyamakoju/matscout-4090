"""Structural mutations for active-learning expansion (strain + rattle + swap)."""

from __future__ import annotations

import random

import numpy as np
from pymatgen.core import Structure

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord, structure_hash
from .pools import build_substitution_map, passes_element_filter


def mutate_structure(structure: Structure, rng: random.Random, strain: float = 0.04,
                     rattle: float = 0.08) -> Structure:
    new = structure.copy()
    # isotropic-ish strain
    f = 1.0 + rng.uniform(-strain, strain)
    new.scale_lattice(new.volume * f**3)
    # rattle atoms
    coords = new.cart_coords + np.array(
        [[rng.gauss(0, rattle) for _ in range(3)] for _ in range(len(new))]
    )
    for i, c in enumerate(coords):
        new[i] = new[i].species, new.lattice.get_fractional_coords(c)
    return new


def expand_candidates(
    parents: list[CandidateRecord],
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    n: int,
    rng: random.Random | None = None,
) -> list[CandidateRecord]:
    """Produce n mutated/substituted children from high-value parents (active loop)."""
    rng = rng or random.Random(global_cfg.seed + 7)
    sub_map = build_substitution_map(campaign, global_cfg)
    out: list[CandidateRecord] = []
    seen: set[str] = {p.structure_hash for p in parents}
    counter = 0
    if not parents:
        return out
    while len(out) < n:
        parent = rng.choice(parents)
        try:
            struct = parent.get_structure()
        except Exception:
            continue
        # 50% structural mutation, 50% extra substitution
        if rng.random() < 0.5:
            new_struct = mutate_structure(struct, rng)
        else:
            els = [str(sp) for sp in struct.composition.elements if str(sp) in sub_map]
            if not els:
                new_struct = mutate_structure(struct, rng)
            else:
                el = rng.choice(els)
                new_struct = struct.copy()
                new_struct.replace_species({el: rng.choice(sub_map[el])})
        elements = sorted({str(e) for e in new_struct.composition.elements})
        if not passes_element_filter(elements, campaign, global_cfg):
            continue
        h = structure_hash(new_struct)
        if h in seen:
            continue
        seen.add(h)
        counter += 1
        rec = CandidateRecord.from_structure(
            new_struct,
            candidate_id=f"mut::{counter:06d}",
            source=parent.source,
            parent_source_id=parent.candidate_id,
            tags=[campaign.name, "active_expand", f"from:{parent.candidate_id}"],
        )
        out.append(rec)
        if counter > n * 50:  # safety
            break
    return out[:n]
