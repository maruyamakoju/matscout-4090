"""Generate candidates by elemental substitution on seed structures.

Same-group and isovalent (charge-preserving) swaps preserve stoichiometry, so they
tend to produce chemically reasonable, often-synthesizable derivatives.
"""

from __future__ import annotations

import itertools
import random

from pymatgen.core import Structure

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord, structure_hash
from .pools import build_substitution_map, passes_element_filter


def _apply_substitution(structure: Structure, mapping: dict[str, str]) -> Structure:
    new = structure.copy()
    new.replace_species(dict(mapping))
    return new


def generate_substitutions(
    seeds: list[CandidateRecord],
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    n: int,
    rng: random.Random | None = None,
    max_double_per_seed: int = 20,
) -> list[CandidateRecord]:
    rng = rng or random.Random(global_cfg.seed)
    sub_map = build_substitution_map(campaign, global_cfg)
    if n <= 0 or not seeds:
        return []

    # 1. Enumerate a finite list of (seed, mapping) operations.
    ops: list[tuple[CandidateRecord, Structure, dict[str, str]]] = []
    for seed in seeds:
        try:
            struct = seed.get_structure()
        except Exception:
            continue
        species = sorted({str(sp) for sp in struct.composition.elements})
        subbable = [e for e in species if e in sub_map]
        if not subbable:
            continue
        # all single substitutions (deterministic, finite)
        for el in subbable:
            for tgt in sub_map[el]:
                ops.append((seed, struct, {el: tgt}))
        # bounded sample of double substitutions
        if len(subbable) >= 2:
            pairs = list(itertools.combinations(subbable, 2))
            rng.shuffle(pairs)
            for a, b in pairs[:max_double_per_seed]:
                ops.append((seed, struct, {a: rng.choice(sub_map[a]),
                                           b: rng.choice(sub_map[b])}))

    # 2. Shuffle and realize until we hit n (single pass -> always terminates).
    rng.shuffle(ops)
    out: list[CandidateRecord] = []
    seen: set[str] = {s.structure_hash for s in seeds}
    counter = 0
    for seed, struct, mapping in ops:
        if len(out) >= n:
            break
        try:
            new_struct = _apply_substitution(struct, mapping)
        except Exception:
            continue
        els = sorted({str(e) for e in new_struct.composition.elements})
        if not passes_element_filter(els, campaign, global_cfg):
            continue
        h = structure_hash(new_struct)
        if h in seen:
            continue
        seen.add(h)
        counter += 1
        sub_str = ",".join(f"{k}->{v}" for k, v in mapping.items())
        rec = CandidateRecord.from_structure(
            new_struct,
            candidate_id=f"sub::{counter:06d}",
            source="substitution",
            parent_mp_id=seed.parent_mp_id,
            parent_source_id=seed.candidate_id,
            tags=[campaign.name, "substitution", f"from:{seed.reduced_formula}", f"swap:{sub_str}"],
        )
        out.append(rec)
    return out
