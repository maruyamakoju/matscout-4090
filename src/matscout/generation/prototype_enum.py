"""Generate candidates by filling known structural prototypes with campaign elements.

A prototype is a seed structure with its distinct Wyckoff "slots". We classify each
slot as cation- or anion-like (by its electronegativity rank within the prototype) and
fill it with allowed elements of that role, sampling combinations.
"""

from __future__ import annotations

import random

from pymatgen.core import Element, Structure

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord, structure_hash
from .pools import passes_element_filter, site_role_pools


def _slot_roles(structure: Structure) -> dict[str, str]:
    """Classify each distinct element slot as 'anion' (most electronegative) or 'cation'."""
    els = sorted({str(sp) for sp in structure.composition.elements})
    if len(els) == 1:
        return {els[0]: "cation"}
    en = {e: Element(e).X or 0.0 for e in els}
    max_en = max(en.values())
    roles = {}
    for e in els:
        # treat the top-electronegativity element(s) as anion slot
        roles[e] = "anion" if en[e] >= max_en - 0.35 else "cation"
    return roles


def generate_prototypes(
    seeds: list[CandidateRecord],
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    n: int,
    rng: random.Random | None = None,
    max_per_prototype: int = 400,
) -> list[CandidateRecord]:
    rng = rng or random.Random(global_cfg.seed + 1)
    pools = site_role_pools(campaign, global_cfg)

    # unique prototype structures keyed by anonymous formula + spacegroup
    protos: dict[str, CandidateRecord] = {}
    for s in seeds:
        key = f"{s.anonymous_formula}|{s.spacegroup_number}"
        protos.setdefault(key, s)

    out: list[CandidateRecord] = []
    seen: set[str] = {s.structure_hash for s in seeds}
    counter = 0

    proto_list = list(protos.values())
    rng.shuffle(proto_list)

    for proto in proto_list:
        if len(out) >= n:
            break
        try:
            struct = proto.get_structure()
        except Exception:
            continue
        roles = _slot_roles(struct)
        slots = sorted(roles.keys())
        choices = []
        for slot in slots:
            pool = pools["anion"] if roles[slot] == "anion" else pools["cation"]
            pool = [p for p in pool if p] or pools["all"]
            choices.append(pool)

        # sample combinations rather than full cartesian product
        made = 0
        attempts = 0
        max_attempts = max_per_prototype * 6
        while made < max_per_prototype and attempts < max_attempts and len(out) < n:
            attempts += 1
            assignment = {slot: rng.choice(choices[i]) for i, slot in enumerate(slots)}
            # require distinct elements across slots
            if len(set(assignment.values())) != len(slots):
                continue
            try:
                new_struct = struct.copy()
                new_struct.replace_species(assignment)
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
            made += 1
            rec = CandidateRecord.from_structure(
                new_struct,
                candidate_id=f"proto::{counter:06d}",
                source="prototype_enum",
                parent_source_id=proto.candidate_id,
                tags=[campaign.name, "prototype_enum",
                      f"proto:{proto.anonymous_formula}",
                      *[t for t in proto.tags if t.startswith("proto:")]],
            )
            out.append(rec)
    return out[:n]
