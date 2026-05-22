"""Synthesizability proxy (technical spec §6.3).

No robotic lab here, so we approximate synthesizability with composition/structure
heuristics: known prototype, simple stoichiometry, common oxidation states, low element
count, no volatile/exotic species.
"""

from __future__ import annotations

from pymatgen.core import Composition

from ..data.elements import EXCLUDE_ELEMENTS
from ..data.schemas import CandidateRecord

VOLATILE_OR_HARD = {"As", "Hg", "Cd", "Tl", "Os", "Se", "Te"}
KNOWN_PROTO_TAG_PREFIX = "proto:"
COMMON_PROTO_FAMILIES = {
    "rocksalt", "perovskite", "spinel", "rutile", "fluorite", "antifluorite",
    "wurtzite", "zincblende", "delafossite", "olivine", "garnet",
}


def synthesizability_score(record: CandidateRecord) -> float:
    score = 0.5  # neutral baseline
    elements = record.elements
    n_el = len(elements)

    # element count: simpler is easier
    if n_el <= 2:
        score += 0.15
    elif n_el == 3:
        score += 0.10
    elif n_el == 4:
        score += 0.0
    elif n_el == 5:
        score -= 0.10
    else:
        score -= 0.25

    # known structural prototype bonus
    if any(t.startswith(KNOWN_PROTO_TAG_PREFIX) and t.split(":", 1)[1] in COMMON_PROTO_FAMILIES
           for t in record.tags):
        score += 0.15

    # charge-balanced / common oxidation states
    try:
        comp = Composition(record.reduced_formula)
        if comp.oxi_state_guesses(max_sites=-20):
            score += 0.10
    except Exception:
        pass

    # simple stoichiometry (small integer ratios)
    try:
        amounts = list(Composition(record.reduced_formula).get_el_amt_dict().values())
        if all(a <= 6 for a in amounts):
            score += 0.05
    except Exception:
        pass

    # penalties for volatile / hard-to-handle species
    if any(e in VOLATILE_OR_HARD for e in elements):
        score -= 0.15
    if any(e in EXCLUDE_ELEMENTS for e in elements):
        score -= 0.4

    return round(min(max(score, 0.0), 1.0), 4)
