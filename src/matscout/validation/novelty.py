"""Novelty scoring against a reference set of known materials.

novelty_score = 0.35*composition + 0.35*structure + 0.20*application + 0.10*combo
(see technical spec §6.2). The reference index is built from the campaign seeds
(Materials Project or fixtures); with a real MP key this is a meaningful "is it
already known" check. Structure matching is restricted to same-composition groups
for speed.
"""

from __future__ import annotations

import itertools
from collections import Counter

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from ..data.schemas import CandidateRecord

# anonymous formula + space group combos considered "common / well-trodden"
COMMON_PROTOTYPES: set[str] = {
    "AB|225", "AB|216", "AB|221", "ABC3|221", "AB2|225", "AB2|136",
    "A|227", "AB2C4|227", "AB|186", "AB|194",
}


class NoveltyIndex:
    def __init__(self) -> None:
        self.known_formulas: set[str] = set()
        self.known_proto: Counter[str] = Counter()
        self.known_pairs: set[frozenset[str]] = set()
        self._structs_by_formula: dict[str, list[Structure]] = {}
        self._matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)

    @classmethod
    def from_records(cls, records: list[CandidateRecord], with_structures: bool = True) -> NoveltyIndex:
        idx = cls()
        for r in records:
            idx.known_formulas.add(r.reduced_formula)
            idx.known_proto[f"{r.anonymous_formula}|{r.spacegroup_number}"] += 1
            for pair in itertools.combinations(sorted(r.elements), 2):
                idx.known_pairs.add(frozenset(pair))
            if with_structures:
                try:
                    idx._structs_by_formula.setdefault(r.reduced_formula, []).append(r.get_structure())
                except Exception:
                    pass
        return idx

    def composition_novelty(self, reduced_formula: str) -> float:
        return 0.0 if reduced_formula in self.known_formulas else 1.0

    def structure_novelty(self, record: CandidateRecord) -> float:
        # unknown composition -> structurally novel by definition
        knowns = self._structs_by_formula.get(record.reduced_formula)
        if not knowns:
            return 1.0
        try:
            cand = record.get_structure()
        except Exception:
            return 0.5
        for known in knowns:
            try:
                if self._matcher.fit(cand, known):
                    return 0.0
            except Exception:
                continue
        return 1.0

    def prototype_penalty(self, record: CandidateRecord) -> float:
        key = f"{record.anonymous_formula}|{record.spacegroup_number}"
        if key in COMMON_PROTOTYPES or self.known_proto.get(key, 0) >= 5:
            return 1.0  # very common prototype
        return 0.0

    def underexplored_combo(self, record: CandidateRecord) -> float:
        pairs = list(itertools.combinations(sorted(record.elements), 2))
        if not pairs:
            return 0.0
        unseen = sum(1 for p in pairs if frozenset(p) not in self.known_pairs)
        return unseen / len(pairs)

    def score(self, record: CandidateRecord, application_novelty: float = 0.0) -> float:
        comp = self.composition_novelty(record.reduced_formula)
        struct = self.structure_novelty(record)
        combo = self.underexplored_combo(record)
        proto_pen = self.prototype_penalty(record)
        novelty = (
            0.35 * comp
            + 0.35 * struct
            + 0.20 * application_novelty
            + 0.10 * combo
        )
        # mild penalty for very common prototype (keeps "near-novel" practical materials high)
        novelty *= (1.0 - 0.15 * proto_pen)
        return round(min(max(novelty, 0.0), 1.0), 4)
