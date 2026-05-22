"""Deterministic ranking, diversity selection, and wildcard selection."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition, Element

from ..config.schema import CampaignConfig
from ..data.schemas import CandidateRecord
from .pareto import pareto_rank
from .uncertainty import is_wildcard

# default Pareto objectives per campaign (all maximized)
_DEFAULT_OBJECTIVES = ["final_score", "novelty_score", "stability", "synthesizability_score"]


def _sort_key(r: CandidateRecord):
    # deterministic: primary final_score desc, tie-break candidate_id asc
    return (-(r.final_score or 0.0), r.candidate_id)


def alive(records: list[CandidateRecord]) -> list[CandidateRecord]:
    return [r for r in records if not r.rejection_reason]


def rank_records(
    records: list[CandidateRecord],
    campaign: CampaignConfig,
    objectives: list[str] | None = None,
) -> list[CandidateRecord]:
    """Sort surviving records by final_score (deterministic) and tag with Pareto rank."""
    survivors = alive(records)
    objectives = objectives or _DEFAULT_OBJECTIVES
    if survivors:
        ranks = pareto_rank(survivors, objectives)
        for r, pr in zip(survivors, ranks):
            r.tags = [t for t in r.tags if not t.startswith("pareto_rank:")]
            r.tags.append(f"pareto_rank:{pr}")
    return sorted(survivors, key=_sort_key)


def top_k(ranked: list[CandidateRecord], k: int) -> list[CandidateRecord]:
    return ranked[:k]


def dedup_by_formula(records: list[CandidateRecord]) -> list[CandidateRecord]:
    """Keep the best-scoring representative per reduced formula, preserving order.

    Used for presentation (shortlists, top-200) so a single composition's polymorphs
    don't crowd out chemical diversity. The full parquet keeps every polymorph.
    """
    seen: set[str] = set()
    out: list[CandidateRecord] = []
    for r in sorted(records, key=_sort_key):
        if r.reduced_formula in seen:
            continue
        seen.add(r.reduced_formula)
        out.append(r)
    return out


def _composition_vector(record: CandidateRecord, n_elements: int = 92) -> np.ndarray:
    vec = np.zeros(n_elements + 2, dtype=float)
    try:
        comp = Composition(record.reduced_formula)
        total = comp.num_atoms
        for el, amt in comp.get_el_amt_dict().items():
            z = Element(el).Z
            if 1 <= z <= n_elements:
                vec[z - 1] = amt / total
    except Exception:
        pass
    vec[n_elements] = (record.ml_e_above_hull or 0.1) * 5.0
    vec[n_elements + 1] = min(record.nsites, 80) / 80.0
    return vec


def diversity_top_k(
    records: list[CandidateRecord], k: int, seed: int = 42
) -> list[CandidateRecord]:
    """Cluster by composition and take the best per cluster for a diverse shortlist."""
    survivors = sorted(alive(records), key=_sort_key)
    if len(survivors) <= k:
        return survivors
    try:
        from sklearn.cluster import KMeans

        X = np.vstack([_composition_vector(r) for r in survivors])
        n_clusters = min(k, len(survivors))
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
        labels = km.fit_predict(X)
    except Exception:
        return survivors[:k]

    best_per_cluster: dict[int, CandidateRecord] = {}
    for r, lab in zip(survivors, labels):
        cur = best_per_cluster.get(lab)
        if cur is None or _sort_key(r) < _sort_key(cur):
            best_per_cluster[lab] = r
    chosen = sorted(best_per_cluster.values(), key=_sort_key)
    # backfill from remaining top-ranked if clusters < k
    if len(chosen) < k:
        chosen_ids = {r.candidate_id for r in chosen}
        for r in survivors:
            if r.candidate_id not in chosen_ids:
                chosen.append(r)
                if len(chosen) >= k:
                    break
        chosen = sorted(chosen, key=_sort_key)
    return chosen[:k]


def select_wildcards(records: list[CandidateRecord], k: int = 20) -> list[CandidateRecord]:
    cands = [r for r in alive(records) if is_wildcard(r)]
    # rank wildcards by novelty then score
    cands.sort(key=lambda r: (-(r.novelty_score or 0.0), -(r.final_score or 0.0), r.candidate_id))
    return cands[:k]
