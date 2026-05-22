"""Multi-objective Pareto front computation."""

from __future__ import annotations

import numpy as np

from ..data.schemas import CandidateRecord


def _objective_matrix(records: list[CandidateRecord], objectives: list[str]) -> np.ndarray:
    """Build a matrix where every column is to be MAXIMIZED.

    Recognized objective keys: final_score, novelty_score, synthesizability_score,
    stability (via -ml_e_above_hull), low_uncertainty (via -uncertainty_score),
    low_cost (via -cost_penalty).
    """
    rows = []
    for r in records:
        row = []
        for obj in objectives:
            if obj == "stability":
                eh = r.ml_e_above_hull if r.ml_e_above_hull is not None else 0.2
                row.append(-eh)
            elif obj == "low_uncertainty":
                row.append(-r.uncertainty_score)
            elif obj == "low_cost":
                row.append(-r.cost_penalty)
            else:
                row.append(float(getattr(r, obj, 0.0) or 0.0))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def pareto_front_indices(matrix: np.ndarray) -> list[int]:
    """Return indices of non-dominated rows (all columns maximized)."""
    n = matrix.shape[0]
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        # j dominates i if j >= i in all and > in at least one
        dominates = np.all(matrix >= matrix[i], axis=1) & np.any(matrix > matrix[i], axis=1)
        if np.any(dominates):
            is_front[i] = False
    return [i for i in range(n) if is_front[i]]


def pareto_rank(records: list[CandidateRecord], objectives: list[str]) -> list[int]:
    """Assign a Pareto rank (0 = first front) to each record by peeling fronts."""
    matrix = _objective_matrix(records, objectives)
    ranks = [-1] * len(records)
    remaining = list(range(len(records)))
    rank = 0
    while remaining:
        sub = matrix[remaining]
        front_local = pareto_front_indices(sub)
        front_global = [remaining[i] for i in front_local]
        for g in front_global:
            ranks[g] = rank
        remaining = [r for r in remaining if r not in set(front_global)]
        rank += 1
        if rank > len(records):  # safety
            break
    return ranks
