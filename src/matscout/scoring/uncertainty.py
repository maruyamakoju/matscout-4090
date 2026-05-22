"""Uncertainty handling + active-learning acquisition (technical spec §8.2)."""

from __future__ import annotations

from ..data.schemas import CandidateRecord

HIGH_UNCERTAINTY_ENERGY = 0.08  # eV/atom std across models
MODEL_FAILURE_UNCERTAINTY = 0.30  # above this, likely a model failure, not exploration


def uncertainty_bonus(uncertainty: float) -> float:
    """Moderate uncertainty is good for exploration; too low is boring, too high is failure."""
    if uncertainty <= 0.0:
        return 0.2
    if uncertainty < 0.02:
        return 0.3  # boring, known-like
    if uncertainty <= HIGH_UNCERTAINTY_ENERGY:
        return 1.0  # sweet spot
    if uncertainty <= MODEL_FAILURE_UNCERTAINTY:
        return 0.5  # interesting but risky
    return 0.1  # likely model failure


def acquisition_score(record: CandidateRecord, app_score: float, diversity: float = 0.5) -> float:
    """Active-learning acquisition (spec §8.2)."""
    from ..data.elements import abundance_score

    return round(
        0.35 * app_score
        + 0.20 * record.novelty_score
        + 0.15 * diversity
        + 0.15 * uncertainty_bonus(record.uncertainty_score)
        + 0.10 * record.synthesizability_score
        + 0.05 * abundance_score(record.elements),
        4,
    )


def is_wildcard(record: CandidateRecord) -> bool:
    """High novelty + moderate stability + decent score + high-but-not-failure uncertainty."""
    eh = record.ml_e_above_hull
    moderate_stability = eh is None or (0.02 <= eh <= 0.12)
    high_novelty = record.novelty_score >= 0.6
    interesting_uncertainty = (
        HIGH_UNCERTAINTY_ENERGY * 0.6 <= record.uncertainty_score <= MODEL_FAILURE_UNCERTAINTY
    )
    decent_score = record.final_score >= 0.4
    return high_novelty and decent_score and (moderate_stability or interesting_uncertainty)
