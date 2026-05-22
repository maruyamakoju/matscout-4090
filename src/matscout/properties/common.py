"""Shared helpers for application scoring modules.

Each application module exposes the same interface:
    compute_features(record)            -> dict[str, float|bool|...]
    score_candidate(record, campaign)   -> ScoreResult
    explain_score(record, score)        -> list[str]

`assemble_score` combines weighted components (from campaign.objectives) into a
ScoreResult with a human-readable breakdown.
"""

from __future__ import annotations

from ..config.schema import CampaignConfig
from ..data.schemas import CandidateRecord, ScoreResult


def assemble_score(
    components: dict[str, float],
    campaign: CampaignConfig,
    *,
    rejection_reason: str | None = None,
    confidence: float = 1.0,
    extra_explanations: list[str] | None = None,
) -> ScoreResult:
    """Weighted-sum of component scores using campaign objective weights.

    Components not present in campaign.objectives default to weight 0 (ignored);
    objectives with no matching component contribute 0.
    """
    weights = campaign.normalized_objectives()
    total = 0.0
    used_weight = 0.0
    explanation: list[str] = list(extra_explanations or [])
    for name, w in weights.items():
        comp = components.get(name)
        if comp is None:
            continue
        total += w * comp
        used_weight += w
        explanation.append(f"{name}: {comp:.3f} x w{w:.2f} = {w * comp:.3f}")
    # renormalize if some objective components were unavailable
    if 0 < used_weight < 0.999:
        total = total / used_weight
        explanation.append(f"(renormalized over available weight {used_weight:.2f})")
    if rejection_reason:
        total = 0.0
    return ScoreResult(
        total=round(total, 4),
        components={k: round(v, 4) for k, v in components.items()},
        rejection_reason=rejection_reason,
        explanation=explanation,
        confidence=confidence,
    )


def target_window_score(value: float | None, lo: float, hi: float, width: float = 0.5) -> float:
    """1.0 inside [lo, hi], decaying smoothly outside over `width` units. None -> 0."""
    if value is None:
        return 0.0
    if lo <= value <= hi:
        return 1.0
    dist = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - (dist / width))


def peak_score(value: float | None, target: float, tolerance: float) -> float:
    """1.0 at target, linearly decaying to 0 at +/- tolerance. None -> 0."""
    if value is None:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def element_fraction(record: CandidateRecord, elements: set[str]) -> float:
    """Atomic fraction of sites occupied by the given elements."""
    try:
        comp = record.get_structure().composition
    except Exception:
        return 0.0
    total = comp.num_atoms
    if total == 0:
        return 0.0
    return sum(amt for el, amt in comp.get_el_amt_dict().items() if el in elements) / total
