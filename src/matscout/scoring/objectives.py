"""Objective routing + cross-cutting scoring.

Computes cross-cutting fields (novelty, synthesizability, toxicity/scarcity/cost
penalties) on each record, routes to the campaign's application scorer, and combines
into final_score.
"""

from __future__ import annotations

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.elements import cost_penalty, scarcity_penalty, toxicity_penalty
from ..data.schemas import CandidateRecord, ScoreResult
from ..properties import battery, catalyst, co2_capture, semiconductor, solar
from ..properties.bandgap import predict_bandgap
from ..properties.synthesizability import synthesizability_score
from ..validation.novelty import NoveltyIndex

APP_SCORERS = {
    "battery": battery,
    "semiconductor": semiconductor,
    "catalyst": catalyst,
    "solar": solar,
    "co2_capture": co2_capture,
}

SCORE_FIELD = {
    "battery": "battery_score",
    "semiconductor": "semiconductor_score",
    "catalyst": "catalyst_score",
    "solar": "solar_score",
    "co2_capture": "co2_capture_score",
}


def cohort_min_energy(records: list[CandidateRecord]) -> dict[str, float]:
    """Lowest ML energy/atom per anonymous formula (for the stability fallback)."""
    out: dict[str, float] = {}
    for r in records:
        if r.ml_energy_per_atom is None:
            continue
        key = r.anonymous_formula
        if key not in out or r.ml_energy_per_atom < out[key]:
            out[key] = r.ml_energy_per_atom
    return out


def final_score(app_score: float, record: CandidateRecord) -> float:
    """Combine application score with cross-cutting penalties.

    Penalties shrink the score multiplicatively; uncertainty is intentionally NOT
    penalized here (high-uncertainty/high-score candidates are surfaced as wildcards).
    """
    penalty_factor = 1.0 - 0.5 * record.toxicity_penalty - 0.25 * record.cost_penalty
    penalty_factor = max(0.0, min(penalty_factor, 1.0))
    return round(app_score * penalty_factor, 4)


def score_records(
    records: list[CandidateRecord],
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    novelty_index: NoveltyIndex | None = None,
    descriptors: dict[str, dict] | None = None,
) -> list[CandidateRecord]:
    """Score all records for a campaign in place; returns the same list."""
    module = APP_SCORERS.get(campaign.name)
    if module is None:
        raise ValueError(f"No application scorer for campaign '{campaign.name}'")
    field = SCORE_FIELD[campaign.name]
    cohort = cohort_min_energy(records)
    descriptors = descriptors or {}

    for r in records:
        if r.rejection_reason and r.rejection_reason.startswith(("post_relax", "relax_failed",
                                                                  "not_converged", "huge_volume",
                                                                  "e_above_hull")):
            # already killed in relaxation; leave final_score at 0
            r.final_score = 0.0
            continue
        # cross-cutting fields
        r.toxicity_penalty = round(toxicity_penalty(r.elements), 4)
        r.scarcity_penalty = round(scarcity_penalty(r.elements), 4)
        r.cost_penalty = round(cost_penalty(r.elements), 4)
        if not r.synthesizability_score:
            r.synthesizability_score = synthesizability_score(r)
        if novelty_index is not None and not r.novelty_score:
            r.novelty_score = novelty_index.score(r)
        # persist a predicted bandgap on the record (used by reports/metadata/parquet)
        if r.predicted_bandgap_ev is None:
            try:
                gap, direct, model_name, _ = predict_bandgap(r.get_structure())
                r.predicted_bandgap_ev = round(gap, 3)
                r.predicted_direct_gap = direct
                r.bandgap_model = model_name
            except Exception:
                pass

        ctx = {"cohort_min_energy": cohort}
        if campaign.name == "co2_capture":
            ctx["descriptors"] = descriptors.get(r.candidate_id, {})

        result: ScoreResult = module.score_candidate(r, campaign, ctx)
        setattr(r, field, result.total)
        r.uncertainty_score = round(r.uncertainty_score, 4)
        if result.rejection_reason:
            r.rejection_reason = r.rejection_reason or result.rejection_reason
            r.final_score = 0.0
        else:
            r.final_score = final_score(result.total, r)
    return records
