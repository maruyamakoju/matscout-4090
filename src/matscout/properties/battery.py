"""Battery application scoring: solid electrolytes, cathodes, multivalent conductors.

Interface (shared by all application modules):
    compute_features(record)               -> dict
    score_candidate(record, campaign, ctx) -> ScoreResult
    explain_score(record, score)           -> list[str]

`ctx` is an optional dict carrying cross-cutting precomputed values
(`cohort_min_energy` for the stability fallback). Cross-cutting record fields
(novelty_score, synthesizability_score) are expected to be pre-populated by the
scoring workflow, but the scorer falls back to computing them if absent.
"""

from __future__ import annotations

from ..config.schema import CampaignConfig
from ..data.elements import CHALCOGENS, HALOGENS, REDOX_ACTIVE_TM, abundance_score
from ..data.schemas import CandidateRecord, ScoreResult
from .bandgap import predict_bandgap
from .common import assemble_score, element_fraction, target_window_score
from .stability import stability_for_record
from .synthesizability import synthesizability_score

_BATTERY_MOBILE = {"Li", "Na", "Mg", "Zn"}
_SOFT_ANIONS = CHALCOGENS - {"O"} | HALOGENS  # S, Se, Te, F, Cl, Br, I


def compute_features(record: CandidateRecord) -> dict:
    els = set(record.elements)
    mobile_present = els & _BATTERY_MOBILE
    mobile_frac = element_fraction(record, _BATTERY_MOBILE)

    if record.predicted_bandgap_ev is None:
        gap, _direct, _model, _ = predict_bandgap(record.get_structure())
    else:
        gap = record.predicted_bandgap_ev

    anions = els & (CHALCOGENS | HALOGENS | {"N", "P"})
    soft = els & _SOFT_ANIONS
    anion_softness = (len(soft) / len(anions)) if anions else 0.0

    return {
        "has_mobile_ion": bool(mobile_present),
        "mobile_ions": sorted(mobile_present),
        "mobile_ion_fraction": round(mobile_frac, 4),
        "predicted_bandgap_ev": gap,
        "anion_softness": round(anion_softness, 3),
        "has_redox_tm": bool(els & REDOX_ACTIVE_TM),
        "is_oxide": "O" in els,
        "n_anion_types": len(anions),
    }


def score_candidate(record: CandidateRecord, campaign: CampaignConfig, ctx: dict | None = None) -> ScoreResult:
    ctx = ctx or {}
    feats = compute_features(record)

    if not feats["has_mobile_ion"]:
        return assemble_score({}, campaign, rejection_reason="no_mobile_ion (Li/Na/Mg/Zn)")

    stability, stab_conf = stability_for_record(record, ctx.get("cohort_min_energy"))

    # mobile-ion fraction: prefer 0.15-0.45 (percolating but not framework-poor)
    mobile_ion = target_window_score(feats["mobile_ion_fraction"], 0.15, 0.45, width=0.2)

    # migration proxy: soft anions lower diffusion barriers; oxides give stability not mobility
    migration_proxy = 0.35 + 0.5 * feats["anion_softness"]
    migration_proxy = min(migration_proxy, 1.0)

    # electrochemical-window proxy:
    #   electrolyte wants wide gap (>2.5 eV insulating); cathode wants redox TM (lower gap ok)
    gap = feats["predicted_bandgap_ev"]
    if feats["has_redox_tm"]:
        window = 0.7  # cathode-type: redox activity present, moderate window
    else:
        window = min(1.0, gap / 3.0) if gap is not None else 0.3  # electrolyte-type

    synth = record.synthesizability_score or synthesizability_score(record)
    novelty = record.novelty_score
    abundance = abundance_score(record.elements)

    components = {
        "stability": stability,
        "mobile_ion": mobile_ion,
        "migration_proxy": migration_proxy,
        "electrochemical_window": window,
        "synthesizability": synth,
        "novelty": novelty,
        "abundance": abundance,
    }
    confidence = min(stab_conf, 1.0)
    res = assemble_score(components, campaign, confidence=confidence)
    res.explanation = explain_score(record, res, feats)
    return res


def explain_score(record: CandidateRecord, score: ScoreResult, feats: dict | None = None) -> list[str]:
    feats = feats or compute_features(record)
    lines = [
        f"Battery candidate {record.reduced_formula} (total={score.total:.3f}, conf={score.confidence:.2f})",
        f"  mobile ion(s): {feats['mobile_ions']} at fraction {feats['mobile_ion_fraction']:.2f}",
        f"  anion softness: {feats['anion_softness']:.2f} (soft -> lower migration barrier)",
        f"  predicted bandgap: {feats['predicted_bandgap_ev']} eV "
        f"({'electrolyte-type' if not feats['has_redox_tm'] else 'cathode-type (redox TM present)'})",
        f"  e_above_hull: {record.ml_e_above_hull} eV/atom; uncertainty: {record.uncertainty_score:.3f}",
    ]
    if score.rejection_reason:
        lines.append(f"  REJECTED: {score.rejection_reason}")
    return lines
