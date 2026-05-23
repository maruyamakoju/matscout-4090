"""Solar absorber scoring: thin-film photovoltaic absorbers (chalcogenides, halides, pnictides).

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
from ..data.elements import CHALCOGENS, HALOGENS, LONE_PAIR_CATIONS, abundance_score
from ..data.schemas import CandidateRecord, ScoreResult
from .bandgap import predict_bandgap
from .common import assemble_score, target_window_score
from .stability import stability_for_record
from .synthesizability import synthesizability_score

# Anions that give strong optical absorption / soft, defect-tolerant lattices.
_ABSORBER_ANIONS = (CHALCOGENS - {"O"}) | HALOGENS  # S, Se, Te, F, Cl, Br, I
# Heavy main toxicity flag for Pb-free differentiation.
_PB = "Pb"


def compute_features(record: CandidateRecord) -> dict:
    els = set(record.elements)

    if record.predicted_bandgap_ev is None:
        gap, direct, model, _ = predict_bandgap(record.get_structure())
    else:
        gap = record.predicted_bandgap_ev
        direct = bool(record.predicted_direct_gap)
        model = record.bandgap_model or "stored"

    lone_pair = els & LONE_PAIR_CATIONS
    soft_anions = els & _ABSORBER_ANIONS
    has_lone_pair = bool(lone_pair)
    has_soft_anion = bool(soft_anions)

    # absorption proxy: lone-pair cation + chalcogenide/halide -> strong, direct-ish absorption
    if has_lone_pair and has_soft_anion:
        absorption = 0.85
    elif has_soft_anion:
        absorption = 0.55
    else:
        absorption = 0.5

    # defect tolerance: ns2 lone-pair chemistry (soft lattice, antibonding VBM)
    if has_lone_pair and has_soft_anion:
        defect_tol = 0.85
    else:
        defect_tol = 0.4

    return {
        "predicted_bandgap_ev": gap,
        "is_direct_gap": bool(direct),
        "has_lone_pair_cation": has_lone_pair,
        "lone_pair_cations": sorted(lone_pair),
        "soft_anions": sorted(soft_anions),
        "absorption_proxy": round(absorption, 3),
        "defect_tolerance_proxy": round(defect_tol, 3),
        "is_pb_free": _PB not in els,
        "bandgap_model": model,
    }


def score_candidate(record: CandidateRecord, campaign: CampaignConfig, ctx: dict | None = None) -> ScoreResult:
    ctx = ctx or {}
    feats = compute_features(record)

    gap = feats["predicted_bandgap_ev"]

    # rejection: metallic / too-small or too-wide for a visible-light absorber
    if gap is None:
        return assemble_score({}, campaign, rejection_reason="no_bandgap_prediction")
    if gap < 0.3:
        return assemble_score({}, campaign, rejection_reason=f"bandgap {gap:.2f} eV too small (metallic)")
    if gap > 3.0:
        return assemble_score({}, campaign, rejection_reason=f"bandgap {gap:.2f} eV too wide (no visible absorption)")

    stability, stab_conf = stability_for_record(record, ctx.get("cohort_min_energy"))

    # bandgap window depends on the target architecture
    mode = campaign.targets.get("mode", "single_junction")
    _windows = {
        "single_junction": campaign.targets.get("bandgap_single_junction", [1.1, 1.6]),
        "tandem": campaign.targets.get("bandgap_tandem", [1.6, 1.9]),
        "indoor_pv": campaign.targets.get("bandgap_indoor_pv", [1.8, 2.2]),
    }
    lo, hi = _windows.get(mode, _windows["single_junction"])
    bandgap = target_window_score(gap, lo, hi, width=0.3)

    # direct gap -> strong absorption; indirect penalized
    direct_gap_proxy = 1.0 if feats["is_direct_gap"] else 0.3

    absorption = feats["absorption_proxy"]
    defect_tolerance = feats["defect_tolerance_proxy"]

    synth = record.synthesizability_score or synthesizability_score(record)
    novelty = record.novelty_score
    abundance = abundance_score(record.elements)

    components = {
        "bandgap": bandgap,
        "stability": stability,
        "direct_gap_proxy": direct_gap_proxy,
        "absorption_proxy": absorption,
        "defect_tolerance": defect_tolerance,
        "abundance": abundance,
        "synthesizability": synth,
        "novelty": novelty,
    }
    # bandgap here is a first-pass heuristic; cap confidence accordingly
    confidence = min(stab_conf, 0.6)
    res = assemble_score(components, campaign, confidence=confidence)
    res.explanation = explain_score(record, res, feats)
    return res


def explain_score(record: CandidateRecord, score: ScoreResult, feats: dict | None = None) -> list[str]:
    feats = feats or compute_features(record)
    lines = [
        f"Solar absorber {record.reduced_formula} (total={score.total:.3f}, conf={score.confidence:.2f})",
        f"  predicted bandgap: {feats['predicted_bandgap_ev']} eV "
        f"({'direct' if feats['is_direct_gap'] else 'indirect'} gap)",
        f"  lone-pair cation(s): {feats['lone_pair_cations']}; soft anion(s): {feats['soft_anions']}",
        f"  absorption proxy: {feats['absorption_proxy']:.2f}; defect tolerance: {feats['defect_tolerance_proxy']:.2f}",
        f"  Pb-free: {feats['is_pb_free']}; e_above_hull: {record.ml_e_above_hull} eV/atom",
    ]
    if score.rejection_reason:
        lines.append(f"  REJECTED: {score.rejection_reason}")
    return lines
