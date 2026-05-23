"""Semiconductor application scoring: target-gap absorbers, power/optoelectronics.

Interface (shared by all application modules):
    compute_features(record)               -> dict
    score_candidate(record, campaign, ctx) -> ScoreResult
    explain_score(record, score)           -> list[str]

`ctx` is an optional dict carrying cross-cutting precomputed values
(`cohort_min_energy` for the stability fallback). Cross-cutting record fields
(novelty_score, synthesizability_score) are expected to be pre-populated by the
scoring workflow, but the scorer falls back to computing them if absent.

Predicted gaps come from a heuristic surrogate (see properties.bandgap); they
are a first-pass filter only, so confidence is capped accordingly.
"""

from __future__ import annotations

from ..config.schema import CampaignConfig
from ..data.elements import CHALCOGENS, HALOGENS, LONE_PAIR_CATIONS, abundance_score
from ..data.schemas import CandidateRecord, ScoreResult
from .bandgap import predict_bandgap
from .common import assemble_score, target_window_score
from .stability import stability_for_record
from .synthesizability import synthesizability_score

# Main-group / sp-bonded elements giving dispersive bands -> light effective mass.
_LIGHT_MASS_ELEMENTS = {"Si", "Ge", "Sn", "Ga", "In", "Zn", "Al", "B", "C", "P", "N"}
# Heavy d-electron transition metals -> flatter bands, heavier carriers (esp. oxides).
_HEAVY_D_ELEMENTS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo",
    "Hf", "Ta", "W", "Ru", "Rh", "Pd", "Re",
}
_METALLIC_GAP_EV = 0.1


def compute_features(record: CandidateRecord) -> dict:
    els = set(record.elements)

    if record.predicted_bandgap_ev is None:
        gap, direct, model, gap_conf = predict_bandgap(record.get_structure())
    else:
        gap = record.predicted_bandgap_ev
        direct = bool(record.predicted_direct_gap)
        model = record.bandgap_model or "stored"
        gap_conf = 0.4

    is_metallic = gap is not None and gap < _METALLIC_GAP_EV

    anions = els & (CHALCOGENS | HALOGENS)
    has_lone_pair = bool(els & LONE_PAIR_CATIONS)

    # effective-mass proxy: sp-bonded main-group compounds disperse well (light carriers);
    # heavy d-electron oxides tend to flat bands (heavy carriers).
    light = els & _LIGHT_MASS_ELEMENTS
    heavy = els & _HEAVY_D_ELEMENTS
    if heavy and "O" in els:
        eff_mass_proxy = 0.4
    elif light:
        eff_mass_proxy = 0.7 + 0.2 * (len(light) / max(len(els), 1))
    else:
        eff_mass_proxy = 0.5
    eff_mass_proxy = min(eff_mass_proxy, 0.9)

    # defect-tolerance proxy: ns2 lone-pair cation with chalcogen/halide anion
    # gives antibonding VBM -> shallow, benign defects (perovskite-like).
    defect_tol_proxy = 0.8 if (has_lone_pair and anions) else 0.4

    return {
        "predicted_bandgap_ev": gap,
        "is_direct_gap": bool(direct),
        "bandgap_model": model,
        "bandgap_confidence": gap_conf,
        "is_metallic": is_metallic,
        "has_lone_pair_cation": has_lone_pair,
        "lone_pair_cations": sorted(els & LONE_PAIR_CATIONS),
        "effective_mass_proxy": round(eff_mass_proxy, 3),
        "defect_tolerance_proxy": round(defect_tol_proxy, 3),
        "n_anion_types": len(anions),
    }


def score_candidate(record: CandidateRecord, campaign: CampaignConfig, ctx: dict | None = None) -> ScoreResult:
    ctx = ctx or {}
    feats = compute_features(record)

    # a metal is not a semiconductor
    if feats["is_metallic"]:
        return assemble_score(
            {}, campaign,
            rejection_reason=f"metallic (predicted gap {feats['predicted_bandgap_ev']} eV < {_METALLIC_GAP_EV})",
        )

    stability, stab_conf = stability_for_record(record, ctx.get("cohort_min_energy"))

    # target-gap window (general semiconductors 0.5-4.5 eV unless campaign overrides)
    lo, hi = campaign.targets.get("bandgap_ev", [0.5, 4.5])
    bandgap_target = target_window_score(feats["predicted_bandgap_ev"], lo, hi, width=0.5)

    # direct-gap proxy: strong absorption / efficient emission if direct
    direct_gap_proxy = 1.0 if feats["is_direct_gap"] else 0.3

    effective_mass_proxy = feats["effective_mass_proxy"]
    defect_tolerance = feats["defect_tolerance_proxy"]

    synth = record.synthesizability_score or synthesizability_score(record)
    novelty = record.novelty_score
    abundance = abundance_score(record.elements)

    components = {
        "stability": stability,
        "bandgap_target": bandgap_target,
        "direct_gap_proxy": direct_gap_proxy,
        "effective_mass_proxy": effective_mass_proxy,
        "defect_tolerance": defect_tolerance,
        "synthesizability": synth,
        "abundance": abundance,
        "novelty": novelty,
    }
    # bandgap is a heuristic proxy; cap overall confidence by both stability and gap model.
    confidence = min(stab_conf, max(feats["bandgap_confidence"], 0.6))
    res = assemble_score(components, campaign, confidence=confidence)
    res.explanation = explain_score(record, res, feats)
    return res


def explain_score(record: CandidateRecord, score: ScoreResult, feats: dict | None = None) -> list[str]:
    feats = feats or compute_features(record)
    lines = [
        f"Semiconductor candidate {record.reduced_formula} (total={score.total:.3f}, conf={score.confidence:.2f})",
        f"  predicted bandgap: {feats['predicted_bandgap_ev']} eV "
        f"({'direct' if feats['is_direct_gap'] else 'indirect'} guess, model={feats['bandgap_model']})",
        f"  effective-mass proxy: {feats['effective_mass_proxy']:.2f} (higher -> lighter carriers / better transport)",
        f"  defect tolerance: {feats['defect_tolerance_proxy']:.2f} "
        f"(lone-pair cation(s): {feats['lone_pair_cations'] or 'none'})",
        f"  e_above_hull: {record.ml_e_above_hull} eV/atom; uncertainty: {record.uncertainty_score:.3f}",
    ]
    if score.rejection_reason:
        lines.append(f"  REJECTED: {score.rejection_reason}")
    return lines
