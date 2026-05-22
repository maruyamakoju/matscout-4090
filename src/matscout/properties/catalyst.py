"""Catalyst application scoring: electrocatalyst bulk pre-screen (HER/OER/CO2RR).

Interface (shared by all application modules):
    compute_features(record)               -> dict
    score_candidate(record, campaign, ctx) -> ScoreResult
    explain_score(record, score)           -> list[str]

`ctx` is an optional dict carrying cross-cutting precomputed values
(`cohort_min_energy` for the stability fallback). Cross-cutting record fields
(novelty_score, synthesizability_score) are expected to be pre-populated by the
scoring workflow, but the scorer falls back to computing them if absent.

NOTE: a full catalyst workflow needs surface slabs + adsorbate DFT
(OC20/OC22-style). That is out of scope here -- this module is a *bulk*
pre-screen proxy for catalytic promise and is therefore reported at lower
confidence (~0.5). Real activity requires surface + adsorbate calculations.
"""

from __future__ import annotations

from ..config.schema import CampaignConfig
from ..data.elements import CHALCOGENS, REDOX_ACTIVE_TM, abundance_score
from ..data.schemas import CandidateRecord, ScoreResult
from .common import assemble_score, peak_score
from .stability import stability_for_record
from .synthesizability import synthesizability_score

# Active transition metals for catalysis: redox-active TMs plus catalytic noble metals.
_NOBLE_ACTIVE = {"Ru", "Ir", "Pt", "Pd"}
_ACTIVE_METALS = REDOX_ACTIVE_TM | _NOBLE_ACTIVE

# Anions that confer metallic / semi-metallic conduction (good for electrocatalysis).
_CONDUCTIVE_ANIONS = {"S", "Se", "Te", "N", "C", "P"}
# Stable framework cations that resist corrosion / passivate.
_CORROSION_STABLE_CATIONS = {"Ti", "Zr", "Nb", "Ta", "Al"}

# Pre-screen confidence ceiling: real activity needs surface + adsorbate DFT.
_PRESCREEN_CONFIDENCE = 0.5


def compute_features(record: CandidateRecord) -> dict:
    els = set(record.elements)
    active_metals = sorted(els & _ACTIVE_METALS)
    has_oxide = "O" in els
    has_sulfide = "S" in els
    anions = els & (CHALCOGENS | {"N", "P", "C"})
    # pure metal: only active/transition metals, no anion framework
    is_pure_metal = bool(active_metals) and not anions

    return {
        "active_metals": active_metals,
        "n_active_metals": len(active_metals),
        "has_oxide": has_oxide,
        "has_sulfide": has_sulfide,
        "is_pure_metal": is_pure_metal,
        "n_anion_types": len(anions),
        "has_conductive_anion": bool(els & _CONDUCTIVE_ANIONS),
        "has_corrosion_cation": bool(els & _CORROSION_STABLE_CATIONS),
        "reaction_targets": [],  # filled in by score_candidate from campaign.reaction_targets
    }


def score_candidate(record: CandidateRecord, campaign: CampaignConfig, ctx: dict | None = None) -> ScoreResult:
    ctx = ctx or {}
    feats = compute_features(record)
    reaction_targets = list(campaign.reaction_targets)
    feats["reaction_targets"] = reaction_targets

    if feats["n_active_metals"] == 0:
        return assemble_score({}, campaign, rejection_reason="no_active_transition_metal")

    els = set(record.elements)
    n_active = feats["n_active_metals"]

    # surface_stability: stable bulk thermodynamics is a prerequisite for stable surfaces.
    surface_stability, stab_conf = stability_for_record(record, ctx.get("cohort_min_energy"))

    # adsorption_target: Sabatier "moderate binding" proxy.
    #   peaks when 1-2 active metals + O or S anion present (tunable, moderate adsorption);
    #   lower for pure metals (often overbind) and for too many active metals.
    has_anion = feats["has_oxide"] or feats["has_sulfide"]
    if feats["is_pure_metal"]:
        adsorption_target = 0.35  # pure metal -> tends to overbind
    elif has_anion:
        # peak at ~2 active metals (mixed-valence tunability), tolerance of 2 metals
        adsorption_target = 0.85 * peak_score(float(n_active), target=2.0, tolerance=3.0)
        adsorption_target = max(adsorption_target, 0.55)  # any active-metal oxide/sulfide is reasonable
    else:
        adsorption_target = 0.45  # active metal with non-O/S framework -> uncertain binding

    # small reaction-target bonus when bulk chemistry matches the target electrochemistry
    bonus = 0.0
    if "CO2RR" in reaction_targets and "Cu" in els:
        bonus += 0.10  # Cu gives moderate CO binding for CO2RR
    if "OER" in reaction_targets and feats["has_oxide"] and n_active >= 2:
        bonus += 0.08  # mixed-valence TM oxide for OER
    if "HER" in reaction_targets and feats["has_sulfide"]:
        bonus += 0.06  # TM sulfides (e.g. MoS2) for HER
    adsorption_target = min(1.0, adsorption_target + bonus)

    # active_site_diversity: more distinct active metals + anion variety -> more diverse sites.
    if n_active >= 3:
        active_site_diversity = 0.9
    elif n_active == 2:
        active_site_diversity = 0.7
    else:
        active_site_diversity = 0.4
    active_site_diversity = min(1.0, active_site_diversity + 0.05 * max(0, feats["n_anion_types"] - 1))

    # conductivity_proxy: metallic / semi-metallic conduction helps electrocatalysis.
    #   sulfides/nitrides/carbides + partially-filled d TMs score high;
    #   wide-gap insulating oxides score low.
    if feats["is_pure_metal"]:
        conductivity_proxy = 0.9
    elif feats["has_conductive_anion"]:
        conductivity_proxy = 0.75
    elif feats["has_oxide"]:
        # oxides: lower (often insulating), small boost for mixed-valence metallic oxides
        conductivity_proxy = 0.3 + (0.15 if n_active >= 2 else 0.0)
    else:
        conductivity_proxy = 0.5

    # corrosion_stability: oxides + stable framework cations resist corrosion;
    #   sulfides / easily-oxidized lower.
    corrosion_stability = 0.6
    if feats["has_oxide"]:
        corrosion_stability = 0.8
    if feats["has_corrosion_cation"]:
        corrosion_stability = max(corrosion_stability, 0.8)
    if feats["has_sulfide"] and not feats["has_oxide"]:
        corrosion_stability = 0.5

    abundance = abundance_score(record.elements)
    synth = record.synthesizability_score or synthesizability_score(record)
    novelty = record.novelty_score

    components = {
        "surface_stability": surface_stability,
        "adsorption_target": adsorption_target,
        "active_site_diversity": active_site_diversity,
        "conductivity_proxy": conductivity_proxy,
        "corrosion_stability": corrosion_stability,
        "abundance": abundance,
        "synthesizability": synth,
        "novelty": novelty,
    }
    # bulk pre-screen: cap confidence; real activity needs surface + adsorbate DFT
    confidence = min(stab_conf, _PRESCREEN_CONFIDENCE)
    res = assemble_score(components, campaign, confidence=confidence)
    res.explanation = explain_score(record, res, feats)
    return res


def explain_score(record: CandidateRecord, score: ScoreResult, feats: dict | None = None) -> list[str]:
    feats = feats or compute_features(record)
    if feats["is_pure_metal"]:
        framework = "pure metal (often overbinds)"
    elif feats["has_oxide"]:
        framework = "oxide framework"
    elif feats["has_sulfide"]:
        framework = "sulfide framework"
    else:
        framework = "non-O/S framework"
    lines = [
        f"Catalyst candidate {record.reduced_formula} (total={score.total:.3f}, conf={score.confidence:.2f})",
        f"  active transition metal(s): {feats['active_metals']} ({feats['n_active_metals']} distinct)",
        f"  framework: {framework}; reaction targets: {feats.get('reaction_targets', [])}",
        f"  e_above_hull: {record.ml_e_above_hull} eV/atom (bulk surface-stability proxy)",
        "  BULK PRE-SCREEN ONLY: confidence is capped; real activity requires "
        "surface slab + adsorbate DFT (OC20/OC22-style) as the next step.",
    ]
    if score.rejection_reason:
        lines.append(f"  REJECTED: {score.rejection_reason}")
    return lines
