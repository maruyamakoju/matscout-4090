"""CO2 capture scoring: porous MOFs/COFs for direct-air / flue-gas capture.

Interface (shared by all application modules):
    compute_features(record)               -> dict
    score_candidate(record, campaign, ctx) -> ScoreResult
    explain_score(record, score)           -> list[str]

`ctx` is an optional dict carrying cross-cutting precomputed values
(`cohort_min_energy` for the stability fallback). Cross-cutting record fields
(novelty_score, synthesizability_score) are expected to be pre-populated by the
scoring workflow, but the scorer falls back to computing them if absent.

CO2 capture targets porous frameworks and requires PORE DESCRIPTORS (PLD, LCD,
void fraction, surface area, density, open metal sites, amine functionality).
These are not computable from a generic bulk crystal here; they are expected to
be carried on the record (via `ctx["descriptors"]` or `record.tags`). When they
are absent the candidate is not rejected outright but receives LOW confidence and
neutral defaults, with an explanation that pore descriptors are required (next
step: screen against CoRE MOF / QMOF or run Zeo++).
"""

from __future__ import annotations

from ..config.schema import CampaignConfig
from ..data.schemas import CandidateRecord, ScoreResult
from .common import assemble_score, target_window_score
from .stability import stability_for_record
from .synthesizability import synthesizability_score

# Filter defaults (mirrors configs/campaigns/co2_capture.yaml filters).
_DEFAULT_MIN_PLD = 3.0
_DEFAULT_MAX_PLD = 8.0
_DEFAULT_MIN_VOID = 0.25

# CO2 kinetic diameter ~3.3 A; narrow apertures near this favor CO2/N2 selectivity.
_NARROW_PLD_HI = 4.0

# Water-stable framework node metals (Zr/Al/Ti/Fe carboxylates) vs. less stable.
_WATER_STABLE_METALS = {"Zr", "Al", "Ti", "Fe"}
_WATER_LABILE_METALS = {"Mg", "Ca", "Zn"}

# Neutral defaults used when pore descriptors are unavailable.
_NEUTRAL_PLD = 5.0
_NEUTRAL_LCD = 7.0
_NEUTRAL_VOID = 0.0
_NEUTRAL_ASA = 0.0
_NEUTRAL_DENSITY = 1.0


def _parse_tag_descriptors(record: CandidateRecord) -> dict:
    """Parse descriptor tags of the form 'pld:3.5', 'oms:1', 'amine:1' from record.tags."""
    out: dict = {}
    aliases = {
        "pld": "pld_a",
        "lcd": "lcd_a",
        "void": "void_fraction",
        "asa": "surface_area_m2_g",
        "density": "density",
        "oms": "open_metal_site",
        "amine": "amine_functional",
    }
    for tag in record.tags:
        if ":" not in tag:
            continue
        key, _, raw = tag.partition(":")
        key = key.strip().lower()
        if key not in aliases:
            continue
        try:
            val = float(raw.strip())
        except (ValueError, TypeError):
            continue
        out[aliases[key]] = val
    return out


def compute_features(record: CandidateRecord, ctx: dict | None = None) -> dict:
    ctx = ctx or {}
    els = set(record.elements)

    # descriptors come from the record: ctx["descriptors"] first, then parsed tags.
    desc = dict(_parse_tag_descriptors(record))
    desc.update(ctx.get("descriptors") or {})  # ctx takes precedence over tags

    pore_keys = ("pld_a", "lcd_a", "void_fraction", "surface_area_m2_g", "density")
    has_pore_descriptors = any(desc.get(k) is not None for k in pore_keys)

    pld = desc.get("pld_a", _NEUTRAL_PLD if not has_pore_descriptors else None)
    lcd = desc.get("lcd_a", _NEUTRAL_LCD if not has_pore_descriptors else None)
    void = desc.get("void_fraction", _NEUTRAL_VOID if not has_pore_descriptors else 0.0)
    asa = desc.get("surface_area_m2_g", _NEUTRAL_ASA if not has_pore_descriptors else 0.0)
    density = desc.get("density", _NEUTRAL_DENSITY if not has_pore_descriptors else None)

    open_metal_site = bool(desc.get("open_metal_site", 0))
    amine_functional = bool(desc.get("amine_functional", 0))

    node_metals = sorted((els & _WATER_STABLE_METALS) | (els & _WATER_LABILE_METALS))

    return {
        "pld_a": pld,
        "lcd_a": lcd,
        "void_fraction": void if void is not None else 0.0,
        "surface_area_m2_g": asa if asa is not None else 0.0,
        "density": density,
        "open_metal_site": open_metal_site,
        "amine_functional": amine_functional,
        "has_pore_descriptors": has_pore_descriptors,
        "node_metals": node_metals,
    }


def score_candidate(record: CandidateRecord, campaign: CampaignConfig, ctx: dict | None = None) -> ScoreResult:
    ctx = ctx or {}
    feats = compute_features(record, ctx)

    filt = campaign.filters
    min_pld = filt.min_pld_a if filt.min_pld_a is not None else _DEFAULT_MIN_PLD
    max_pld = filt.max_pld_a if filt.max_pld_a is not None else _DEFAULT_MAX_PLD
    min_void = filt.min_void_fraction if filt.min_void_fraction is not None else _DEFAULT_MIN_VOID

    stability, stab_conf = stability_for_record(record, ctx.get("cohort_min_energy"))

    pld = feats["pld_a"]
    void = feats["void_fraction"]
    oms = feats["open_metal_site"]
    amine = feats["amine_functional"]

    # pore_geometry: PLD inside [min_pld, max_pld] + void fraction ramp above min_void.
    pld_score = target_window_score(pld, min_pld, max_pld, width=2.0)
    void_score = min(1.0, max(0.0, void / min_void)) if min_void > 0 else (1.0 if void > 0 else 0.0)
    pore_geometry = 0.6 * pld_score + 0.4 * void_score

    # narrow-aperture bonus: PLD near CO2 kinetic diameter (~3.3 A) favors selectivity.
    narrow = (pld is not None and min_pld <= pld <= _NARROW_PLD_HI)
    has_polar_site = oms or amine

    # co2_affinity: open metal sites / amines raise CO2 affinity (moderate-to-strong).
    if has_polar_site:
        co2_affinity = 0.85
    else:
        co2_affinity = 0.55

    # selectivity (CO2/N2): narrow pores + polar sites both increase it.
    selectivity = 0.4
    if narrow:
        selectivity += 0.3
    if has_polar_site:
        selectivity += 0.3
    selectivity = min(selectivity, 1.0)

    # water_tolerance: water-stable nodes (Zr/Al/Ti/Fe) high; labile (Mg/Ca/Zn) lower.
    els = set(record.elements)
    if els & _WATER_STABLE_METALS:
        water_tolerance = 0.8
    elif els & _WATER_LABILE_METALS:
        water_tolerance = 0.5
    else:
        water_tolerance = 0.6

    # regeneration: moderate binding regenerates easily; very strong binding (amine+OMS)
    # raises regeneration energy -> lower score.
    if oms and amine:
        regeneration = 0.5
    elif has_polar_site:
        regeneration = 0.7
    else:
        regeneration = 0.8

    synth = record.synthesizability_score or synthesizability_score(record)
    novelty = record.novelty_score

    components = {
        "co2_affinity": co2_affinity,
        "selectivity": selectivity,
        "water_tolerance": water_tolerance,
        "regeneration": regeneration,
        "pore_geometry": pore_geometry,
        "stability": stability,
        "synthesizability": synth,
        "novelty": novelty,
    }

    # CO2 candidates require pore descriptors or receive low confidence.
    if not feats["has_pore_descriptors"]:
        confidence = 0.2
    else:
        confidence = min(stab_conf, 1.0)

    res = assemble_score(components, campaign, confidence=confidence)
    res.explanation = explain_score(record, res, feats)
    return res


def explain_score(record: CandidateRecord, score: ScoreResult, feats: dict | None = None) -> list[str]:
    feats = feats or compute_features(record)
    lines = [
        f"CO2 capture candidate {record.reduced_formula} (total={score.total:.3f}, conf={score.confidence:.2f})",
        f"  PLD: {feats['pld_a']} A; LCD: {feats['lcd_a']} A; void fraction: {feats['void_fraction']:.2f}",
        f"  surface area: {feats['surface_area_m2_g']} m2/g; density: {feats['density']} g/cm3",
        f"  open metal site: {feats['open_metal_site']}; amine functional: {feats['amine_functional']}; "
        f"node metals: {feats['node_metals']}",
        f"  e_above_hull: {record.ml_e_above_hull} eV/atom; uncertainty: {record.uncertainty_score:.3f}",
    ]
    if not feats["has_pore_descriptors"]:
        lines.append(
            "  NOTE: pore descriptors required (PLD/LCD/void/surface area) -> low confidence; "
            "next step: screen against CoRE MOF/QMOF or run Zeo++."
        )
    if score.rejection_reason:
        lines.append(f"  REJECTED: {score.rejection_reason}")
    return lines
