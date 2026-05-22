"""Scoring + ranking tests (covers the behaviors required in the technical spec §10)."""

from __future__ import annotations

import warnings

from pymatgen.core import Lattice, Structure

from matscout.config.loader import load_campaign_config
from matscout.data.schemas import CandidateRecord
from matscout.properties import battery, catalyst, co2_capture, semiconductor
from matscout.properties.common import peak_score, target_window_score
from matscout.properties.stability import stability_score
from matscout.scoring.ranker import rank_records

warnings.simplefilter("ignore")


def _rec(structure, **kw):
    return CandidateRecord.from_structure(structure, candidate_id=kw.pop("cid", "t::x"),
                                          source="manual_seed", **kw)


def test_stability_score_monotonic():
    assert stability_score(0.0) == 1.0
    assert stability_score(0.04) == 0.75
    assert stability_score(0.09) == 0.25
    # high e_above_hull -> zero stability
    assert stability_score(0.5) == 0.0
    assert stability_score(None) == 0.0
    assert stability_score(0.0) > stability_score(0.04) > stability_score(0.09) > stability_score(0.2)


def test_target_window_peaks_inside_range():
    # peaks (==1) inside, decays outside
    assert target_window_score(1.4, 1.1, 1.6) == 1.0
    assert target_window_score(1.1, 1.1, 1.6) == 1.0
    assert target_window_score(0.5, 1.1, 1.6, width=0.5) < 1.0
    assert target_window_score(3.0, 1.1, 1.6, width=0.5) == 0.0
    assert target_window_score(None, 1.1, 1.6) == 0.0


def test_peak_score_at_target():
    assert peak_score(0.0, 0.0, 0.2) == 1.0
    assert peak_score(0.2, 0.0, 0.2) == 0.0
    assert peak_score(0.1, 0.0, 0.2) == 0.5


def test_battery_rejects_no_mobile_ion():
    # SiO2-like: no Li/Na/Mg/Zn -> rejected
    s = Structure(Lattice.cubic(5.0), ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    cfg = load_campaign_config("battery")
    res = battery.score_candidate(rec, cfg)
    assert res.rejection_reason is not None
    assert "mobile_ion" in res.rejection_reason
    assert res.total == 0.0


def test_battery_accepts_mobile_ion():
    s = Structure(Lattice.cubic(4.2), ["Li", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    rec.novelty_score = 0.5
    cfg = load_campaign_config("battery")
    res = battery.score_candidate(rec, cfg)
    assert res.rejection_reason is None
    assert res.total > 0.0


def test_semiconductor_bandgap_target_peaks_in_range():
    cfg = load_campaign_config("semiconductor")
    lo, hi = cfg.targets.get("bandgap_ev", [0.5, 4.5])
    inside = target_window_score((lo + hi) / 2, lo, hi, width=0.5)
    outside = target_window_score(hi + 2.0, lo, hi, width=0.5)
    assert inside == 1.0
    assert inside > outside


def test_semiconductor_rejects_metal():
    # pure metal -> gap ~0 -> rejected as not a semiconductor
    s = Structure(Lattice.cubic(3.6), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    res = semiconductor.score_candidate(rec, load_campaign_config("semiconductor"))
    assert res.rejection_reason is not None


def test_catalyst_rejects_without_active_metal():
    s = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    res = catalyst.score_candidate(rec, load_campaign_config("catalyst"))
    assert res.rejection_reason is not None


def test_catalyst_accepts_active_metal_oxide():
    # Fe-Ti-O type oxide with active metals scores and has high adsorption_target
    s = Structure(Lattice.cubic(4.2), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    rec.novelty_score = 0.5
    res = catalyst.score_candidate(rec, load_campaign_config("catalyst"))
    assert res.rejection_reason is None
    assert res.confidence <= 0.6  # bulk pre-screen -> reduced confidence


def test_co2_low_confidence_without_descriptors():
    s = Structure(Lattice.cubic(4.2), ["Zn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    rec.novelty_score = 0.5
    res = co2_capture.score_candidate(rec, load_campaign_config("co2_capture"))
    # no pore descriptors -> low confidence
    assert res.confidence <= 0.3


def test_co2_uses_descriptors_when_present():
    s = Structure(Lattice.cubic(4.2), ["Zn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = _rec(s)
    rec.novelty_score = 0.5
    rec.ml_e_above_hull = 0.0
    rec.tags += ["pld:4.0", "void:0.45", "oms:1"]
    res = co2_capture.score_candidate(rec, load_campaign_config("co2_capture"))
    assert res.confidence > 0.3


def test_ranking_deterministic(seeds, battery_cfg, global_cfg):
    from matscout.generation.generate import generate_candidates
    from matscout.scoring.objectives import score_records
    from matscout.validation.novelty import NoveltyIndex

    idx = NoveltyIndex.from_records(seeds)
    cands1 = generate_candidates(seeds, battery_cfg, global_cfg, n=150)
    score_records(cands1, battery_cfg, global_cfg, novelty_index=idx)
    order1 = [r.candidate_id for r in rank_records(cands1, battery_cfg)]

    cands2 = generate_candidates(seeds, battery_cfg, global_cfg, n=150)
    score_records(cands2, battery_cfg, global_cfg, novelty_index=idx)
    order2 = [r.candidate_id for r in rank_records(cands2, battery_cfg)]
    assert order1 == order2
