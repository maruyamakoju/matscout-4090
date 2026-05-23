"""Coverage for core logic: Pareto, relaxation verdicts, DFT INCAR, ranker, uncertainty."""

from __future__ import annotations

import warnings

import numpy as np
from pymatgen.core import Lattice, Structure

from matscout.config.loader import load_campaign_config
from matscout.data.schemas import CandidateRecord
from matscout.scoring.pareto import pareto_front_indices, pareto_rank
from matscout.scoring.ranker import dedup_by_formula
from matscout.scoring.uncertainty import acquisition_score, is_wildcard, uncertainty_bonus

warnings.simplefilter("ignore")


def _rec(structure, cid, **kw):
    return CandidateRecord.from_structure(structure, candidate_id=cid, source="manual_seed", **kw)


# ---------------- Pareto ----------------
def test_pareto_front_indices():
    # rows maximized; (2,2) dominates all, (1,3)&(3,1) are also non-dominated
    m = np.array([[2.0, 2.0], [1.0, 3.0], [3.0, 1.0], [1.0, 1.0]])
    front = set(pareto_front_indices(m))
    assert front == {0, 1, 2}  # row 3 (1,1) dominated
    assert 3 not in front


def test_pareto_rank_layers():
    m = np.array([[3.0, 3.0], [1.0, 1.0]])
    ranks = pareto_rank_records(m)
    assert ranks[0] == 0 and ranks[1] == 1


def pareto_rank_records(m):
    # tiny wrapper so we exercise pareto_rank with a list-like
    class _R:
        def __init__(self, a, b):
            self.final_score, self.novelty_score = a, b
            self.ml_e_above_hull = None
            self.synthesizability_score = 0.0
            self.uncertainty_score = 0.0
            self.cost_penalty = 0.0
    recs = [_R(*row) for row in m]
    return pareto_rank(recs, ["final_score", "novelty_score"])


# ---------------- relaxation verdicts ----------------
def test_classify_kill_on_huge_volume_change(nacl_structure):
    from matscout.relaxation.ensemble import _classify

    r = _rec(nacl_structure, "k::1")
    r.ml_relaxed = True
    r.relaxation_converged = True
    r.volume_change_pct = 45.0  # > 30 cap
    verdict = _classify(r, load_campaign_config("battery"))
    assert verdict == "KILL"
    assert "volume" in (r.rejection_reason or "")


def test_classify_keep_on_good_candidate(nacl_structure):
    from matscout.relaxation.ensemble import _classify

    r = _rec(nacl_structure, "k::2")
    r.ml_relaxed = True
    r.relaxation_converged = True
    r.volume_change_pct = 2.0
    r.ml_e_above_hull = 0.0
    r.uncertainty_score = 0.0
    assert _classify(r, load_campaign_config("battery")) == "KEEP"


def test_classify_kill_on_nonconverged(nacl_structure):
    from matscout.relaxation.ensemble import _classify

    r = _rec(nacl_structure, "k::3")
    r.ml_relaxed = True
    r.relaxation_converged = False
    r.volume_change_pct = 1.0
    assert _classify(r, load_campaign_config("battery")) == "KILL"


# ---------------- DFT INCAR ----------------
def test_incar_ldau_for_tm_oxide():
    from matscout.workflows.export_dft import _incar

    s = Structure(Lattice.cubic(4.3), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    incar = _incar(s, "relax", use_ldau=True)
    assert "LDAU = .TRUE." in incar
    assert "LDAUU" in incar and "ISIF = 3" in incar


def test_incar_metallic_ismear():
    from matscout.workflows.export_dft import _incar

    s = Structure(Lattice.cubic(3.6), ["Fe"], [[0, 0, 0]])
    incar = _incar(s, "static", use_ldau=False, metallic=True)
    assert "ISMEAR = 1" in incar  # metals -> Gaussian/MP smearing, not tetrahedron


def test_incar_insulator_static_tetrahedron():
    from matscout.workflows.export_dft import _incar

    s = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    incar = _incar(s, "static", use_ldau=False, metallic=False)
    assert "ISMEAR = -5" in incar


def test_priority_tiers(nacl_structure):
    from matscout.workflows.export_dft import _priority

    a = _rec(nacl_structure, "p::a")
    a.final_score, a.uncertainty_score, a.novelty_score, a.ml_e_above_hull = 0.8, 0.01, 0.7, 0.01
    assert _priority(a) == "A"
    c = _rec(nacl_structure, "p::c")
    c.final_score = 0.2
    assert _priority(c) == "C"


# ---------------- ranker / uncertainty ----------------
def test_dedup_by_formula_keeps_best(nacl_structure):
    a = _rec(nacl_structure, "d::1"); a.final_score = 0.4
    b = _rec(nacl_structure, "d::2"); b.final_score = 0.9  # same formula, higher score
    out = dedup_by_formula([a, b])
    assert len(out) == 1 and out[0].candidate_id == "d::2"


def test_uncertainty_bonus_shape():
    assert uncertainty_bonus(0.0) < uncertainty_bonus(0.05)  # too-low < sweet spot
    assert uncertainty_bonus(0.05) > uncertainty_bonus(0.5)  # sweet spot > model-failure
    assert uncertainty_bonus(0.5) <= 0.2


def test_is_wildcard(nacl_structure):
    r = _rec(nacl_structure, "w::1")
    r.novelty_score = 0.8
    r.final_score = 0.5
    r.ml_e_above_hull = 0.06  # moderate stability
    r.uncertainty_score = 0.06
    assert is_wildcard(r) is True
    boring = _rec(nacl_structure, "w::2")
    boring.novelty_score = 0.1
    boring.final_score = 0.5
    assert is_wildcard(boring) is False


def test_acquisition_score_bounded(nacl_structure):
    r = _rec(nacl_structure, "a::1")
    r.novelty_score = 0.6
    r.synthesizability_score = 0.7
    r.uncertainty_score = 0.05
    s = acquisition_score(r, app_score=0.7, diversity=0.5)
    assert 0.0 <= s <= 1.0
