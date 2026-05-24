"""Formation-energy math + stability fallback (CI-safe; synthetic references)."""

from __future__ import annotations

from pymatgen.core import Composition

from matscout.data.schemas import CandidateRecord
from matscout.properties.formation import formation_energy_per_atom
from matscout.properties.stability import formation_stability_score, stability_for_record

# synthetic per-atom reference energies (eV/atom)
REFS = {"Mg": -1.5, "O": -4.5, "Na": -1.3, "Cl": -1.8}


def test_formation_energy_math():
    # MgO: E/atom - 0.5*(-1.5) - 0.5*(-4.5) = E/atom + 3.0
    ef = formation_energy_per_atom(Composition("MgO"), -6.0, REFS)
    assert abs(ef - (-3.0)) < 1e-6


def test_formation_energy_missing_reference_returns_none():
    # Ti has no reference -> None (no fabrication)
    assert formation_energy_per_atom(Composition("TiO2"), -8.0, REFS) is None


def test_formation_energy_none_inputs():
    assert formation_energy_per_atom(Composition("MgO"), None, REFS) is None
    assert formation_energy_per_atom(Composition("MgO"), -6.0, {}) is None


def test_formation_stability_score_monotonic():
    assert formation_stability_score(-1.5) == 0.9
    assert formation_stability_score(0.1) == 0.1  # positive -> unstable
    assert formation_stability_score(-0.75) > formation_stability_score(-0.25)
    assert formation_stability_score(None) == 0.0


def test_stability_prefers_formation_over_cohort(nacl_structure):
    rec = CandidateRecord.from_structure(nacl_structure, candidate_id="s::1", source="manual_seed")
    rec.ml_relaxed = True
    rec.ml_formation_energy_per_atom = -1.5  # strongly stable
    score, conf = stability_for_record(rec, cohort_min_energy={})
    assert score == 0.9
    assert conf == 0.55  # formation-energy confidence tier


def test_stability_real_hull_takes_priority(nacl_structure):
    rec = CandidateRecord.from_structure(nacl_structure, candidate_id="s::2", source="manual_seed")
    rec.ml_e_above_hull = 0.0
    rec.ml_formation_energy_per_atom = -1.5
    score, conf = stability_for_record(rec)
    assert score == 1.0 and conf == 1.0  # hull beats formation energy
