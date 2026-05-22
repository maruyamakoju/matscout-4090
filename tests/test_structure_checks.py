"""Structure + chemistry validation tests."""

from __future__ import annotations

from pymatgen.core import Composition

from matscout.validation.chemistry_checks import validate_chemistry
from matscout.validation.structure_checks import (
    min_interatomic_distance,
    validate_structure,
)


def test_overlapping_atoms_rejected(overlapping_structure):
    res = validate_structure(overlapping_structure, min_dist=0.75)
    assert not res.ok
    assert "overlap" in res.reason


def test_valid_structure_passes(nacl_structure):
    res = validate_structure(nacl_structure, min_dist=0.75)
    assert res.ok
    assert res.reason is None


def test_min_distance_reasonable(nacl_structure):
    d = min_interatomic_distance(nacl_structure)
    assert 2.0 < d < 5.0


def test_toxic_excluded_element_rejected():
    res = validate_chemistry(Composition("HgO"))
    assert not res.ok
    assert "Hg" in res.reason


def test_normal_compound_passes_chemistry():
    res = validate_chemistry(Composition("LiFePO4"))
    assert res.ok


def test_too_many_elements_rejected():
    res = validate_chemistry(Composition("LiNaKMgCaFeO12"), max_elements=5)
    assert not res.ok
    assert "too_many_elements" in res.reason
