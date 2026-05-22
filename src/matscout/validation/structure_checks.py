"""Geometric structure sanity checks (run before and after relaxation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure


@dataclass
class CheckResult:
    ok: bool
    reason: str | None = None


def min_interatomic_distance(structure: Structure) -> float:
    """Smallest pairwise distance (Angstrom), accounting for periodic images."""
    if len(structure) < 2:
        return float("inf")
    dmat = structure.distance_matrix
    n = len(structure)
    iu = np.triu_indices(n, k=1)
    if iu[0].size == 0:
        return float("inf")
    return float(dmat[iu].min())


def volume_per_atom(structure: Structure) -> float:
    return structure.volume / max(len(structure), 1)


def validate_structure(
    structure: Structure,
    min_dist: float = 0.75,
    min_vol_per_atom: float = 3.0,
    max_vol_per_atom: float = 200.0,
) -> CheckResult:
    """Reject overlapping atoms, collapsed/exploded cells, and degenerate structures."""
    if len(structure) == 0:
        return CheckResult(False, "empty_structure")
    # partial / disordered occupancy
    if not structure.is_ordered:
        return CheckResult(False, "disordered_occupancy")
    d = min_interatomic_distance(structure)
    if d < min_dist:
        return CheckResult(False, f"atom_overlap(min_dist={d:.2f}A)")
    vpa = volume_per_atom(structure)
    if vpa < min_vol_per_atom:
        return CheckResult(False, f"cell_collapse(vol/atom={vpa:.1f})")
    if vpa > max_vol_per_atom:
        return CheckResult(False, f"cell_too_large(vol/atom={vpa:.1f})")
    # degenerate lattice angles
    angles = structure.lattice.angles
    if any(a < 20 or a > 160 for a in angles):
        return CheckResult(False, "degenerate_lattice_angles")
    return CheckResult(True, None)


def volume_change_pct(before: Structure, after: Structure) -> float:
    v0 = before.volume / max(len(before), 1)
    v1 = after.volume / max(len(after), 1)
    return 100.0 * (v1 - v0) / v0
