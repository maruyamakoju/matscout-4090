"""Thermodynamic stability scoring (technical spec §6.1)."""

from __future__ import annotations

from ..data.schemas import CandidateRecord


def stability_score(e_hull: float | None) -> float:
    """Stepwise stability score from energy above hull (eV/atom)."""
    if e_hull is None:
        return 0.0
    if e_hull <= 0.000:
        return 1.0
    if e_hull <= 0.025:
        return 0.9
    if e_hull <= 0.050:
        return 0.75
    if e_hull <= 0.080:
        return 0.45
    if e_hull <= 0.100:
        return 0.25
    return 0.0


def cohort_relative_stability(record: CandidateRecord, cohort_min_energy: dict[str, float]) -> float:
    """Fallback when no e_above_hull is available (no MP hull).

    Ranks a candidate's ML energy/atom relative to the lowest-energy candidate sharing
    its anonymous formula. This is a *relative* signal, not a true hull distance; the
    caller should mark confidence < 1.
    """
    if record.ml_energy_per_atom is None:
        return 0.0
    key = record.anonymous_formula
    best = cohort_min_energy.get(key)
    if best is None:
        return 0.5  # no cohort -> neutral
    delta = record.ml_energy_per_atom - best  # eV/atom above cohort best
    # map delta in [0, 0.3] -> [1, 0]
    return max(0.0, 1.0 - delta / 0.3)


def stability_for_record(record: CandidateRecord, cohort_min_energy: dict[str, float] | None = None) -> tuple[float, float]:
    """Return (stability_score, confidence). Uses real hull if present, else cohort proxy."""
    if record.ml_e_above_hull is not None:
        return stability_score(record.ml_e_above_hull), 1.0
    if cohort_min_energy is not None and record.ml_relaxed:
        return cohort_relative_stability(record, cohort_min_energy), 0.5
    return 0.0, 0.2
