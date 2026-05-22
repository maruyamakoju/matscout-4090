"""Chemistry-level validity checks: excluded elements, charge balance plausibility."""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Composition

from ..data.elements import EXCLUDE_ELEMENTS


@dataclass
class ChemCheckResult:
    ok: bool
    reason: str | None = None
    charge_balanced: bool | None = None


def has_excluded_element(elements: list[str], extra_exclude: set[str] | None = None) -> str | None:
    excl = EXCLUDE_ELEMENTS | (extra_exclude or set())
    for e in elements:
        if e in excl:
            return e
    return None


def charge_neutral_possible(composition: Composition) -> bool:
    """True if pymatgen can assign integer oxidation states summing to zero."""
    try:
        states = composition.oxi_state_guesses(max_sites=-20)
        return len(states) > 0
    except Exception:
        return False


def validate_chemistry(
    composition: Composition,
    extra_exclude: set[str] | None = None,
    max_elements: int = 6,
    require_charge_balance: bool = False,
) -> ChemCheckResult:
    elements = [str(e) for e in composition.elements]
    bad = has_excluded_element(elements, extra_exclude)
    if bad:
        return ChemCheckResult(False, f"excluded_element:{bad}")
    if len(elements) > max_elements:
        return ChemCheckResult(False, f"too_many_elements({len(elements)})")
    neutral = charge_neutral_possible(composition)
    if require_charge_balance and not neutral:
        return ChemCheckResult(False, "charge_imbalance", charge_balanced=False)
    return ChemCheckResult(True, None, charge_balanced=neutral)
