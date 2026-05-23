"""Bandgap surrogate.

If a trained surrogate model exists at models/bandgap_surrogate.pkl it is used; otherwise
a transparent composition/chemistry heuristic estimates the gap. Per the technical spec,
predicted gaps are used only as a *first-pass filter*; final candidates go to HSE06/GW/expt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pymatgen.core import Composition, Element, Structure

from ..data.elements import CHALCOGENS, HALOGENS

# Transition metals whose partially-filled d-states often close the gap (metallic-prone).
_METALLIC_PRONE = {"Fe", "Co", "Ni", "Mn", "Cr", "V", "Cu", "Mo", "W", "Ru", "Rh", "Pd", "Re"}
_SURROGATE_PATH = Path("models/bandgap_surrogate.pkl")


@lru_cache(maxsize=1)
def _load_surrogate():
    if _SURROGATE_PATH.exists():
        try:
            import joblib

            return joblib.load(_SURROGATE_PATH)
        except Exception:
            return None
    return None


def _heuristic_gap(comp: Composition) -> tuple[float, bool, float]:
    """Return (gap_ev, direct_guess, confidence) from a chemistry heuristic."""
    elements = {str(e) for e in comp.elements}
    en = {e: (Element(e).X or 1.8) for e in elements}

    anions = elements & (CHALCOGENS | HALOGENS | {"N", "P"})
    cations = elements - anions
    if not anions:
        # intermetallic / alloy -> metallic
        return 0.0, False, 0.6

    # electronegativity difference cation<->anion drives ionic gap
    max_anion_en = max((en[a] for a in anions), default=3.0)
    min_cation_en = min((en[c] for c in cations), default=1.0) if cations else 1.0
    delta_en = max_anion_en - min_cation_en

    # base gap grows with ionicity
    gap = 0.8 * delta_en

    # anion identity shifts: oxides/fluorides wide, tellurides/iodides narrow
    if "O" in anions or "F" in anions:
        gap += 0.8
    if "Te" in anions or "I" in anions:
        gap -= 0.6
    if "S" in anions or "Se" in anions or "Br" in anions:
        gap -= 0.1

    # partially-filled d transition metals tend to close the gap
    if elements & _METALLIC_PRONE:
        gap *= 0.45

    gap = max(0.0, min(gap, 6.0))
    # direct-gap guess: simple main-group / lone-pair cation chalcogenides/halides
    direct = bool(cations & {"Cu", "Ag", "Zn", "Ga", "In", "Sn", "Bi", "Sb", "Ge"}) and bool(anions & (CHALCOGENS | HALOGENS))
    return round(gap, 3), direct, 0.4


def predict_bandgap(obj: Structure | Composition) -> tuple[float, bool, str, float]:
    """Return (gap_ev, is_direct, model_name, confidence)."""
    comp = obj.composition if isinstance(obj, Structure) else obj
    elements = {str(e) for e in comp.elements}
    # Intermetallic / no electronegative anion -> metallic. Trust this over any
    # composition-only regressor, which cannot reliably predict an exact zero gap.
    if not (elements & (CHALCOGENS | HALOGENS | {"N", "P"})):
        return 0.0, False, "metallic_chemistry", 0.6

    model = _load_surrogate()
    if model is not None:
        try:
            from matminer.featurizers.composition import ElementProperty

            feat = ElementProperty.from_preset("magpie")
            X = [feat.featurize(comp)]
            gap = max(0.0, float(model.predict(X)[0]))
            # surrogate gives the magnitude; reuse the chemistry heuristic for the
            # direct/indirect guess (the regressor doesn't predict that).
            _, direct, _ = _heuristic_gap(comp)
            return gap, direct, "surrogate_magpie", 0.7
        except Exception:
            pass
    gap, direct, conf = _heuristic_gap(comp)
    return gap, direct, "heuristic_en", conf
