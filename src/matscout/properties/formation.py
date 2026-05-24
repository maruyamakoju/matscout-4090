"""Formation energy per atom from ML elemental reference energies.

CHGNet/MACE are trained on MP-compatible energies, so a compound's ML energy and the ML
energies of the elemental ground states live on the same scale. Therefore

    E_form/atom = E_compound/atom - Σ_i x_i · E_elem_i/atom

is a real (ML-level) formation energy — a far better stability signal than a cohort-relative
proxy when no Materials Project hull is available. Reference energies are computed once by
relaxing curated elemental ground-state structures and cached to disk. Elements without a
curated reference yield `None` (no fabrication).
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

from pymatgen.core import Composition, Lattice, Molecule, Structure

# Curated elemental ground-state (or near-ground-state) structures. Lattice constants are
# approximate; CHGNet relaxes them. "mol" entries are isolated molecules in a box (gases).
# system: cubic_bcc | cubic_fcc | hcp | diamond | mol
_ELEMENTAL: dict[str, dict] = {
    # bcc metals
    "Li": {"sys": "bcc", "a": 3.51}, "Na": {"sys": "bcc", "a": 4.29},
    "K": {"sys": "bcc", "a": 5.23}, "Rb": {"sys": "bcc", "a": 5.59},
    "Cs": {"sys": "bcc", "a": 6.05}, "Ba": {"sys": "bcc", "a": 5.02},
    "V": {"sys": "bcc", "a": 3.03}, "Nb": {"sys": "bcc", "a": 3.30},
    "Ta": {"sys": "bcc", "a": 3.31}, "Cr": {"sys": "bcc", "a": 2.88},
    "Mo": {"sys": "bcc", "a": 3.15}, "W": {"sys": "bcc", "a": 3.16},
    "Fe": {"sys": "bcc", "a": 2.87},
    # fcc metals
    "Al": {"sys": "fcc", "a": 4.05}, "Cu": {"sys": "fcc", "a": 3.61},
    "Ni": {"sys": "fcc", "a": 3.52}, "Ca": {"sys": "fcc", "a": 5.58},
    "Sr": {"sys": "fcc", "a": 6.08}, "Ag": {"sys": "fcc", "a": 4.09},
    "Pb": {"sys": "fcc", "a": 4.95}, "Pd": {"sys": "fcc", "a": 3.89},
    "Pt": {"sys": "fcc", "a": 3.92}, "Rh": {"sys": "fcc", "a": 3.80},
    "Ir": {"sys": "fcc", "a": 3.84}, "Au": {"sys": "fcc", "a": 4.08},
    # hcp metals
    "Mg": {"sys": "hcp", "a": 3.21, "c": 5.21}, "Zn": {"sys": "hcp", "a": 2.66, "c": 4.95},
    "Ti": {"sys": "hcp", "a": 2.95, "c": 4.68}, "Zr": {"sys": "hcp", "a": 3.23, "c": 5.15},
    "Hf": {"sys": "hcp", "a": 3.20, "c": 5.05}, "Co": {"sys": "hcp", "a": 2.51, "c": 4.07},
    "Sc": {"sys": "hcp", "a": 3.31, "c": 5.27}, "Y": {"sys": "hcp", "a": 3.65, "c": 5.73},
    "Cd": {"sys": "hcp", "a": 2.98, "c": 5.62}, "Ru": {"sys": "hcp", "a": 2.71, "c": 4.28},
    "Mn": {"sys": "bcc", "a": 2.90},  # alpha-Mn is complex; bcc is an approximation
    # diamond-cubic
    "Si": {"sys": "diamond", "a": 5.43}, "Ge": {"sys": "diamond", "a": 5.66},
    "Sn": {"sys": "diamond", "a": 6.49}, "C": {"sys": "diamond", "a": 3.57},
    # diatomic / molecular (isolated in a box)
    "H": {"sys": "mol", "atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.74]]},
    "N": {"sys": "mol", "atoms": ["N", "N"], "coords": [[0, 0, 0], [0, 0, 1.10]]},
    "O": {"sys": "mol", "atoms": ["O", "O"], "coords": [[0, 0, 0], [0, 0, 1.21]]},
    "F": {"sys": "mol", "atoms": ["F", "F"], "coords": [[0, 0, 0], [0, 0, 1.42]]},
    "Cl": {"sys": "mol", "atoms": ["Cl", "Cl"], "coords": [[0, 0, 0], [0, 0, 1.99]]},
    "Br": {"sys": "mol", "atoms": ["Br", "Br"], "coords": [[0, 0, 0], [0, 0, 2.29]]},
    "I": {"sys": "mol", "atoms": ["I", "I"], "coords": [[0, 0, 0], [0, 0, 2.67]]},
    "P": {"sys": "mol", "atoms": ["P", "P", "P", "P"],
          "coords": [[0, 0, 0], [2.2, 0, 0], [1.1, 1.9, 0], [1.1, 0.63, 1.8]]},  # P4
    "S": {"sys": "mol", "atoms": ["S", "S"], "coords": [[0, 0, 0], [0, 0, 1.89]]},  # approx
    "Se": {"sys": "mol", "atoms": ["Se", "Se"], "coords": [[0, 0, 0], [0, 0, 2.17]]},  # approx
}

# Shipped with the repo so formation energies are available without recomputing.
# Energies are device-independent, so the cache is keyed by model only.
_REF_CACHE = Path(__file__).resolve().parents[3] / "models" / "elemental_refs.json"


def _build_element_structure(symbol: str) -> Structure:
    spec = _ELEMENTAL[symbol]
    sysname = spec["sys"]
    if sysname == "bcc":
        return Structure.from_spacegroup(229, Lattice.cubic(spec["a"]), [symbol], [[0, 0, 0]])
    if sysname == "fcc":
        return Structure.from_spacegroup(225, Lattice.cubic(spec["a"]), [symbol], [[0, 0, 0]])
    if sysname == "diamond":
        return Structure.from_spacegroup(227, Lattice.cubic(spec["a"]), [symbol], [[0, 0, 0]])
    if sysname == "hcp":
        return Structure.from_spacegroup(
            194, Lattice.hexagonal(spec["a"], spec["c"]), [symbol], [[1 / 3, 2 / 3, 0.25]]
        )
    if sysname == "mol":
        mol = Molecule(spec["atoms"], spec["coords"])
        return mol.get_boxed_structure(15, 15, 15)  # type: ignore[return-value]
    raise ValueError(sysname)


@lru_cache(maxsize=4)
def elemental_reference_energies(device: str = "auto", model: str = "chgnet") -> dict[str, float]:
    """Return {element: energy_per_atom} for curated elements, computing+caching as needed."""
    cache_key = model  # energies are device-independent
    cached: dict[str, dict[str, float]] = {}
    if _REF_CACHE.exists():
        try:
            cached = json.loads(_REF_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
    if cache_key in cached and len(cached[cache_key]) >= len(_ELEMENTAL):
        return cached[cache_key]

    from ..relaxation.chgnet_relaxer import CHGNetRelaxer

    relaxer = CHGNetRelaxer(device=device, fmax=0.05, max_steps=200, relax_cell=True)
    if not relaxer.available():
        return cached.get(cache_key, {})

    refs: dict[str, float] = dict(cached.get(cache_key, {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for el in _ELEMENTAL:
            if el in refs:
                continue
            try:
                # molecules: no cell relaxation (preserve the box)
                rc = _ELEMENTAL[el]["sys"] != "mol"
                relaxer.relax_cell = rc
                res = relaxer.relax(_build_element_structure(el))
                if res.ok and res.energy_per_atom is not None:
                    refs[el] = round(res.energy_per_atom, 5)
            except Exception:
                continue
    cached[cache_key] = refs
    try:
        _REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _REF_CACHE.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    except Exception:
        pass
    return refs


def formation_energy_per_atom(
    composition: Composition, energy_per_atom: float | None, refs: dict[str, float]
) -> float | None:
    """E_form/atom = E/atom - Σ x_i E_ref_i. None if energy or any reference is missing."""
    if energy_per_atom is None or not refs:
        return None
    amt = composition.get_el_amt_dict()
    total = sum(amt.values())
    if total == 0:
        return None
    e_ref = 0.0
    for el, n in amt.items():
        if el not in refs:
            return None  # incomplete references -> don't fabricate
        e_ref += (n / total) * refs[el]
    return round(energy_per_atom - e_ref, 4)
