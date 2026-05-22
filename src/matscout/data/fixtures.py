"""Bundled seed structures used when no Materials Project API key is available.

These are real, well-known inorganic prototypes generated from their space group +
Wyckoff positions via pymatgen. They are deliberately diverse (rocksalt, perovskite,
spinel, rutile, wurtzite, fluorite, delafossite, ...) so the substitution / prototype
engines have meaningful chemistry to expand across.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

# (label, prototype_family, builder) — builder returns a pymatgen Structure.
_PROTOTYPE_SPECS: list[dict] = [
    dict(label="NaCl", family="rocksalt", sg=225, a=5.64,
         species=["Na", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="MgO", family="rocksalt", sg=225, a=4.21,
         species=["Mg", "O"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="ZnS", family="zincblende", sg=216, a=5.41,
         species=["Zn", "S"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="CaF2", family="fluorite", sg=225, a=5.46,
         species=["Ca", "F"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="Li2O", family="antifluorite", sg=225, a=4.61,
         species=["O", "Li"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="CsCl", family="cscl", sg=221, a=4.12,
         species=["Cs", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="SrTiO3", family="perovskite", sg=221, a=3.905,
         species=["Sr", "Ti", "O"], coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]),
    dict(label="MgAl2O4", family="spinel", sg=227, a=8.08,
         species=["Al", "Mg", "O"],
         coords=[[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0.262, 0.262, 0.262]]),
    dict(label="Si", family="diamond", sg=227, a=5.43,
         species=["Si"], coords=[[0.0, 0.0, 0.0]]),
]

_HEX_SPECS: list[dict] = [
    dict(label="ZnO", family="wurtzite", sg=186, a=3.25, c=5.20,
         species=["Zn", "O"], coords=[[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.375]]),
    dict(label="NiAs", family="niasl", sg=194, a=3.60, c=5.00,
         species=["Ni", "As"], coords=[[0, 0, 0], [1 / 3, 2 / 3, 0.25]]),
    dict(label="CuAlO2", family="delafossite", sg=166, a=2.86, c=16.94,
         species=["Cu", "Al", "O"], coords=[[0, 0, 0], [0, 0, 0.5], [0, 0, 0.11]]),
]

_TETRA_SPECS: list[dict] = [
    dict(label="TiO2", family="rutile", sg=136, a=4.59, c=2.96,
         species=["Ti", "O"], coords=[[0, 0, 0], [0.305, 0.305, 0.0]]),
]


def _build_cubic(spec: dict) -> Structure:
    lat = Lattice.cubic(spec["a"])
    return Structure.from_spacegroup(spec["sg"], lat, spec["species"], spec["coords"])


def _build_hex(spec: dict) -> Structure:
    lat = Lattice.hexagonal(spec["a"], spec["c"])
    return Structure.from_spacegroup(spec["sg"], lat, spec["species"], spec["coords"])


def _build_tetra(spec: dict) -> Structure:
    lat = Lattice.tetragonal(spec["a"], spec["c"])
    return Structure.from_spacegroup(spec["sg"], lat, spec["species"], spec["coords"])


def seed_structures() -> list[tuple[str, str, Structure]]:
    """Return list of (label, prototype_family, Structure)."""
    out: list[tuple[str, str, Structure]] = []
    for spec in _PROTOTYPE_SPECS:
        try:
            out.append((spec["label"], spec["family"], _build_cubic(spec)))
        except Exception:  # pragma: no cover - prototype build robustness
            continue
    for spec in _HEX_SPECS:
        try:
            out.append((spec["label"], spec["family"], _build_hex(spec)))
        except Exception:  # pragma: no cover
            continue
    for spec in _TETRA_SPECS:
        try:
            out.append((spec["label"], spec["family"], _build_tetra(spec)))
        except Exception:  # pragma: no cover
            continue
    return out
