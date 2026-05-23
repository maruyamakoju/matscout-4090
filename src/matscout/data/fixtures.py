"""Bundled seed structures used when no Materials Project API key is available.

These are real, well-known inorganic prototypes generated from their space group +
Wyckoff positions via pymatgen. The set is deliberately diverse across structure
families (rocksalt, perovskite, spinel, rutile, anatase, wurtzite, fluorite,
delafossite, chalcopyrite, layered oxide/dichalcogenide, half-/full-Heusler,
anti-perovskite, pyrite, cuprite, corundum, ReO3) so the substitution / prototype
engines have meaningful chemistry to expand across without an MP key.

Each spec produces a *skeleton*; ML relaxation fixes the geometry afterwards. Any
spec whose Wyckoff coords yield overlapping atoms is dropped automatically.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

# system: how to build the conventional lattice. params are passed to the constructor.
# coords are one representative site per distinct Wyckoff position; from_spacegroup
# expands them by symmetry.
_SPECS: list[dict] = [
    # --- cubic ---
    dict(label="NaCl", family="rocksalt", system="cubic", a=5.64, sg=225,
         species=["Na", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="MgO", family="rocksalt", system="cubic", a=4.21, sg=225,
         species=["Mg", "O"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="ZnS", family="zincblende", system="cubic", a=5.41, sg=216,
         species=["Zn", "S"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="CaF2", family="fluorite", system="cubic", a=5.46, sg=225,
         species=["Ca", "F"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="Li2O", family="antifluorite", system="cubic", a=4.61, sg=225,
         species=["O", "Li"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]),
    dict(label="CsCl", family="cscl", system="cubic", a=4.12, sg=221,
         species=["Cs", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="SrTiO3", family="perovskite", system="cubic", a=3.905, sg=221,
         species=["Sr", "Ti", "O"], coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]),
    dict(label="MgAl2O4", family="spinel", system="cubic", a=8.08, sg=227,
         species=["Al", "Mg", "O"],
         coords=[[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0.3866, 0.3866, 0.3866]]),
    dict(label="Si", family="diamond", system="cubic", a=5.43, sg=227,
         species=["Si"], coords=[[0.0, 0.0, 0.0]]),
    dict(label="Cu2O", family="cuprite", system="cubic", a=4.27, sg=224,
         species=["Cu", "O"], coords=[[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]]),
    dict(label="FeS2", family="pyrite", system="cubic", a=5.42, sg=205,
         species=["Fe", "S"], coords=[[0.0, 0.0, 0.0], [0.384, 0.384, 0.384]]),
    dict(label="Li3OCl", family="anti-perovskite", system="cubic", a=3.91, sg=221,
         species=["Cl", "O", "Li"], coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]),
    dict(label="TiNiSn", family="half-heusler", system="cubic", a=5.93, sg=216,
         species=["Ti", "Ni", "Sn"], coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]),
    dict(label="Cu2MnAl", family="full-heusler", system="cubic", a=5.95, sg=225,
         species=["Cu", "Mn", "Al"], coords=[[0.25, 0.25, 0.25], [0, 0, 0], [0.5, 0.5, 0.5]]),
    dict(label="ReO3", family="reo3", system="cubic", a=3.74, sg=221,
         species=["Re", "O"], coords=[[0, 0, 0], [0.5, 0.0, 0.0]]),
    # --- tetragonal ---
    dict(label="TiO2", family="rutile", system="tetra", a=4.59, c=2.96, sg=136,
         species=["Ti", "O"], coords=[[0, 0, 0], [0.305, 0.305, 0.0]]),
    dict(label="TiO2-anatase", family="anatase", system="tetra", a=3.78, c=9.51, sg=141,
         species=["Ti", "O"], coords=[[0, 0, 0], [0.0, 0.0, 0.208]]),
    dict(label="CuFeS2", family="chalcopyrite", system="tetra", a=5.29, c=10.43, sg=122,
         species=["Cu", "Fe", "S"], coords=[[0, 0, 0], [0.0, 0.0, 0.5], [0.25, 0.25, 0.125]]),
    # --- hexagonal / rhombohedral (hex setting) ---
    dict(label="ZnO", family="wurtzite", system="hex", a=3.25, c=5.20, sg=186,
         species=["Zn", "O"], coords=[[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.375]]),
    dict(label="NiAs", family="niasl", system="hex", a=3.60, c=5.00, sg=194,
         species=["Ni", "As"], coords=[[0, 0, 0], [1 / 3, 2 / 3, 0.25]]),
    dict(label="CuAlO2", family="delafossite", system="hex", a=2.86, c=16.94, sg=166,
         species=["Cu", "Al", "O"], coords=[[0, 0, 0], [0, 0, 0.5], [0, 0, 0.11]]),
    dict(label="LiCoO2", family="layered-oxide", system="hex", a=2.82, c=14.05, sg=166,
         species=["Li", "Co", "O"], coords=[[0, 0, 0], [0, 0, 0.5], [0, 0, 0.26]]),
    dict(label="MoS2", family="layered-dichalcogenide", system="hex", a=3.16, c=12.30, sg=194,
         species=["Mo", "S"], coords=[[1 / 3, 2 / 3, 0.25], [1 / 3, 2 / 3, 0.621]]),
    dict(label="Al2O3", family="corundum", system="hex", a=4.76, c=12.99, sg=167,
         species=["Al", "O"], coords=[[0, 0, 0.352], [0.306, 0.0, 0.25]]),
]


def _build(spec: dict) -> Structure:
    system = spec["system"]
    if system == "cubic":
        lat = Lattice.cubic(spec["a"])
    elif system == "tetra":
        lat = Lattice.tetragonal(spec["a"], spec["c"])
    elif system == "hex":
        lat = Lattice.hexagonal(spec["a"], spec["c"])
    elif system == "ortho":
        lat = Lattice.orthorhombic(spec["a"], spec["b"], spec["c"])
    else:  # pragma: no cover
        raise ValueError(f"unknown system {system}")
    return Structure.from_spacegroup(spec["sg"], lat, spec["species"], spec["coords"])


def _min_distance_ok(structure: Structure, cutoff: float = 0.9) -> bool:
    if len(structure) < 2:
        return True
    import numpy as np

    dm = structure.distance_matrix
    iu = np.triu_indices(len(structure), k=1)
    return bool(dm[iu].min() >= cutoff)


def seed_structures() -> list[tuple[str, str, Structure]]:
    """Return list of (label, prototype_family, Structure), dropping degenerate builds."""
    out: list[tuple[str, str, Structure]] = []
    for spec in _SPECS:
        try:
            s = _build(spec)
        except Exception:  # pragma: no cover - prototype build robustness
            continue
        if _min_distance_ok(s):
            out.append((spec["label"], spec["family"], s))
    return out
