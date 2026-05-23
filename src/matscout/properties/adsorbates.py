"""Adsorbate molecules + computational-hydrogen-electrode (CHE) reference energies.

Adsorption energy E_ads(*X) = E(slab+X) - E(slab) - E_ref(X), where E_ref is built
from gas-phase references (H2, H2O, CO, CO2) so the values map onto the standard CHE
free-energy diagrams for HER / OER / CO2RR. The binding atom is listed first in each
adsorbate Molecule (AdsorbateSiteFinder anchors on the first site).
"""

from __future__ import annotations

from pymatgen.core import Molecule

# Adsorbates (binding atom first). Geometries are approximate; ML relaxation refines them.
ADSORBATES: dict[str, Molecule] = {
    "H": Molecule(["H"], [[0.0, 0.0, 0.0]]),
    "O": Molecule(["O"], [[0.0, 0.0, 0.0]]),
    "OH": Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
    "OOH": Molecule(["O", "O", "H"], [[0.0, 0.0, 0.0], [1.06, 0.0, 0.70], [1.00, 0.0, 1.67]]),
    "CO": Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]),
    "CO2": Molecule(["C", "O", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]]),
    "COOH": Molecule(["C", "O", "O", "H"],
                     [[0.0, 0.0, 0.0], [0.0, 1.02, 0.69], [0.0, -1.06, 0.66], [0.0, -0.97, 1.62]]),
}

# Gas-phase reference molecules whose ML energies define E_ref.
GAS_MOLECULES: dict[str, Molecule] = {
    "H2": Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    "H2O": Molecule(["O", "H", "H"], [[0.0, 0.0, 0.0], [0.76, 0.0, 0.59], [-0.76, 0.0, 0.59]]),
    "CO": Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]),
    "CO2": Molecule(["C", "O", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]]),
}

# Gas-phase free-energy corrections (ZPE + integrated heat capacity - TS) at 298 K, eV.
# Applied so adsorption *energies* become approximate *free* energies for the diagrams.
ADSORBATE_DG_CORRECTION: dict[str, float] = {
    "H": 0.24, "O": 0.05, "OH": 0.30, "OOH": 0.35, "CO": 0.10, "COOH": 0.25, "CO2": 0.31,
}


def reference_energy(name: str, gas_energy: dict[str, float]) -> float | None:
    """E_ref for an adsorbate from gas-phase total energies (eV).

    gas_energy keys: 'H2', 'H2O', 'CO', 'CO2' (total energies, not per atom).
    """
    g = gas_energy
    try:
        if name == "H":
            return 0.5 * g["H2"]
        if name == "O":
            return g["H2O"] - g["H2"]
        if name == "OH":
            return g["H2O"] - 0.5 * g["H2"]
        if name == "OOH":
            return 2 * g["H2O"] - 1.5 * g["H2"]
        if name == "CO":
            return g["CO"]
        if name == "CO2":
            return g["CO2"]
        if name == "COOH":
            return g["CO2"] + 0.5 * g["H2"]
    except KeyError:
        return None
    return None
