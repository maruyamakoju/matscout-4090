"""Curated element property tables used by scoring / penalties.

Values are approximate, sourced from common references (USGS crustal abundance,
order-of-magnitude commodity prices). They are intended for *ranking heuristics*,
not quantitative accuracy.
"""

from __future__ import annotations

# Hard-exclude elements (radioactive / highly toxic / not wanted in any campaign).
EXCLUDE_ELEMENTS: set[str] = {
    "Hg", "Cd", "Tl", "Be", "U", "Th", "Pu", "Ra", "Po", "At", "Rn",
    "Ac", "Pa", "Np", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Tc",
}

# Toxic elements -> toxicity penalty (0 = none, 1 = severe). Excluded ones implied severe.
TOXICITY: dict[str, float] = {
    "Hg": 1.0, "Cd": 1.0, "Tl": 1.0, "Be": 1.0, "Pb": 0.8, "As": 0.7,
    "Sb": 0.4, "Se": 0.3, "Te": 0.3, "Ba": 0.3, "Cr": 0.3, "Ni": 0.25,
    "Co": 0.2, "V": 0.2, "F": 0.15, "Os": 0.6,
}

# Crustal abundance (mass fraction, ppm). Higher = more abundant / cheaper.
CRUSTAL_ABUNDANCE_PPM: dict[str, float] = {
    "O": 461000, "Si": 282000, "Al": 82300, "Fe": 56300, "Ca": 41500,
    "Na": 23600, "Mg": 23300, "K": 20900, "Ti": 5650, "H": 1400,
    "P": 1050, "Mn": 950, "F": 585, "Ba": 425, "Sr": 370, "S": 350,
    "C": 200, "Zr": 165, "V": 120, "Cl": 145, "Cr": 102, "Ni": 84,
    "Zn": 70, "Cu": 60, "Ce": 66, "Nd": 41, "La": 39, "Y": 33, "Co": 25,
    "Sc": 22, "Li": 20, "Nb": 20, "Ga": 19, "B": 10, "Pb": 14, "Th": 9.6,
    "Sm": 7.0, "Gd": 6.2, "Ge": 1.5, "As": 1.8, "Mo": 1.2, "W": 1.25,
    "Sn": 2.3, "Br": 2.4, "Hf": 3.0, "Cs": 3.0, "Be": 2.8, "U": 2.7,
    "Ta": 2.0, "Sb": 0.2, "Cd": 0.15, "Ag": 0.075, "Se": 0.05, "Hg": 0.085,
    "I": 0.45, "Bi": 0.009, "In": 0.25, "Te": 0.001, "Tl": 0.85,
    "Ru": 0.001, "Rh": 0.001, "Pd": 0.015, "Os": 0.0015, "Ir": 0.001,
    "Pt": 0.005, "Au": 0.004, "Re": 0.0007, "Rb": 90, "Ne": 0.005,
    "N": 19, "Pr": 9.2, "Dy": 5.2, "Er": 3.5, "Yb": 3.2,
    "Eu": 2.0, "Ho": 1.3, "Tb": 1.2, "Lu": 0.8, "Tm": 0.52,
}

# Rough relative cost class (1 = cheap commodity, 5 = precious / very scarce).
COST_CLASS: dict[str, int] = {
    # precious metals / very scarce
    "Au": 5, "Pt": 5, "Ir": 5, "Os": 5, "Rh": 5, "Ru": 5, "Pd": 5, "Re": 5,
    "Ag": 4, "Ga": 4, "In": 4, "Ge": 4, "Te": 4, "Sc": 4, "Hf": 4, "Ta": 4,
    "Co": 3, "Li": 3, "Ni": 3, "W": 3, "Mo": 3, "Nb": 3, "V": 3, "Bi": 3,
    "Sn": 3, "Se": 3, "Sb": 3, "Cs": 3, "Rb": 3, "Y": 3,
    "Cu": 2, "Zn": 2, "Mn": 2, "Ti": 2, "Zr": 2, "B": 2, "Sr": 2, "Ba": 2,
}

# Same-group substitution families (for substitution generation).
SAME_GROUP_FAMILIES: list[list[str]] = [
    ["Li", "Na", "K", "Rb", "Cs"],
    ["Mg", "Ca", "Sr", "Ba"],
    ["Sc", "Y", "La"],
    ["Al", "Ga", "In"],
    ["Si", "Ge", "Sn"],
    ["P", "As", "Sb"],
    ["O", "S", "Se", "Te"],
    ["F", "Cl", "Br", "I"],
    ["Ti", "Zr", "Hf"],
    ["V", "Nb", "Ta"],
    ["Cr", "Mo", "W"],
    ["Mn", "Fe", "Co", "Ni"],
    ["Cu", "Ag"],
    ["Zn", "Cd"],
    ["Bi", "Sb"],
]

# Isovalent substitution sets keyed by oxidation state (for charge-preserving swaps).
ISOVALENT_BY_STATE: dict[str, list[str]] = {
    "anion_-2": ["O", "S", "Se"],
    "anion_-1": ["F", "Cl", "Br", "I"],
    "cation_+5": ["P", "V", "As", "Nb", "Ta"],
    "cation_+4": ["Ti", "Zr", "Hf", "Sn", "Ge"],
    "cation_+3": ["Al", "Ga", "In", "Sc", "Y", "Bi", "Sb"],
    "cation_+2": ["Mg", "Ca", "Sr", "Zn", "Mn", "Fe", "Co", "Ni"],
    "cation_+1": ["Li", "Na", "K", "Cu", "Ag"],
}

REDOX_ACTIVE_TM: set[str] = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W", "Nb"}

MOBILE_IONS: set[str] = {"Li", "Na", "Mg", "Zn", "K", "Ca"}

# ns2 lone-pair cations -> defect tolerance bonus for solar absorbers.
LONE_PAIR_CATIONS: set[str] = {"Pb", "Sn", "Bi", "Sb", "Tl", "Ge", "In"}

HALOGENS: set[str] = {"F", "Cl", "Br", "I"}
CHALCOGENS: set[str] = {"O", "S", "Se", "Te"}


def abundance_score(elements: list[str]) -> float:
    """0..1 abundance score: limited by the *scarcest* element present (log-scaled)."""
    import math

    if not elements:
        return 0.0
    scores = []
    for el in set(elements):
        ppm = CRUSTAL_ABUNDANCE_PPM.get(el, 0.5)  # unknown -> assume scarce-ish
        # map log10(ppm) in [-3, 5] -> [0, 1]
        s = (math.log10(max(ppm, 1e-3)) + 3.0) / 8.0
        scores.append(min(max(s, 0.0), 1.0))
    return min(scores)  # bottleneck element dominates


def toxicity_penalty(elements: list[str]) -> float:
    """0..1 penalty; driven by the most toxic element present."""
    if not elements:
        return 0.0
    return max((TOXICITY.get(el, 0.0) for el in set(elements)), default=0.0)


def cost_penalty(elements: list[str]) -> float:
    """0..1 penalty derived from the most expensive element's cost class."""
    if not elements:
        return 0.0
    worst = max((COST_CLASS.get(el, 1) for el in set(elements)), default=1)
    return (worst - 1) / 4.0  # class 1 -> 0.0, class 5 -> 1.0


def scarcity_penalty(elements: list[str]) -> float:
    """1 - abundance_score, i.e. penalty for scarce elements."""
    return 1.0 - abundance_score(elements)
