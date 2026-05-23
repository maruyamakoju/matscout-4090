"""Energy-above-hull via Materials Project phase diagrams.

CHGNet/MACE total energies are trained on MP-compatible (MPtrj) energies, so a CHGNet
energy can be inserted into an MP phase diagram as an approximate ComputedEntry to get
an e_above_hull estimate. When no MP API key is available we cannot build a real hull,
so we return None and the stability scorer falls back to a clearly-flagged heuristic
(never fabricating a hull distance).
"""

from __future__ import annotations

from functools import lru_cache

from pymatgen.core import Structure

from ..config.schema import GlobalConfig


@lru_cache(maxsize=64)
def _mp_entries_for_chemsys(chemsys: str, api_key: str) -> tuple:
    """Fetch (and cache) MP ComputedEntries for a chemical system like 'Li-P-S'."""
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys.split("-"))
    return tuple(entries)


def compute_e_above_hull(
    structure: Structure,
    ml_energy_per_atom: float | None,
    global_cfg: GlobalConfig,
) -> float | None:
    """Estimate e_above_hull (eV/atom) for a candidate using its ML energy + MP hull.

    Returns None if MP is unavailable or the computation fails.
    """
    if ml_energy_per_atom is None or not global_cfg.mp_api_key:
        return None
    try:
        import mp_api  # noqa: F401
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.computed_entries import ComputedEntry
    except ImportError:
        return None

    comp = structure.composition
    chemsys = "-".join(sorted({str(e) for e in comp.elements}))
    try:
        entries = list(_mp_entries_for_chemsys(chemsys, global_cfg.mp_api_key))
        cand_entry = ComputedEntry(
            composition=comp,
            energy=ml_energy_per_atom * comp.num_atoms,
            entry_id="candidate",
        )
        pd = PhaseDiagram(entries + [cand_entry])
        return float(pd.get_e_above_hull(cand_entry))  # type: ignore[arg-type]
    except Exception:
        return None


def formation_energy_proxy(ml_energy_per_atom: float | None) -> float | None:
    """Placeholder: without elemental references we cannot compute a true Ef.

    Returns the raw ML energy/atom unchanged; downstream stability uses it only as a
    *relative* cohort signal, not an absolute formation energy.
    """
    return ml_energy_per_atom
