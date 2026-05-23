"""Unified ASE-based ML relaxation loop, shared by CHGNet / MACE / matgl.

Each backend only supplies an ASE calculator; the FIRE relaxation, force/energy
extraction and cell handling live here so all potentials behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure


@dataclass
class RelaxResult:
    model: str
    relaxed_structure: Structure | None
    energy_per_atom: float | None
    max_force_ev_a: float | None
    converged: bool
    volume_change_pct: float | None
    n_steps: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.relaxed_structure is not None and self.error is None


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class AseRelaxer:
    """Wraps an ASE-calculator factory in a FIRE relaxation with optional cell relax."""

    name: str = "ase"

    def __init__(
        self,
        device: str = "auto",
        fmax: float = 0.05,
        max_steps: int = 300,
        relax_cell: bool = True,
    ):
        self.device = resolve_device(device)
        self.fmax = fmax
        self.max_steps = max_steps
        self.relax_cell = relax_cell
        self._calc = None

    # --- backends override these ---
    def _make_calculator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    # --- shared machinery ---
    def _calc_instance(self):
        if self._calc is None:
            self._calc = self._make_calculator()
        return self._calc

    def relax(self, structure: Structure) -> RelaxResult:
        if not self.available():
            return RelaxResult(self.name, None, None, None, False, None,
                               error="backend_unavailable")
        try:
            import contextlib
            import io

            from ase.optimize import FIRE
            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = self._calc_instance()
            target: object = atoms  # FIRE accepts Atoms or a cell filter
            if self.relax_cell:
                try:
                    from ase.filters import FrechetCellFilter

                    target = FrechetCellFilter(atoms)
                except Exception:
                    from ase.constraints import ExpCellFilter  # type: ignore[attr-defined]

                    target = ExpCellFilter(atoms)

            v0 = structure.volume
            with contextlib.redirect_stdout(io.StringIO()):
                opt = FIRE(target, logfile=None)
                opt.run(fmax=self.fmax, steps=self.max_steps)

            forces = atoms.get_forces()
            fnorms = np.linalg.norm(forces, axis=1)
            max_force = float(fnorms.max()) if len(fnorms) else 0.0
            energy = float(atoms.get_potential_energy())
            relaxed: Structure = AseAtomsAdaptor.get_structure(atoms)
            converged = max_force <= self.fmax * 1.5
            vchg = 100.0 * (relaxed.volume - v0) / v0
            return RelaxResult(
                model=self.name,
                relaxed_structure=relaxed,
                energy_per_atom=energy / len(atoms),
                max_force_ev_a=max_force,
                converged=converged,
                volume_change_pct=vchg,
                n_steps=getattr(opt, "nsteps", None),
            )
        except Exception as exc:
            return RelaxResult(self.name, None, None, None, False, None, error=str(exc)[:200])
