"""matgl M3GNet relaxation backend (optional tertiary potential)."""

from __future__ import annotations

from .base import AseRelaxer


class M3GNetRelaxer(AseRelaxer):
    name = "m3gnet"

    def available(self) -> bool:
        try:
            import matgl  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _make_calculator(self):
        import matgl
        from matgl.ext.ase import PESCalculator

        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return PESCalculator(pot)
