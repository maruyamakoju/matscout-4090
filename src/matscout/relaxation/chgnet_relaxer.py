"""CHGNet relaxation backend (primary, fast)."""

from __future__ import annotations

from .base import AseRelaxer


class CHGNetRelaxer(AseRelaxer):
    name = "chgnet"

    def available(self) -> bool:
        try:
            import chgnet  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _make_calculator(self):
        from chgnet.model.dynamics import CHGNetCalculator

        use_device = "cuda" if self.device == "cuda" else "cpu"
        return CHGNetCalculator(use_device=use_device)
