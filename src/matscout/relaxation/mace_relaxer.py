"""MACE-MP / MACE-OMAT relaxation backend (secondary cross-check)."""

from __future__ import annotations

from .base import AseRelaxer


class MACERelaxer(AseRelaxer):
    """MACE foundation-model relaxer. `variant` selects the pretrained training data."""

    def __init__(self, variant: str = "mp", model_size: str = "medium", **kwargs):
        self.variant = variant  # "mp" | "omat" | "mpa"
        self.model_size = model_size
        self.name = f"mace_{variant}"
        super().__init__(**kwargs)

    def available(self) -> bool:
        try:
            import mace  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _make_calculator(self):
        from mace.calculators import mace_mp

        # mace_mp downloads the foundation model on first use (cached thereafter).
        return mace_mp(
            model=self.model_size,
            device="cuda" if self.device == "cuda" else "cpu",
            default_dtype="float64",
        )
