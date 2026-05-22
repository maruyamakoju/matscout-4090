"""Open Catalyst (OC20/OC22) integration stub.

A full catalyst workflow uses surface slabs + adsorbate configurations evaluated with
OCP / fairchem models. That is heavy and out of scope for the bulk pre-screen here.
This module documents the intended interface and, if fairchem is installed and a
slab/adsorbate dataset is present, can be extended to load OCP-style relaxations.
"""

from __future__ import annotations

from rich.console import Console

console = Console()


def fairchem_available() -> bool:
    try:
        import fairchem  # noqa: F401

        return True
    except ImportError:
        return False


def note_catalyst_scope() -> str:
    msg = (
        "Catalyst campaign runs a BULK pre-screen only. For real activity, generate "
        "slabs (low-index Miller planes), place adsorbates (H / O / OH / OOH / CO2 / CO / "
        "COOH), and relax with an OCP/fairchem model or DFT to get adsorption energies."
    )
    if not fairchem_available():
        console.print(f"[yellow]NOTE:[/] {msg} (fairchem not installed)")
    return msg
