"""CoRE MOF / QMOF loader for the CO2-capture campaign.

Expects MOF CIFs under data/external/core_mof/*.cif and an optional descriptor table
(CSV) named descriptors.csv with columns:
    filename, pld, lcd, void_fraction, surface_area_m2_g, density, open_metal_site,
    amine_functional
Returns CandidateRecords carrying the pore descriptors as tags (pld:..., void:..., ...)
so the co2_capture scorer can read them. If the directory is absent, returns [] with a
clear warning (never fabricates pore data).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord

console = Console()


def _descriptor_tags(row: dict) -> list[str]:
    tags = []
    mapping = {"pld": "pld", "lcd": "lcd", "void_fraction": "void",
               "surface_area_m2_g": "asa", "density": "density",
               "open_metal_site": "oms", "amine_functional": "amine"}
    for col, prefix in mapping.items():
        val = row.get(col)
        if val not in (None, ""):
            tags.append(f"{prefix}:{val}")
    return tags


def load_core_mof(
    campaign: CampaignConfig,
    g: GlobalConfig,
    base_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[CandidateRecord]:
    from pymatgen.core import Structure

    base = Path(base_dir) if base_dir else Path(g.data_dir) / "external" / "core_mof"
    if not base.exists():
        console.print(
            f"[yellow]WARNING:[/] CoRE MOF directory not found ({base}). "
            "CO2-capture screening needs MOF structures + pore descriptors. "
            "Download CoRE MOF / QMOF CIFs into that folder (+ descriptors.csv)."
        )
        return []

    desc: dict[str, dict] = {}
    csv_path = base / "descriptors.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                desc[row.get("filename", "")] = row

    cifs = sorted(base.glob("*.cif"))
    limit = limit or campaign.n_screen or len(cifs)
    out: list[CandidateRecord] = []
    for i, cif in enumerate(cifs[:limit]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct = Structure.from_file(cif)
        except Exception:
            continue
        tags = [campaign.name, "core_mof"] + _descriptor_tags(desc.get(cif.name, {}))
        out.append(
            CandidateRecord.from_structure(
                struct,
                candidate_id=f"coremof::{cif.stem}",
                source="core_mof",
                parent_source_id=cif.stem,
                tags=tags,
            )
        )
    console.print(f"[green]CoRE MOF:[/] loaded {len(out)} MOFs from {base}")
    return out
