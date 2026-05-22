"""Optional MatterGen integration.

MatterGen is used purely as a *candidate generator*: we ingest CIFs it produced
(unconditional / chemical-system-constrained generation) and let MatScout's own
relaxation + scoring handle evaluation. We never treat raw generated CIFs as
"discoveries". If no MatterGen output directory is present, this is a no-op.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from .pools import passes_element_filter

console = Console()


def load_mattergen_outputs(
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    output_dir: str | Path | None = None,
) -> list[CandidateRecord]:
    """Load CIFs from data/external/mattergen/<campaign>/ if present."""
    from pymatgen.core import Structure

    base = Path(output_dir) if output_dir else Path(global_cfg.data_dir) / "external" / "mattergen" / campaign.name
    if not base.exists():
        return []
    cifs = sorted(base.glob("*.cif"))
    if not cifs:
        return []
    console.print(f"[green]MatterGen:[/] loading {len(cifs)} CIFs from {base}")
    out: list[CandidateRecord] = []
    for i, cif in enumerate(cifs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct = Structure.from_file(cif)
        except Exception:
            continue
        els = sorted({str(e) for e in struct.composition.elements})
        if not passes_element_filter(els, campaign, global_cfg):
            continue
        out.append(
            CandidateRecord.from_structure(
                struct,
                candidate_id=f"mattergen::{i:06d}",
                source="mattergen",
                parent_source_id=cif.stem,
                tags=[campaign.name, "mattergen"],
            )
        )
    return out
