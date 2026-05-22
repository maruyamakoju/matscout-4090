"""Materials Project seed fetcher with graceful fixture fallback.

If `mp-api` is installed and an MP API key is present, query Materials Project for
stable/metastable structures matching the campaign element filters. Otherwise, fall
back to bundled fixture prototypes and print a clear warning (never fabricate data).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from .cache import DiskCache
from .elements import EXCLUDE_ELEMENTS
from .fixtures import seed_structures
from .schemas import CandidateRecord

if TYPE_CHECKING:
    from pymatgen.core import Structure

console = Console()


def _mp_available(global_cfg: GlobalConfig) -> bool:
    if not global_cfg.mp_api_key:
        return False
    try:
        import mp_api  # noqa: F401
    except ImportError:
        return False
    return True


def fetch_mp_seeds(
    campaign: CampaignConfig,
    global_cfg: GlobalConfig,
    limit: int = 1000,
) -> list[CandidateRecord]:
    """Return seed CandidateRecords for a campaign."""
    if _mp_available(global_cfg):
        try:
            return _fetch_from_mp(campaign, global_cfg, limit)
        except Exception as exc:  # pragma: no cover - network/credential failure
            console.print(
                f"[yellow]WARNING:[/] Materials Project query failed ({exc}). "
                "Falling back to bundled fixture seeds."
            )
    else:
        console.print(
            "[yellow]WARNING:[/] No Materials Project API key / mp-api not installed. "
            "Using bundled fixture seed structures. Set MP_API_KEY in .env for real data."
        )
    return _fixture_seeds(campaign, limit)


def _exclude_set(campaign: CampaignConfig, global_cfg: GlobalConfig) -> set[str]:
    return (
        EXCLUDE_ELEMENTS
        | set(campaign.elements.exclude)
        | set(global_cfg.extra_exclude_elements)
    )


def _fetch_from_mp(
    campaign: CampaignConfig, global_cfg: GlobalConfig, limit: int
) -> list[CandidateRecord]:
    from mp_api.client import MPRester

    cache = DiskCache(global_cfg.cache_dir)
    include = campaign.elements.all_include()
    exclude = sorted(_exclude_set(campaign, global_cfg))
    params = dict(include=include, exclude=exclude, limit=limit,
                  max_nsites=campaign.filters.max_nsites,
                  max_ehull=campaign.filters.max_e_hull_initial)

    cached = cache.get("mp_seeds", params)
    records: list[CandidateRecord] = []
    if cached is not None:
        for d in cached:
            records.append(CandidateRecord(**d))
        console.print(f"[green]MP cache hit:[/] {len(records)} seeds for {campaign.name}")
        return records

    fields = ["material_id", "structure", "formula_pretty", "energy_above_hull",
              "band_gap", "symmetry"]
    with MPRester(global_cfg.mp_api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=include or None,
            exclude_elements=exclude or None,
            num_sites=(1, campaign.filters.max_nsites),
            energy_above_hull=(0, campaign.filters.max_e_hull_initial),
            fields=fields,
        )
    for doc in docs[:limit]:
        struct: Structure = doc.structure
        rec = CandidateRecord.from_structure(
            struct,
            candidate_id=f"mp::{doc.material_id}",
            source="materials_project",
            parent_mp_id=str(doc.material_id),
            tags=[campaign.name, "seed"],
            ml_e_above_hull=float(doc.energy_above_hull) if doc.energy_above_hull is not None else None,
            predicted_bandgap_ev=float(doc.band_gap) if getattr(doc, "band_gap", None) is not None else None,
            bandgap_model="mp_pbe" if getattr(doc, "band_gap", None) is not None else None,
        )
        records.append(rec)
    cache.set("mp_seeds", params, [r.model_dump() for r in records])
    console.print(f"[green]Fetched {len(records)} MP seeds[/] for {campaign.name}")
    return records


def _fixture_seeds(campaign: CampaignConfig, limit: int) -> list[CandidateRecord]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seeds = seed_structures()
    records: list[CandidateRecord] = []
    for label, family, struct in seeds:
        rec = CandidateRecord.from_structure(
            struct,
            candidate_id=f"fixture::{label}",
            source="manual_seed",
            parent_source_id=label,
            tags=[campaign.name, "seed", "fixture", f"proto:{family}"],
        )
        records.append(rec)
    return records[:limit]
