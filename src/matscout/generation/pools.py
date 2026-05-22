"""Shared helpers for generation: allowed element pools + substitution maps."""

from __future__ import annotations

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.elements import (
    EXCLUDE_ELEMENTS,
    ISOVALENT_BY_STATE,
    SAME_GROUP_FAMILIES,
)


def excluded_elements(campaign: CampaignConfig, global_cfg: GlobalConfig) -> set[str]:
    return EXCLUDE_ELEMENTS | set(campaign.elements.exclude) | set(global_cfg.extra_exclude_elements)


def allowed_elements(campaign: CampaignConfig, global_cfg: GlobalConfig) -> set[str]:
    """Campaign include set minus exclusions. Empty include -> all non-excluded common elements."""
    excl = excluded_elements(campaign, global_cfg)
    include = set(campaign.elements.all_include())
    if include:
        return include - excl
    # default broad pool of common, well-behaved elements
    common = {
        "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Si", "Ge", "Sn",
        "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni",
        "Cu", "Ag", "Zn", "Sc", "Y", "La", "B", "C", "N", "P", "O", "S", "Se", "F",
        "Cl", "Br", "I", "Bi", "Sb",
    }
    return common - excl


def is_anion(symbol: str) -> bool:
    """Heuristic: high electronegativity p-block elements act as anions."""
    return symbol in {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P"}


def build_substitution_map(
    campaign: CampaignConfig, global_cfg: GlobalConfig
) -> dict[str, list[str]]:
    """For each element, list isovalent / same-group replacements allowed in this campaign."""
    allowed = allowed_elements(campaign, global_cfg)
    excl = excluded_elements(campaign, global_cfg)
    sub_map: dict[str, set[str]] = {}

    def add(src: str, targets: list[str]) -> None:
        sub_map.setdefault(src, set())
        for t in targets:
            if t != src and t not in excl and t in allowed:
                sub_map[src].add(t)

    for family in SAME_GROUP_FAMILIES:
        for el in family:
            add(el, family)
    for targets in ISOVALENT_BY_STATE.values():
        for el in targets:
            add(el, targets)
    return {k: sorted(v) for k, v in sub_map.items() if v}


def site_role_pools(campaign: CampaignConfig, global_cfg: GlobalConfig) -> dict[str, list[str]]:
    """Element pools split by structural role, for prototype filling."""
    allowed = allowed_elements(campaign, global_cfg)
    anions = sorted(e for e in allowed if is_anion(e))
    cations = sorted(e for e in allowed if not is_anion(e))
    return {"anion": anions, "cation": cations, "all": sorted(allowed)}


def passes_element_filter(
    elements: list[str], campaign: CampaignConfig, global_cfg: GlobalConfig
) -> bool:
    """True if composition uses no excluded elements and includes >=1 required element."""
    excl = excluded_elements(campaign, global_cfg)
    if any(e in excl for e in elements):
        return False
    include_any = set(campaign.elements.include_any) | set(campaign.elements.mobile_ions)
    if include_any and not (set(elements) & include_any):
        return False
    return True
