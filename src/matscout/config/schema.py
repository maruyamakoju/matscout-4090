"""Pydantic config schemas for global + per-campaign settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RelaxationConfig(BaseModel):
    primary_model: str = "chgnet"
    secondary_models: list[str] = Field(default_factory=lambda: ["mace_mp"])
    fmax_ev_a: float = 0.05
    max_steps: int = 300
    cell_relax: bool = True
    optimizer: str = "FIRE"
    max_volume_change_pct: float = 30.0
    max_lattice_angle_change_deg: float = 15.0
    reject_if_atoms_too_close: bool = True
    min_interatomic_distance_a: float = 0.75


class GlobalConfig(BaseModel):
    device: str = "auto"  # auto|cuda|cpu
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "data/external/cache"
    mp_api_key: str | None = None
    relaxation: RelaxationConfig = Field(default_factory=RelaxationConfig)
    # global hard exclusions augmenting the built-in list
    extra_exclude_elements: list[str] = Field(default_factory=list)


class ElementSpec(BaseModel):
    include_any: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    penalize: list[str] = Field(default_factory=list)
    # battery uses structured roles; flatten via helpers below
    mobile_ions: list[str] = Field(default_factory=list)
    anions: list[str] = Field(default_factory=list)
    framework_cations: list[str] = Field(default_factory=list)

    def all_include(self) -> list[str]:
        return sorted(set(self.include_any + self.mobile_ions + self.anions + self.framework_cations))


class FilterSpec(BaseModel):
    max_nsites: int = 80
    max_e_hull_initial: float = 0.12
    max_predicted_e_hull: float = 0.10
    min_bandgap_electrolyte: float | None = None
    min_pld_a: float | None = None
    max_pld_a: float | None = None
    min_void_fraction: float | None = None


class CampaignConfig(BaseModel):
    """Per-campaign configuration. Permissive: campaign-specific blocks live in `extras`."""

    model_config = {"extra": "allow"}

    name: str
    n_generate: int = 5000
    n_relax: int = 2000
    n_screen: int = 0
    top_k: int = 200

    elements: ElementSpec = Field(default_factory=ElementSpec)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    objectives: dict[str, float] = Field(default_factory=dict)

    target_structures: list[str] = Field(default_factory=list)
    reaction_targets: list[str] = Field(default_factory=list)

    # campaign-specific structured config (targets, surfaces, adsorbates, descriptors, ...)
    targets: dict = Field(default_factory=dict)
    surfaces: dict = Field(default_factory=dict)
    adsorbates: dict = Field(default_factory=dict)
    descriptors: dict = Field(default_factory=dict)
    source_priority: list[str] = Field(default_factory=list)

    def normalized_objectives(self) -> dict[str, float]:
        """Return objective weights normalized to sum to 1 (if any are present)."""
        total = sum(self.objectives.values())
        if total <= 0:
            return dict(self.objectives)
        return {k: v / total for k, v in self.objectives.items()}
