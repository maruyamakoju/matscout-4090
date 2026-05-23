"""Canonical data schemas: CandidateRecord and ScoreResult.

CandidateRecord is the single record type that flows through the whole pipeline
(generation -> validation -> relaxation -> scoring -> ranking -> export). It is a
pydantic model so it validates on construction and serializes cleanly to Parquet.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field
from pymatgen.core import Structure

Source = Literal[
    "materials_project",
    "mattergen",
    "substitution",
    "prototype_enum",
    "core_mof",
    "ocp_surface",
    "manual_seed",
]


class CandidateRecord(BaseModel):
    """One material candidate. See technical spec §3.1."""

    model_config = {"extra": "ignore"}

    candidate_id: str
    source: Source

    formula: str
    reduced_formula: str
    anonymous_formula: str
    elements: list[str]
    nsites: int
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None

    structure_cif: str
    structure_hash: str
    composition_hash: str

    parent_mp_id: str | None = None
    parent_source_id: str | None = None

    # ML relaxation outputs
    ml_relaxed: bool = False
    ml_model: str | None = None
    ml_energy_per_atom: float | None = None
    ml_formation_energy_per_atom: float | None = None
    ml_e_above_hull: float | None = None
    relaxation_converged: bool = False
    max_force_ev_a: float | None = None
    volume_change_pct: float | None = None

    # surrogate property predictions
    predicted_bandgap_ev: float | None = None
    bandgap_model: str | None = None
    predicted_direct_gap: bool | None = None

    # DFT verification (filled by `matscout ingest-dft` once cluster runs complete)
    dft_verified: bool = False
    dft_energy_per_atom: float | None = None
    dft_bandgap_ev: float | None = None

    # per-application scores (filled by the active campaign)
    battery_score: float | None = None
    semiconductor_score: float | None = None
    catalyst_score: float | None = None
    solar_score: float | None = None
    co2_capture_score: float | None = None

    # cross-cutting scores / penalties
    novelty_score: float = 0.0
    synthesizability_score: float = 0.0
    toxicity_penalty: float = 0.0
    scarcity_penalty: float = 0.0
    cost_penalty: float = 0.0

    uncertainty_score: float = 0.0
    final_score: float = 0.0

    tags: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None

    # ---- structure round-tripping ---------------------------------------
    def get_structure(self) -> Structure:
        """Parse the stored CIF back into a pymatgen Structure."""
        return Structure.from_str(self.structure_cif, fmt="cif")

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        *,
        candidate_id: str,
        source: Source,
        parent_mp_id: str | None = None,
        parent_source_id: str | None = None,
        tags: list[str] | None = None,
        **extra: Any,
    ) -> CandidateRecord:
        comp = structure.composition
        cif = structure.to(fmt="cif")
        try:
            sga_num = structure.get_space_group_info()
            sg_symbol, sg_number = sga_num[0], sga_num[1]
        except Exception:
            sg_symbol, sg_number = None, None
        return cls(
            candidate_id=candidate_id,
            source=source,
            formula=comp.formula,
            reduced_formula=comp.reduced_formula,
            anonymous_formula=comp.anonymized_formula,
            elements=sorted({str(el) for el in comp.elements}),
            nsites=len(structure),
            spacegroup_number=sg_number,
            spacegroup_symbol=sg_symbol,
            structure_cif=cif,
            structure_hash=structure_hash(structure),
            composition_hash=composition_hash(comp.reduced_formula),
            parent_mp_id=parent_mp_id,
            parent_source_id=parent_source_id,
            tags=tags or [],
            **extra,
        )


class ScoreResult(BaseModel):
    """Output of an application-specific scorer."""

    total: float
    components: dict[str, float] = Field(default_factory=dict)
    rejection_reason: str | None = None
    explanation: list[str] = Field(default_factory=list)
    confidence: float = 1.0  # lowered when key features are missing/proxied


def structure_hash(structure: Structure) -> str:
    """Stable hash of composition + lattice + sorted frac coords (loose, for dedup)."""
    comp = structure.composition.reduced_formula
    latt = structure.lattice
    cell = f"{latt.a:.2f}{latt.b:.2f}{latt.c:.2f}{latt.alpha:.1f}{latt.beta:.1f}{latt.gamma:.1f}"
    coords = sorted(
        f"{site.specie.symbol}{site.frac_coords[0]:.2f}{site.frac_coords[1]:.2f}{site.frac_coords[2]:.2f}"
        for site in structure
    )
    raw = comp + cell + "".join(coords)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def composition_hash(reduced_formula: str) -> str:
    return hashlib.sha1(reduced_formula.encode()).hexdigest()[:16]
