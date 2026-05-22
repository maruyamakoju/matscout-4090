"""Schema + config tests."""

from __future__ import annotations

from matscout.config.loader import available_campaigns, load_campaign_config
from matscout.data.schemas import CandidateRecord, composition_hash, structure_hash


def test_candidate_record_roundtrip(nacl_structure):
    rec = CandidateRecord.from_structure(
        nacl_structure, candidate_id="t::1", source="manual_seed"
    )
    assert rec.reduced_formula == "NaCl"
    assert set(rec.elements) == {"Na", "Cl"}
    assert rec.nsites == len(nacl_structure)
    # CIF round-trips back to a structure
    back = rec.get_structure()
    assert back.composition.reduced_formula == "NaCl"


def test_parquet_roundtrip_preserves_none(nacl_structure, tmp_path):
    # None string/float fields must survive a parquet save/load (pandas turns them to NaN)
    from matscout.data.store import load_parquet, save_parquet

    rec = CandidateRecord.from_structure(nacl_structure, candidate_id="t::1", source="manual_seed")
    assert rec.rejection_reason is None and rec.ml_energy_per_atom is None
    p = save_parquet([rec], tmp_path / "rt.parquet")
    back = load_parquet(p)
    assert len(back) == 1
    assert back[0].rejection_reason is None
    assert back[0].ml_energy_per_atom is None
    assert back[0].elements == rec.elements  # list column survives


def test_structure_hash_stable(nacl_structure):
    h1 = structure_hash(nacl_structure)
    h2 = structure_hash(nacl_structure.copy())
    assert h1 == h2


def test_composition_hash_distinct():
    assert composition_hash("NaCl") != composition_hash("KCl")


def test_all_campaign_configs_load():
    names = available_campaigns()
    assert set(names) >= {"battery", "semiconductor", "catalyst", "solar", "co2_capture"}
    for name in names:
        c = load_campaign_config(name)
        assert c.name == name
        # objective weights present and normalizable
        norm = c.normalized_objectives()
        if norm:
            assert abs(sum(norm.values()) - 1.0) < 1e-6


def test_battery_config_elements(battery_cfg):
    inc = battery_cfg.elements.all_include()
    assert "Li" in inc and "Na" in inc
    assert "Hg" in battery_cfg.elements.exclude
