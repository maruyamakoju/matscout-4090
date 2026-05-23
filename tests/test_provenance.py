"""Provenance manifest + checkpoint-state mechanics (CI-safe, no GPU/torch)."""

from __future__ import annotations

import json

from matscout.config.schema import GlobalConfig
from matscout.data.schemas import CandidateRecord
from matscout.relaxation.ensemble import _copy_relaxed_state, _load_checkpoint


def test_run_manifest_written(tmp_path):
    from matscout.workflows.provenance import write_run_manifest

    g = GlobalConfig(output_dir=str(tmp_path), device="cpu")
    path = write_run_manifest(g, ["battery", "solar"], extra={"note": "test"})
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["matscout_version"]
    assert data["campaigns"] == ["battery", "solar"]
    assert "packages" in data and "pymatgen" in data["packages"]
    assert data["note"] == "test"
    assert data["mp_api_key_present"] is False


def test_copy_relaxed_state(nacl_structure):
    src = CandidateRecord.from_structure(nacl_structure, candidate_id="x::1", source="manual_seed")
    src.ml_relaxed = True
    src.ml_energy_per_atom = -3.5
    src.uncertainty_score = 0.04
    dst = CandidateRecord.from_structure(nacl_structure, candidate_id="x::1", source="manual_seed")
    assert dst.ml_relaxed is False
    _copy_relaxed_state(dst, src)
    assert dst.ml_relaxed is True
    assert dst.ml_energy_per_atom == -3.5
    assert dst.uncertainty_score == 0.04


def test_load_checkpoint_filters_done(nacl_structure, tmp_path):
    from matscout.data.store import save_parquet

    done = CandidateRecord.from_structure(nacl_structure, candidate_id="d::1", source="manual_seed")
    done.ml_relaxed = True
    notdone = CandidateRecord.from_structure(nacl_structure, candidate_id="d::2", source="manual_seed")
    p = save_parquet([done, notdone], tmp_path / "ckpt.parquet")
    loaded = _load_checkpoint(str(p))
    assert "d::1" in loaded  # relaxed -> considered done
    assert "d::2" not in loaded  # not relaxed -> not done
