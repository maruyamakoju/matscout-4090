"""DFT ingestion: comparison-report logic + QE parser (CI-safe, no DFT binaries)."""

from __future__ import annotations

from matscout.config.schema import GlobalConfig
from matscout.workflows.ingest_dft import DFTResult, parse_qe, write_comparison_report


def test_comparison_report_renders(tmp_path):
    g = GlobalConfig(output_dir=str(tmp_path))
    results = [
        DFTResult(deck="B_001_Cu2SnS4_x", formula="Cu2SnS4", source="vasp",
                  energy_per_atom=-4.32, bandgap_ev=0.95, is_direct=True, converged=True,
                  ml={"predicted_bandgap_ev": 1.08, "formula": "Cu2SnS4"}),
        DFTResult(deck="B_002_NaCl_x", formula="NaCl", source="qe",
                  energy_per_atom=-3.10, bandgap_ev=None, converged=False, ml={}),
    ]
    path = write_comparison_report(results, g)
    text = open(path, encoding="utf-8").read()
    assert "Cu2SnS4" in text
    assert "0.95" in text  # DFT gap
    assert "1.08" in text  # ML gap
    assert "0.13" in text  # |delta gap|
    assert "yes" in text and "no" in text  # convergence flags


def test_parse_qe_extracts_energy(tmp_path):
    qe = tmp_path / "deck" / "qe"
    qe.mkdir(parents=True)
    (qe / "relax.out").write_text(
        "some preamble\n"
        "!    total energy              =    -100.50000000 Ry\n"
        "highest occupied, lowest unoccupied level (ev):     2.0000    3.2000\n"
        "JOB DONE.\n",
        encoding="utf-8",
    )
    res = parse_qe(qe)
    assert res is not None
    assert res.converged is True
    assert res.energy_per_atom is not None and res.energy_per_atom < 0
    assert abs(res.bandgap_ev - 1.2) < 1e-6


def test_parse_qe_missing_returns_none(tmp_path):
    assert parse_qe(tmp_path) is None


def test_update_records_with_dft(tmp_path, nacl_structure):
    from matscout.config.schema import GlobalConfig
    from matscout.data.schemas import CandidateRecord
    from matscout.data.store import load_parquet, save_parquet
    from matscout.workflows.ingest_dft import update_records_with_dft
    from matscout.workflows.paths import Paths

    g = GlobalConfig(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"))
    rec = CandidateRecord.from_structure(nacl_structure, candidate_id="sub::1", source="manual_seed")
    save_parquet([rec], Paths(g).ranked("solar"))

    results = [DFTResult(deck="d", formula="NaCl", source="vasp", candidate_id="sub::1",
                         energy_per_atom=-3.6, bandgap_ev=8.4, converged=True)]
    n = update_records_with_dft(results, "solar", g)
    assert n == 1
    updated = load_parquet(Paths(g).data / "processed" / "solar" / "dft_verified.parquet")
    assert updated[0].dft_verified is True
    assert updated[0].dft_energy_per_atom == -3.6
    assert updated[0].dft_bandgap_ev == 8.4
    assert "dft:verified" in updated[0].tags
