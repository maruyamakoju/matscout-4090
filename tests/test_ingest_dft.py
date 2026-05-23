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
