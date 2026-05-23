"""FastAPI service tests (synthetic data via monkeypatch — CI-safe, no real outputs)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from matscout.service import api  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    df = pd.DataFrame([
        {"candidate_id": "sub::1", "reduced_formula": "Cu2SnS4", "final_score": 0.59,
         "novelty_score": 0.65, "uncertainty_score": 0.0, "ml_e_above_hull": None,
         "predicted_bandgap_ev": 1.08, "source": "prototype_enum", "solar_score": 0.67,
         "structure_cif": "CIFDATA", "tags": ["solar", "prototype_enum"]},
        {"candidate_id": "sub::2", "reduced_formula": "MgAl2O4", "final_score": 0.75,
         "novelty_score": 0.0, "uncertainty_score": 0.19, "ml_e_above_hull": None,
         "predicted_bandgap_ev": 2.34, "source": "manual_seed", "semiconductor_score": 0.75,
         "structure_cif": "CIFDATA2", "tags": ["semiconductor", "manual_seed"]},
    ])
    monkeypatch.setattr(api, "_load_df", lambda: df)
    monkeypatch.setattr(api, "_ranked_path", lambda: __import__("pathlib").Path("dummy"))
    return TestClient(api.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_campaigns(client):
    r = client.get("/campaigns")
    assert r.status_code == 200
    counts = r.json()["campaigns"]
    assert counts.get("solar") == 1 and counts.get("semiconductor") == 1


def test_candidates_filter_by_campaign(client):
    r = client.get("/candidates", params={"campaign": "solar"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["candidates"][0]["reduced_formula"] == "Cu2SnS4"


def test_candidates_min_score(client):
    r = client.get("/candidates", params={"min_score": 0.7})
    assert r.status_code == 200
    assert all(c["final_score"] >= 0.7 for c in r.json()["candidates"])


def test_unknown_campaign_400(client):
    assert client.get("/candidates", params={"campaign": "nope"}).status_code == 400


def test_candidate_detail_and_cif(client):
    r = client.get("/candidate/sub::1")
    assert r.status_code == 200
    assert "structure_cif" not in r.json()  # excluded by default
    r2 = client.get("/candidate/sub::1", params={"include_cif": True})
    assert r2.json()["structure_cif"] == "CIFDATA"
    assert client.get("/candidate/missing").status_code == 404
