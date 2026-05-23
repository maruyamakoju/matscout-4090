"""FastAPI service exposing MatScout outputs (read-only query API).

Run:  uvicorn matscout.service.api:app --reload
      (or `python -m matscout.service.api`)

Serves the ranked candidates, per-campaign shortlists, and the run manifest produced by
`matscout run-campaign`. Read-only; pair with the Streamlit dashboard as a front end.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .. import CAMPAIGNS, __version__
from ..config.loader import load_global_config
from ..workflows.paths import Paths

app = FastAPI(title="MatScout-4090 API", version=__version__,
              description="Read-only query API over MatScout discovery outputs.")

_CAMPAIGN_TAGS = set(CAMPAIGNS)


def _paths() -> Paths:
    return Paths(load_global_config())


@lru_cache(maxsize=1)
def _ranked_path() -> Path:
    return _paths().candidates_ranked()


def _load_df() -> pd.DataFrame:
    p = _ranked_path()
    if not p.exists():
        raise HTTPException(503, "No outputs yet. Run `matscout run-campaign` first.")
    return pd.read_parquet(p)


def _campaign_of(tags) -> str | None:
    for t in tags:
        if t in _CAMPAIGN_TAGS:
            return t
    return None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__, "outputs_ready": _ranked_path().exists()}


@app.get("/campaigns")
def campaigns() -> dict:
    df = _load_df()
    counts: dict[str, int] = {}
    for tags in df["tags"]:
        c = _campaign_of(tags)
        if c:
            counts[c] = counts.get(c, 0) + 1
    return {"campaigns": counts}


@app.get("/manifest")
def manifest() -> dict:
    p = _paths().out / "run_manifest.json"
    if not p.exists():
        raise HTTPException(404, "No run_manifest.json yet.")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/candidates")
def candidates(
    campaign: str | None = Query(None, description="filter by campaign tag"),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=1000),
    dedup_formula: bool = Query(True),
) -> dict:
    df = _load_df()
    if campaign:
        if campaign not in _CAMPAIGN_TAGS:
            raise HTTPException(400, f"unknown campaign '{campaign}'")
        df = df[df["tags"].apply(lambda ts: campaign in ts)]
    df = df[df["final_score"] >= min_score].sort_values("final_score", ascending=False)
    if dedup_formula:
        df = df.drop_duplicates("reduced_formula")
    cols = ["candidate_id", "reduced_formula", "final_score", "novelty_score",
            "uncertainty_score", "ml_e_above_hull", "predicted_bandgap_ev", "source",
            "battery_score", "semiconductor_score", "catalyst_score", "solar_score",
            "co2_capture_score"]
    cols = [c for c in cols if c in df.columns]
    rows = df[cols].head(limit).to_dict(orient="records")
    return {"count": len(rows), "campaign": campaign, "candidates": rows}


@app.get("/candidate/{candidate_id}")
def candidate(candidate_id: str, include_cif: bool = Query(False)) -> dict:
    df = _load_df()
    match = df[df["candidate_id"] == candidate_id]
    if match.empty:
        raise HTTPException(404, f"candidate '{candidate_id}' not found")
    rec = match.iloc[0].to_dict()
    if not include_cif:
        rec.pop("structure_cif", None)
    # numpy/array tags -> list for JSON
    if "tags" in rec and not isinstance(rec["tags"], list):
        rec["tags"] = list(rec["tags"])
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rec.items()}


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
