# CLAUDE.md — MatScout-4090

Guidance for working in this repo. Read this first.

## What this is
Local high-throughput AI-for-materials discovery pipeline. Flow:
`MP/fixtures seeds -> generate (substitution+prototype) -> validate -> ML relax
(CHGNet+MACE) -> application scoring -> novelty -> rank -> DFT-ready exports + reports`.
Five campaigns: battery, semiconductor, catalyst, solar, co2_capture.

## Environment
- Windows 11, Python 3.11, venv at `.venv` (use `.venv/Scripts/python.exe`).
- GPU: RTX 5090 (Blackwell) — torch is the **cu128** build (`torch 2.11.0+cu128`).
- Package manager: `uv`. Install: `uv pip install -e . --python .venv/Scripts/python.exe`.
- Heavy ML deps already installed: torch (cu128), chgnet, mace-torch.
- Run tests: `.venv/Scripts/python.exe -m pytest tests/ -q`. Lint: `ruff check src tests`.

## Architecture (src/matscout/)
- `config/` — pydantic schemas (`GlobalConfig`, `CampaignConfig`) + YAML loader. Configs in `configs/`.
- `data/` — `schemas.py` (the central **CandidateRecord** + **ScoreResult**), `elements.py`
  (toxicity/abundance/cost tables + substitution families), `mp_client.py` (MP fetch with
  **fixture fallback** when no `MP_API_KEY`), `fixtures.py` (13 real prototype seeds),
  `store.py` (Parquet+DuckDB), `core_mof.py` (CO₂ MOF loader), `ocp_loader.py` (catalyst stub).
- `generation/` — `substitution.py`, `prototype_enum.py`, `mutate.py`, `mattergen_runner.py`,
  orchestrated by `generate.py`. `pools.py` builds allowed-element pools per campaign.
- `validation/` — `structure_checks.py` (overlap/collapse), `chemistry_checks.py`,
  `novelty.py` (`NoveltyIndex`), `phase_diagram.py` (e_above_hull via MP, `None` w/o key).
- `relaxation/` — `base.py` (shared ASE/FIRE loop), `chgnet_relaxer.py`, `mace_relaxer.py`,
  `matgl_relaxer.py`, `ensemble.py` (primary+secondary, uncertainty, KEEP/MAYBE/KILL).
  **Degrades gracefully**: no torch -> dry pass (`ml_relaxed=False`).
- `properties/` — one module per campaign (`battery/semiconductor/catalyst/solar/co2_capture`),
  each exposing `compute_features / score_candidate(record, campaign, ctx) / explain_score`.
  Shared: `common.py` (assemble_score, target/peak windows), `stability.py`, `bandgap.py`
  (heuristic surrogate), `synthesizability.py`.
- `scoring/` — `objectives.py` (routes to app scorer + cross-cutting penalties + final_score),
  `pareto.py`, `uncertainty.py` (acquisition + wildcards), `ranker.py` (deterministic sort,
  diversity clustering, wildcard selection).
- `workflows/` — `campaign.py` (per-stage + `run_campaign`), `export_dft.py` (VASP+QE decks,
  priority A/B/C), `report.py` (shortlists + top-200 + executive summary), `active_loop.py`,
  `paths.py` (output layout).
- `viz/` — `plots.py` (3 figures), `dashboard.py` (streamlit).
- `cli.py` — Typer app: `init, fetch-mp, generate, relax, score, rank, export-dft, report,
  run-campaign`.

## Key conventions
- **CandidateRecord** is the one record type flowing through every stage; structures are stored
  as CIF strings (`get_structure()` to parse). Persisted as Parquet.
- Application scorer component-dict keys MUST match the campaign YAML `objectives:` keys
  (that's how `assemble_score` weights them).
- **Never fabricate data**: missing MP key -> fixtures + warning; missing torch -> dry relax;
  missing pore descriptors -> CO₂ score with low confidence. Keep this contract.
- e_above_hull and bandgap are **surrogate estimates** (flagged), not DFT.

## Gotchas
- `replace_species` mutates a copy in generation; dedup by `structure_hash`.
- Generation must be single-pass/finite (an earlier `itertools.cycle` caused a hang) — keep it.
- With only the 13 fixture seeds, generation caps around ~200-260 candidates/campaign; with a
  real `MP_API_KEY` it scales to thousands.
- Relaxing prototype-enumerated structures is slow (many run the full 300 FIRE steps); use
  `--relax-limit` to bound wall-time.
