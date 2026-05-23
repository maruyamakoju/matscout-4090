# MatScout-4090

Local high-throughput AI-for-materials discovery pipeline for inorganic materials across
**battery, semiconductor, catalyst, solar, and CO₂-capture** applications.

```
existing DB  ->  candidate generation  ->  structure validity  ->  ML relaxation
  ->  stability / property / application scoring  ->  novelty  ->  top CIF/POSCAR + reports
  ->  DFT/experiment queue
```

Generation, ML relaxation, scoring and ranking run locally (RTX 4090/5090 class GPU).
The final DFT verification is exported as ready-to-run VASP/QE input decks for a cluster.

## Quickstart

```bash
uv venv --python 3.11 .venv
uv pip install -e .            # light core: pymatgen, ase, duckdb, typer ...
# optional heavy deps:
uv pip install -e ".[ml]"      # chgnet, mace-torch, matgl  (needs cu128 torch on Blackwell)
uv pip install -e ".[data]"    # mp-api, matminer

matscout init
matscout run-campaign --campaign all
```

Without a Materials Project API key, `fetch-mp` falls back to bundled fixture structures and
prints a clear warning. Without `torch`/`chgnet`/`mace`, relaxation runs in **dry mode**
(structures pass through, flagged `ml_relaxed=false`) so the full pipeline still produces output.

## Pipeline commands

| command | what it does |
|---|---|
| `matscout init` | scaffold `data/`, `outputs/`, copy `.env` |
| `matscout fetch-mp --campaign battery --limit 1000` | seed structures from Materials Project (or fixtures) |
| `matscout generate --campaign battery --n 5000` | substitution + prototype enumeration |
| `matscout relax --model chgnet --input ...parquet` | CHGNet/MACE relaxation ensemble |
| `matscout score --campaign battery` | application-specific scoring |
| `matscout rank --campaign battery --top-k 200` | Pareto + weighted ranking |
| `matscout export-dft --campaign battery --top-k 50` | VASP/QE input decks |
| `matscout report --campaign battery` | Markdown shortlists + figures |
| `matscout run-campaign --campaign all` | full pipeline, all campaigns (writes `run_manifest.json`) |
| `matscout catalyst-surfaces --top-k 8` | slab + adsorbate HER/OER/CO2RR analysis on top catalysts |
| `matscout active-loop --campaign solar` | active-learning loop: select → mutate → relax → score (spec §8) |
| `matscout ingest-dft --campaign solar` | parse completed DFT runs, write ML-vs-DFT comparison |
| `matscout benchmark` | validate pipeline: surrogate accuracy, relaxation sanity, determinism |

Relaxation **checkpoints/resumes** automatically (`--relax-limit` bounds long runs). See
`docs/DFT_VERIFICATION.md` for cluster submission and `ROADMAP.md` for the research/product plan.

### Service + dashboard

```bash
uvicorn matscout.service.api:app          # read-only query API (uv pip install -e ".[service]")
streamlit run src/matscout/viz/dashboard.py   # interactive candidate explorer
```

### Reproducible (Docker, GPU)

```bash
docker build -t matscout:latest .
docker run --gpus all -e MP_API_KEY=$MP_API_KEY -v $PWD/outputs:/app/outputs \
  matscout:latest run-campaign --campaign all
```

## Outputs

```
outputs/
  candidates_all.parquet        candidates_ranked.parquet
  top_200/ *.cif *.poscar metadata.json
  dft_queue/{battery,semiconductor,catalyst,solar,co2_capture}/
  reports/  executive_summary.md  *_shortlist.md
  figures/  pareto_fronts.png  stability_vs_novelty.png  composition_maps.png
```

See `configs/` for global + per-campaign settings.
