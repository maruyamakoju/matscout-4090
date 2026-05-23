# MatScout-4090 — Roadmap to serious research & product

The tonight build is a complete, working *prototype*. This roadmap is the honest path to
(A) **research-grade rigor** and (B) **a deployable product**. They share a foundation;
after that they diverge.

## Where we are (prototype, done)
- Full pipeline: generate → validate → CHGNet/MACE relax → score → novelty → rank → DFT export.
- 5 application campaigns, surface-catalyst workflow, active-learning loop.
- 24 fixture prototypes, trained bandgap surrogate (MAE 0.43 eV), 24 tests, ruff clean, CI.
- **Known limitations:** results are fixture-seeded (not real discoveries); e_above_hull is a
  proxy without an MP key; surrogate gaps and catalyst surface energetics are coarse.

---

## Phase 0 — Shared foundation (in progress, no external blockers)
Needed for *both* research and product. Most can be done now.
- [x] CI (ruff + pytest on every push).
- [x] Run manifest / provenance (git SHA, config, lib versions, seeds, timestamps in outputs).
- [x] Checkpoint/resume for relaxation (long runs survive crashes; skip done candidates).
- [x] Dockerfile + pinned environment (reproducible cu128 GPU image).
- [x] `matscout benchmark` harness (relaxation consistency + surrogate holdout).
- [ ] Structured logging + per-run log file.
- [ ] Coverage > 80%; property-based tests for scorers.

## Phase 1 — Real data (unblocks everything; needs MP_API_KEY)
- [ ] Materials Project seeds at scale (thousands of stable/metastable structures per campaign).
- [ ] **Real e_above_hull** via MP phase diagrams + `MaterialsProject2020Compatibility`
      energy corrections (mix CHGNet energies onto the MP-corrected scale carefully).
- [ ] Re-run all campaigns; battery especially becomes credible (NASICON/garnet/argyrodite).
- [ ] Cache + incremental MP sync; respect API limits.

---

## Track A — Research-grade rigor
Goal: defensible, ideally publishable, materials candidates.
1. **Benchmark the method, not just run it.** Adopt Matbench-Discovery-style evaluation:
   stability *classification* metrics (precision/recall/F1, not just energy MAE) on a held-out
   set of known materials. Report the pipeline's discovery precision.
2. **Better surrogates.** Replace composition-only bandgap with structure-aware models
   (MEGNet/M3GNet/matgl bandgap, or fine-tune); add a formation-energy model with proper
   elemental references; calibrate uncertainty (ensemble + conformal intervals).
   _Deferred: matgl's pretrained MEGNet bandgap needs `dgl`, which has no wheel for the
   installed torch 2.11/cu128 (Blackwell). Revisit when dgl supports torch ≥2.11, or pin an
   older torch in a separate env._
3. **Close the DFT loop.** [done: `ingest-dft` parses VASP/QE, writes ML-vs-DFT report, and
   writes DFT energy/gap back into records (`dft_verified.parquet`).] Remaining: recompute true
   e_hull with MP corrections and feed DFT deltas back into the active-learning acquisition.
4. **Catalyst done right.** Surface phase diagrams (terminations, coverage, Pourbaix/applied
   potential, solvation) or an OCP/fairchem adsorption-energy model instead of the bulk proxy.
5. **CO₂ for real.** Ingest CoRE MOF / QMOF; add GCMC (RASPA) or ML isotherm models for
   CO₂/N₂/H₂O selectivity rather than geometric proxies.
6. **Reproducible study.** Fixed datasets, versioned configs, a paper-style report with ablations
   (does generation source / ensemble / novelty term actually help?).

## Track B — Productization
Goal: a robust tool others can run/operate.
1. **Scale & throughput.** Batched/parallel relaxation (multi-GPU, ASE batching), job queue
   (Ray/Dask or SLURM submission), resumable runs, partial-failure isolation. (checkpointing: done)
2. **Service layer.** FastAPI endpoints (submit job, query candidates) + the Streamlit UI as a
   front end; background workers; object storage for artifacts.
3. **Data & experiment management.** DuckDB → a real store; experiment tracking (MLflow/W&B);
   dataset/version lineage; immutable run manifests (done as a start).
4. **Quality gates.** >80% coverage, type-checking (mypy in CI), property tests, perf regression
   tests, security scan; semantic-versioned releases; published package.
5. **Deploy.** Docker image (started) → Compose/K8s; GPU node provisioning; secrets management
   for MP/other keys; observability (logs/metrics/traces).
6. **UX.** Polished dashboard (filters, structure viewer, provenance), exportable reports,
   multi-tenant configs/campaigns.

---

## Recommended sequence
1. **Finish Phase 0** (foundation — happening now).
2. **Get an MP key → Phase 1** (biggest single quality jump; ~hours of compute).
3. **Pick ONE depth-first win:** for research, the **solar absorber shortlist verified by DFT**
   (Cu₂SnS₄ / Cu(Sn,Bi) halides, decks + HSE06 ready); for product, the **FastAPI + scaled
   resumable runner**.
4. Decide the real target (a paper? an internal tool? a startup?) — that choice dictates whether
   Track A or Track B leads. Until then, Phase 0 + Phase 1 are unambiguously the right work.

## Decisions needed from you
- **MP_API_KEY** (free) — unblocks Phase 1, the highest-leverage step.
- **Compute for DFT verification** (MIT cluster / lab / collaborator).
- **End goal** — publication, internal research tool, or commercial product? This sets A vs B priority.
