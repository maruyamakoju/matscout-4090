# Contributing to MatScout-4090

Thanks for your interest! This guide covers local setup and the quality gates.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"          # light core + pytest/ruff/mypy/fastapi
# optional, for actual relaxation / data:
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # Blackwell/cu128
uv pip install chgnet "mace-torch>=0.3.6" matminer
```

The pipeline runs without the heavy ML deps (relaxation degrades to a dry pass, the bandgap
surrogate falls back to a heuristic), so the test suite is CPU-only.

## Quality gates (must pass; CI enforces all three)

```bash
ruff check src tests scripts     # lint + import order
mypy                             # type check (pragmatic config in pyproject.toml)
pytest -q                        # 56 tests, CPU-only
```

- Run `ruff check --fix` and `ruff format` before committing.
- Keep mypy clean. For unavoidable pymatgen/ase stub mismatches, use a *targeted*
  `# type: ignore[code]` (not a blanket ignore).
- Add tests for new logic. Keep tests **CI-safe**: no GPU, no network, no MP key. Use
  synthetic data / monkeypatch for anything that would need torch or the Materials Project.

## Conventions

- `CandidateRecord` (in `data/schemas.py`) is the one record type flowing through every stage.
- Application scorers expose `compute_features / score_candidate(record, campaign, ctx) /
  explain_score`, and their component keys must match the campaign YAML `objectives:` keys.
- **Never fabricate data.** Missing MP key → fixtures + warning; missing torch → dry relax;
  missing pore descriptors → low-confidence CO₂ score. Preserve this contract.
- ML energies / e_above_hull / bandgaps are surrogate estimates — label them as such.

## Architecture

See `CLAUDE.md` for the module map and `ROADMAP.md` for the research/product plan.

## Commits / PRs

- Small, focused commits; imperative subject lines.
- Open a PR against `master`; CI (ruff + mypy + pytest) must be green.
