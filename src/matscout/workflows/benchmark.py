"""Validation harness: does the pipeline behave correctly? (`matscout benchmark`)

Three checks, all runnable without an MP key:
1. Bandgap surrogate vs. known experimental gaps (accuracy + metal/non-metal separation).
2. Relaxation sanity on known-stable prototypes (convergence, forces, volume drift).
3. Ranking determinism (same seed -> same order).

This is the scaffold for research-grade evaluation; with an MP key it extends naturally to
Matbench-Discovery-style stability classification (precision/recall on the convex hull).
"""

from __future__ import annotations

import json

from rich.console import Console

from ..config.schema import GlobalConfig
from .paths import Paths

console = Console()

# Reference experimental bandgaps (eV) for a chemically diverse spot-check.
KNOWN_GAPS = {
    "Si": 1.12, "Ge": 0.66, "GaAs": 1.42, "GaN": 3.40, "ZnO": 3.37, "MgO": 7.8,
    "NaCl": 8.5, "TiO2": 3.0, "CdTe": 1.49, "ZnS": 3.68, "AlN": 6.0, "SiC": 2.36,
    "Cu2O": 2.17, "Fe": 0.0, "Al": 0.0, "Cu": 0.0,
}


def bandgap_benchmark() -> dict:
    from pymatgen.core import Composition

    from ..properties.bandgap import predict_bandgap

    errs, rows = [], []
    metal_ok = 0
    metals = 0
    model = "n/a"
    for formula, ref in KNOWN_GAPS.items():
        gap, _direct, m, _ = predict_bandgap(Composition(formula))
        if ref > 0:  # report the model used for actual semiconductors, not the metal path
            model = m
        err = abs(gap - ref)
        errs.append(err)
        rows.append({"formula": formula, "ref": ref, "pred": round(gap, 2), "abs_err": round(err, 2)})
        if ref == 0.0:
            metals += 1
            if gap < 0.3:
                metal_ok += 1
    mae = sum(errs) / len(errs)
    return {
        "model": model,
        "mae_ev": round(mae, 3),
        "n": len(KNOWN_GAPS),
        "metal_detection": f"{metal_ok}/{metals}",
        "rows": rows,
    }


def relaxation_benchmark(g: GlobalConfig, n: int = 6) -> dict:
    from ..data.fixtures import seed_structures
    from ..relaxation.chgnet_relaxer import CHGNetRelaxer

    relaxer = CHGNetRelaxer(device=g.device, max_steps=200)
    if not relaxer.available():
        return {"status": "skipped (no torch/chgnet)"}
    seeds = seed_structures()[:n]
    conv, forces, dvol = 0, [], []
    rows = []
    for label, _fam, s in seeds:
        res = relaxer.relax(s)
        if res.ok:
            conv += int(res.converged)
            forces.append(res.max_force_ev_a)
            dvol.append(abs(res.volume_change_pct))
            rows.append({"label": label, "E_per_atom": round(res.energy_per_atom, 3),
                         "maxF": round(res.max_force_ev_a, 3),
                         "dVol_pct": round(res.volume_change_pct, 1)})
    return {
        "status": "ok",
        "model": relaxer.name,
        "converged": f"{conv}/{len(seeds)}",
        "mean_maxF_ev_a": round(sum(forces) / len(forces), 4) if forces else None,
        "mean_abs_dvol_pct": round(sum(dvol) / len(dvol), 2) if dvol else None,
        "rows": rows,
    }


def determinism_benchmark(g: GlobalConfig) -> dict:
    from ..config.loader import load_campaign_config
    from ..data.mp_client import fetch_mp_seeds
    from ..generation.generate import generate_candidates
    from ..scoring.objectives import score_records
    from ..scoring.ranker import rank_records
    from ..validation.novelty import NoveltyIndex

    c = load_campaign_config("solar")
    seeds = fetch_mp_seeds(c, g, limit=50)
    idx = NoveltyIndex.from_records(seeds)

    def order():
        cands = generate_candidates(seeds, c, g, n=120)
        score_records(cands, c, g, novelty_index=idx)
        return [r.candidate_id for r in rank_records(cands, c)]

    o1, o2 = order(), order()
    return {"deterministic": o1 == o2, "n": len(o1)}


def run_benchmark(g: GlobalConfig) -> dict:
    console.rule("[bold]MatScout benchmark")
    result = {
        "bandgap_surrogate": bandgap_benchmark(),
        "relaxation": relaxation_benchmark(g),
        "determinism": determinism_benchmark(g),
    }
    bg = result["bandgap_surrogate"]
    console.print(f"[cyan]Bandgap[/] ({bg['model']}): MAE={bg['mae_ev']} eV over {bg['n']} known, "
                  f"metal detection {bg['metal_detection']}")
    rx = result["relaxation"]
    if rx.get("status") == "ok":
        console.print(f"[cyan]Relaxation[/] ({rx['model']}): converged {rx['converged']}, "
                      f"mean|maxF|={rx['mean_maxF_ev_a']} eV/Å, mean|ΔV|={rx['mean_abs_dvol_pct']}%")
    else:
        console.print(f"[yellow]Relaxation:[/] {rx['status']}")
    console.print(f"[cyan]Determinism[/]: {'PASS' if result['determinism']['deterministic'] else 'FAIL'}")

    out = Paths(g).out
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green]Benchmark written:[/] {out / 'benchmark.json'}")
    return result
