"""Figure generation: Pareto fronts, stability-vs-novelty, composition maps."""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config.schema import GlobalConfig  # noqa: E402
from ..data.schemas import CandidateRecord  # noqa: E402
from ..workflows.paths import Paths  # noqa: E402

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _pareto_rank(r: CandidateRecord) -> int:
    for t in r.tags:
        if t.startswith("pareto_rank:"):
            try:
                return int(t.split(":")[1])
            except ValueError:
                return 99
    return 99


def stability_vs_novelty(all_ranked: dict[str, list[CandidateRecord]], g: GlobalConfig) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, recs) in enumerate(all_ranked.items()):
        xs = [(r.ml_e_above_hull if r.ml_e_above_hull is not None else 0.15) for r in recs]
        ys = [r.novelty_score for r in recs]
        ss = [20 + 120 * (r.final_score or 0) for r in recs]
        ax.scatter(xs, ys, s=ss, alpha=0.5, label=name, color=_COLORS[i % len(_COLORS)],
                   edgecolors="none")
    ax.axvline(0.05, ls="--", color="gray", lw=1)
    ax.set_xlabel("e_above_hull (eV/atom)  [proxy 0.15 if unknown]")
    ax.set_ylabel("novelty score")
    ax.set_title("Stability vs Novelty (marker size ∝ final score)")
    ax.legend(fontsize=8)
    out = Paths(g).figures() / "stability_vs_novelty.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def pareto_fronts(all_ranked: dict[str, list[CandidateRecord]], g: GlobalConfig) -> str:
    n = len(all_ranked)
    cols = min(3, n) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (name, recs) in enumerate(all_ranked.items()):
        ax = axes[idx // cols][idx % cols]
        xs = [r.novelty_score for r in recs]
        ys = [r.final_score or 0 for r in recs]
        ranks = [_pareto_rank(r) for r in recs]
        sc = ax.scatter(xs, ys, c=ranks, cmap="viridis_r", s=30, alpha=0.7)
        front = [(x, y) for x, y, rk in zip(xs, ys, ranks) if rk == 0]
        if front:
            front.sort()
            ax.plot([p[0] for p in front], [p[1] for p in front], "r-", lw=1, alpha=0.6)
        ax.set_title(name); ax.set_xlabel("novelty"); ax.set_ylabel("final score")
        fig.colorbar(sc, ax=ax, label="Pareto rank")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    out = Paths(g).figures() / "pareto_fronts.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def composition_maps(all_ranked: dict[str, list[CandidateRecord]], g: GlobalConfig, top_n: int = 50) -> str:
    n = len(all_ranked)
    cols = min(3, n) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for idx, (name, recs) in enumerate(all_ranked.items()):
        ax = axes[idx // cols][idx % cols]
        counter: Counter[str] = Counter()
        for r in recs[:top_n]:
            counter.update(r.elements)
        common = counter.most_common(15)
        if common:
            els, freqs = zip(*common)
            ax.bar(range(len(els)), freqs, color=_COLORS[idx % len(_COLORS)])
            ax.set_xticks(range(len(els))); ax.set_xticklabels(els, rotation=45)
        ax.set_title(f"{name}: element freq (top {top_n})")
        ax.set_ylabel("count")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    out = Paths(g).figures() / "composition_maps.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def make_all_figures(all_ranked: dict[str, list[CandidateRecord]], g: GlobalConfig) -> list[str]:
    return [
        pareto_fronts(all_ranked, g),
        stability_vs_novelty(all_ranked, g),
        composition_maps(all_ranked, g),
    ]
