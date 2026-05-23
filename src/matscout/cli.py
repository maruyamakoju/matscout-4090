"""MatScout-4090 command-line interface."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import typer
from rich.console import Console

from . import CAMPAIGNS, __version__
from .config.loader import REPO_ROOT, available_campaigns, load_campaign_config, load_global_config
from .data.store import load_parquet, save_parquet
from .workflows.paths import Paths

warnings.simplefilter("ignore")
app = typer.Typer(add_completion=False, help="MatScout-4090: AI-for-materials discovery pipeline.")
console = Console()


def _campaigns_arg(campaign: str) -> list[str]:
    if campaign == "all":
        return list(available_campaigns() or CAMPAIGNS)
    return [campaign]


@app.command()
def init():
    """Scaffold data/ and outputs/ directories and create .env from .env.example."""
    g = load_global_config()
    for sub in ["raw", "interim", "processed", "external"]:
        (Path(g.data_dir) / sub).mkdir(parents=True, exist_ok=True)
    Path(g.output_dir).mkdir(parents=True, exist_ok=True)
    env = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        console.print(f"[green]Created[/] {env} (add your MP_API_KEY)")
    console.print(f"MatScout v{__version__} initialized. Campaigns: {available_campaigns()}")


@app.command("fetch-mp")
def fetch_mp(campaign: str = typer.Option(...), limit: int = 1000):
    """Fetch Materials Project seed structures (or fixtures if no API key)."""
    g = load_global_config()
    paths = Paths(g)
    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        from .data.mp_client import fetch_mp_seeds

        seeds = fetch_mp_seeds(c, g, limit=limit)
        save_parquet(seeds, paths.seeds(name))
        console.print(f"[green]{name}:[/] {len(seeds)} seeds -> {paths.seeds(name)}")


@app.command()
def generate(campaign: str = typer.Option(...), n: int = typer.Option(None)):
    """Generate candidate structures via substitution + prototype enumeration."""
    g = load_global_config()
    paths = Paths(g)
    from .workflows.campaign import stage_fetch, stage_generate

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        seeds = load_parquet(paths.seeds(name)) if paths.seeds(name).exists() else stage_fetch(c, g, max(1000, n or c.n_generate))
        save_parquet(seeds, paths.seeds(name))
        cands = stage_generate(seeds, c, g, n=n)
        save_parquet(cands, paths.generated(name))
        console.print(f"[green]{name}:[/] {len(cands)} candidates -> {paths.generated(name)}")


@app.command()
def relax(
    campaign: str = typer.Option(...),
    model: str = typer.Option(None, help="override primary model (chgnet|mace_mp|m3gnet)"),
    input: str = typer.Option(None, help="explicit parquet path"),
    limit: int = typer.Option(None),
    no_hull: bool = typer.Option(False, help="skip MP e_above_hull computation"),
):
    """Relax candidates with the ML potential ensemble."""
    g = load_global_config()
    if model:
        g.relaxation.primary_model = model
    paths = Paths(g)
    from .relaxation.ensemble import relax_records

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        src = Path(input) if input else paths.generated(name)
        records = load_parquet(src)
        valid = [r for r in records if not r.rejection_reason]
        relax_records(valid, g, c, limit=limit if limit is not None else c.n_relax,
                      compute_hull=not no_hull)
        save_parquet(records, paths.relaxed(name))
        n_relaxed = sum(1 for r in records if r.ml_relaxed)
        console.print(f"[green]{name}:[/] relaxed {n_relaxed} -> {paths.relaxed(name)}")


@app.command()
def score(campaign: str = typer.Option(...)):
    """Score relaxed candidates with application-specific objectives."""
    g = load_global_config()
    paths = Paths(g)
    from .workflows.campaign import stage_score

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        src = paths.relaxed(name) if paths.relaxed(name).exists() else paths.generated(name)
        records = load_parquet(src)
        seeds = load_parquet(paths.seeds(name)) if paths.seeds(name).exists() else []
        stage_score(records, c, g, seeds=seeds)
        save_parquet(records, paths.scored(name))
        console.print(f"[green]{name}:[/] scored {len(records)} -> {paths.scored(name)}")


@app.command()
def rank(campaign: str = typer.Option(...), top_k: int = typer.Option(200)):
    """Rank scored candidates (Pareto + weighted) and persist the shortlist."""
    g = load_global_config()
    paths = Paths(g)
    from .scoring.ranker import rank_records

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        records = load_parquet(paths.scored(name))
        ranked = rank_records(records, c)
        save_parquet(ranked[:top_k], paths.ranked(name))
        console.print(f"[green]{name}:[/] {len(ranked)} survivors, top-{top_k} -> {paths.ranked(name)}")


@app.command("export-dft")
def export_dft(campaign: str = typer.Option(...), top_k: int = typer.Option(50)):
    """Export top candidates as VASP + QE input decks."""
    g = load_global_config()
    paths = Paths(g)
    from .workflows.export_dft import export_dft_queue

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        ranked = load_parquet(paths.ranked(name))
        export_dft_queue(ranked, c, g, top_k=top_k)


@app.command("catalyst-surfaces")
def catalyst_surfaces(
    top_k: int = typer.Option(8, help="number of top bulk catalysts to analyze"),
    miller: str = typer.Option("111", help="Miller index, e.g. 111 or 100"),
):
    """Surface + adsorbate analysis (HER/OER/CO2RR) on the top bulk catalyst candidates."""
    g = load_global_config()
    paths = Paths(g)
    c = load_campaign_config("catalyst")
    if not paths.ranked("catalyst").exists():
        console.print("[red]No catalyst ranking found.[/] Run `matscout run-campaign --campaign catalyst` first.")
        raise typer.Exit(1)
    ranked = load_parquet(paths.ranked("catalyst"))
    mi = tuple(int(x) for x in miller)
    from .workflows.catalyst_surface import run_catalyst_surfaces, write_surface_report

    results = run_catalyst_surfaces(ranked, c, g, top_k=top_k, miller=mi)
    write_surface_report(results, g)


@app.command()
def report(campaign: str = typer.Option(...)):
    """Write Markdown shortlist report(s)."""
    g = load_global_config()
    paths = Paths(g)
    from .workflows.report import write_campaign_report

    for name in _campaigns_arg(campaign):
        c = load_campaign_config(name)
        ranked = load_parquet(paths.ranked(name))
        write_campaign_report(ranked, c, g)


@app.command("run-campaign")
def run_campaign_cmd(
    campaign: str = typer.Option("all"),
    n_generate: int = typer.Option(None, help="override candidates generated per campaign"),
    relax_limit: int = typer.Option(None, help="cap structures relaxed per campaign"),
    export_top: int = typer.Option(50),
    figures: bool = typer.Option(True),
):
    """Run the full pipeline for one or all campaigns + combined outputs + figures."""
    g = load_global_config()
    from .workflows.campaign import run_campaign
    from .workflows.report import write_top200_and_outputs

    names = _campaigns_arg(campaign)
    all_ranked: dict[str, list] = {}
    campaigns: dict = {}
    for name in names:
        ranked = run_campaign(name, g, n_generate=n_generate, relax_limit=relax_limit,
                              export_top=export_top)
        all_ranked[name] = ranked
        campaigns[name] = load_campaign_config(name)

    write_top200_and_outputs(all_ranked, g, campaigns)
    if figures:
        from .viz.plots import make_all_figures

        figs = make_all_figures(all_ranked, g)
        console.print(f"[green]Figures:[/] {figs}")
    console.rule("[bold green]Done")


if __name__ == "__main__":
    app()
