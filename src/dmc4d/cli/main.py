from __future__ import annotations
from pathlib import Path
import typer
from rich import print

from dmc4d.backtest.engine import BacktestConfig, run_backtest
from dmc4d.backtest.report import write_reports
from dmc4d.config import CFG
from dmc4d.crowd.crowd_model import explain_number, score_all_numbers
from dmc4d.datasets.births_store import build_birth_maps
from dmc4d.datasets.results_store import upsert_results
from dmc4d.data_sources.local_csv import LocalCSVResults
from dmc4d.strategies.factory import create_strategy

app = typer.Typer(add_completion=False)

@app.command("build-births")
def cmd_build_births(
    input: Path = typer.Option(...),
    out: Path = typer.Option(Path("data/processed")),
    year_min: int = typer.Option(1950),
    year_max: int = typer.Option(2007),
):
    paths = build_birth_maps(input, out, year_min=year_min, year_max=year_max)
    print({"ok": True, "outputs": {k: str(v) for k, v in paths.items()}})

@app.command("fetch-results")
def cmd_fetch_results(
    source: str = typer.Option("local"),
    input: Path = typer.Option(...),
    out: Path = typer.Option(Path("data/processed")),
):
    if source.lower() != "local":
        raise typer.BadParameter("Only source=local implemented")
    rows = LocalCSVResults(input).load()
    out_path = out / "results.csv"
    upsert_results(rows, out_path)
    print({"ok": True, "results_csv": str(out_path), "rows": len(rows)})

@app.command("score-numbers")
def cmd_score_numbers(
    births_dir: Path = typer.Option(Path("data/processed")),
    out: Path = typer.Option(Path("data/processed/crowd_scores.csv")),
):
    score_all_numbers(births_dir, out)
    print({"ok": True, "crowd_scores": str(out)})

@app.command("explain-number")
def cmd_explain_number(
    number: str = typer.Argument(...),
    births_dir: Path = typer.Option(Path("data/processed")),
):
    info = explain_number(number, births_dir)
    print(info)

@app.command("backtest")
def cmd_backtest(
    results: Path = typer.Option(CFG.results_csv),
    strategy: str = typer.Option("low_crowd"),
    budget: int = typer.Option(20),
    start: str = typer.Option(CFG.start_date),
    end: str = typer.Option(""),
    report_dir: Path = typer.Option(Path("reports")),
    seed: int = typer.Option(0),
):
    import pandas as pd
    strat = create_strategy(strategy, seed=seed)

    df = pd.read_csv(results)
    end_used = end.strip() or str(df["date"].max())

    per_draw = run_backtest(results, strat, BacktestConfig(start=start, end=end_used, budget_rm=budget))
    out = write_reports(per_draw, report_dir / f"{strategy}_budget{budget}_{start}_to_{end_used}")
    print(out["summary_obj"])

def main():
    app()

if __name__ == "__main__":
    main()
