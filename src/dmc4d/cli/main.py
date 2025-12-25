from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer

from dmc4d.data_sources.businesslist import BusinessListResults
from dmc4d.data_sources.local_csv import LocalCSVResults
from dmc4d.datasets.results_store import upsert_results

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: str = typer.Option("businesslist", "--source"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    rate_per_sec: float = typer.Option(1.0, "--rate-per-sec"),
    out: str = typer.Option("data/raw", "--out"),
    input: str | None = typer.Option(None, "--input"),
) -> dict:
    """
    Fetch 4D results into a CSV.

    Examples:
      python -m dmc4d.cli.main --start 2022-01-01 --end 2022-02-28 --out data/raw
      python -m dmc4d.cli.main --source local --input data/raw/results.csv --start 2022-01-01 --end 2022-02-28
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"

    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()

    if source == "businesslist":
        rows = BusinessListResults(rate_per_sec=rate_per_sec).fetch_range(start_d, end_d)
    elif source == "local":
        if not input:
            raise typer.BadParameter("--input is required when --source local")
        rows = LocalCSVResults(path=Path(input)).fetch_range(start_d, end_d)
    else:
        raise typer.BadParameter("Only --source businesslist or local supported")

    upsert_results(rows, out_path)

    if not rows:
        print("No rows parsed", flush=True)
        return {
            "ok": False,
            "rows": 0,
            "msg": "No rows parsed (check scraper/parser).",
            "results_csv": str(out_path),
        }

    return {"ok": True, "rows": len(rows), "results_csv": str(out_path)}


if __name__ == "__main__":
    app()
