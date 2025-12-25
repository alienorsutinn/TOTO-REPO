from __future__ import annotations
from pathlib import Path
import pandas as pd
from dmc4d.logging import get_logger

log = get_logger(__name__)

def build_birth_maps(births_csv: Path, out_dir: Path, year_min: int = 1950, year_max: int = 2007) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(births_csv)
    if "date" not in df.columns or "births" not in df.columns:
        raise ValueError("births csv must contain columns: date,births")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["monthday"] = df["date"].dt.strftime("%m-%d")

    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    by_year = df.groupby("year", as_index=False)["births"].sum()
    by_md = df.groupby("monthday", as_index=False)["births"].sum()

    by_year["p"] = by_year["births"] / by_year["births"].sum()
    by_md["p"] = by_md["births"] / by_md["births"].sum()

    out_year = out_dir / "births_by_year.csv"
    out_md = out_dir / "births_by_monthday.csv"
    by_year.to_csv(out_year, index=False)
    by_md.to_csv(out_md, index=False)
    log.info("Saved births maps -> %s, %s", out_year, out_md)
    return {"births_by_year": out_year, "births_by_monthday": out_md}
