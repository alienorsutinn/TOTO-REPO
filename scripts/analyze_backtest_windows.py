# scripts/analyze_backtest_windows.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_picks(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    # Wilson score interval for a binomial proportion
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_hit_rate(any_hit: np.ndarray, n_boot: int = 5000, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(any_hit)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = rng.integers(0, n, size=(n_boot, n))
    rates = any_hit[samples].mean(axis=1)
    return (float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to backtest output CSV (e.g. data/processed/backtest_multifeature.csv)")
    ap.add_argument("--freq", default="M", help="Pandas resample frequency: M=month, W=week, 14D, 28D, etc.")
    ap.add_argument("--bootstrap", type=int, default=0, help="If >0, run bootstrap CI for any_hit_rate with N resamples.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise SystemExit("CSV missing required column: date")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Sanity checks
    min_d, max_d = df["date"].min(), df["date"].max()
    print(f"[sanity] rows={len(df)} date_min={min_d.date()} date_max={max_d.date()}")

    for col in ["hits_count", "any_hit", "picks"]:
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column: {col}")

    # Parse picks and compute duplication stats
    picks_list = df["picks"].apply(parse_picks)
    df["n_picks"] = picks_list.apply(len)
    df["n_unique"] = picks_list.apply(lambda xs: len(set(xs)))
    df["dup_rate"] = 1.0 - (df["n_unique"] / df["n_picks"].replace(0, np.nan))

    # Window summary
    df = df.set_index("date")
    g = df.resample(args.freq)

    out = pd.DataFrame({
        "days_tested": g.size(),
        "hit_days": g["any_hit"].sum(),
        "any_hit_rate": g["any_hit"].mean(),
        "avg_hits_per_day": g["hits_count"].mean(),
        "avg_n_picks": g["n_picks"].mean(),
        "avg_n_unique": g["n_unique"].mean(),
        "avg_dup_rate": g["dup_rate"].mean(),
    })

    # Wilson CI per window
    cis = []
    for idx, row in out.iterrows():
        lo, hi = wilson_ci(int(row["hit_days"]), int(row["days_tested"]))
        cis.append((lo, hi))
    out["hit_rate_ci_lo"] = [x[0] for x in cis]
    out["hit_rate_ci_hi"] = [x[1] for x in cis]

    print("\n=== Window summary ===")
    print(out.to_string(float_format=lambda x: f"{x:.4f}"))

    # Overall CI + optional bootstrap
    overall_k = int(df["any_hit"].sum())
    overall_n = int(len(df))
    lo, hi = wilson_ci(overall_k, overall_n)
    print(f"\n[overall] any_hit_rate={overall_k/overall_n:.4f} (Wilson 95% CI: {lo:.4f}–{hi:.4f})")

    if args.bootstrap > 0:
        any_hit = df["any_hit"].to_numpy(dtype=float)
        blo, bhi = bootstrap_hit_rate(any_hit, n_boot=args.bootstrap)
        print(f"[overall] bootstrap 95% CI (n={args.bootstrap}): {blo:.4f}–{bhi:.4f}")


if __name__ == "__main__":
    main()
