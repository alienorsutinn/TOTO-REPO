from __future__ import annotations
import argparse
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DIGITS = list("0123456789")

def _zfill4(x: str) -> str:
    s = str(x).strip()
    return s.zfill(4)

def read_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_col = "num" if "num" in df.columns else ("number" if "number" in df.columns else None)
    if num_col is None:
        raise ValueError(f"Missing number column. Have: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["operator"] = df["operator"].astype(str).str.lower().str.strip()
    df["n4"] = df[num_col].astype(str).map(_zfill4)

    # dedupe any duplicated (date,bucket,number)
    df = df.drop_duplicates(subset=["date", "bucket", "n4"], keep="first").copy()
    return df

def all_numbers_0000_9999() -> np.ndarray:
    return np.array([f"{i:04d}" for i in range(10000)], dtype="<U4")

def fit_pos0_probs(train_nums: pd.Series, alpha: float) -> Dict[str, float]:
    d0 = train_nums.str[0]
    counts = d0.value_counts().reindex(DIGITS, fill_value=0).astype(float)
    n = float(counts.sum())
    # Laplace smoothing
    probs = (counts + alpha) / (n + 10.0 * alpha)
    return probs.to_dict()

def rank_by_pos0(pos0_probs: dict[int, float], top_n: int, seed: int) -> list[str]:
    """
    Convert pos0 digit probabilities into a concrete list of numbers to pick.
    Important: all numbers with the same leading digit are tied under this model,
    so we use a deterministic per-day shuffle to avoid pathological fixed blocks.
    """
    # Full universe 0000..9999
    universe = [f"{i:04d}" for i in range(10000)]

    # Order digits by model probability (desc), then digit as stable tie-break
    digits_sorted = sorted(pos0_probs.items(), key=lambda kv: (-kv[1], kv[0]))
    digit_order = [d for d, _ in digits_sorted]

    rng = random.Random(seed)

    picks: list[str] = []
    for d in digit_order:
        # candidates with leading digit d
        cand = [x for x in universe if x[0] == str(d)]
        rng.shuffle(cand)  # deterministic given seed
        need = top_n - len(picks)
        if need <= 0:
            break
        picks.extend(cand[:need])

    # Defensive: ensure unique + length
    picks = list(dict.fromkeys(picks))[:top_n]
    return picks

def monte_carlo_any_hit_rate(
    bt: pd.DataFrame, top_n: int, trials: int, seed: int
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    universe = all_numbers_0000_9999()

    any_hits = []
    avg_hits = []

    for _ in range(trials):
        hits = 0
        any_hit = 0
        for _, row in bt.iterrows():
            winners = row["winners_list"]
            picks = rng.choice(universe, size=top_n, replace=False)
            h = int(np.isin(picks, winners).sum())
            hits += h
            any_hit += int(h > 0)

        any_hits.append(any_hit / len(bt))
        avg_hits.append(hits / len(bt))

    any_hits = np.array(any_hits)
    avg_hits = np.array(avg_hits)
    return {
        "rand_any_hit_rate_mean": float(any_hits.mean()),
        "rand_any_hit_rate_p10": float(np.quantile(any_hits, 0.10)),
        "rand_any_hit_rate_p50": float(np.quantile(any_hits, 0.50)),
        "rand_any_hit_rate_p90": float(np.quantile(any_hits, 0.90)),
        "rand_avg_hits_mean": float(avg_hits.mean()),
        "rand_avg_hits_p10": float(np.quantile(avg_hits, 0.10)),
        "rand_avg_hits_p50": float(np.quantile(avg_hits, 0.50)),
        "rand_avg_hits_p90": float(np.quantile(avg_hits, 0.90)),
    }

def p_values_vs_random(bt: pd.DataFrame, top_n: int, trials: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    universe = all_numbers_0000_9999()

    # observed
    obs_any = float(bt["any_hit"].mean())
    obs_avg = float(bt["hits_count"].mean())

    any_samp = []
    avg_samp = []
    for _ in range(trials):
        hits = 0
        any_hit = 0
        for _, row in bt.iterrows():
            winners = row["winners_list"]
            picks = rng.choice(universe, size=top_n, replace=False)
            h = int(np.isin(picks, winners).sum())
            hits += h
            any_hit += int(h > 0)
        any_samp.append(any_hit / len(bt))
        avg_samp.append(hits / len(bt))

    any_samp = np.array(any_samp)
    avg_samp = np.array(avg_samp)

    return {
        "p_any_hit_rate": float((any_samp >= obs_any).mean()),
        "p_avg_hits_per_day": float((avg_samp >= obs_avg).mean()),
        "trials": int(trials),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-window-days", type=int, default=365)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--random-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/processed/backtest_top3_pos0_model.csv")
    args = ap.parse_args()

    df = read_long(args.numbers_long)
    df = df[df["bucket"] == "top3"].copy()  # IMPORTANT: model and evaluation only on top3
    if df.empty:
        raise ValueError("No top3 rows found.")

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    # group winners by date
    winners_by_date = df.groupby("date")["n4"].apply(list).to_dict()
    all_dates = sorted([d for d in winners_by_date.keys() if start <= d <= end])

    rows = []
    for d in all_dates:
        train_end = d - pd.Timedelta(days=1)
        train_start = d - pd.Timedelta(days=args.train_window_days)

        train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
        if len(train) < 200:
            continue

        probs = fit_pos0_probs(train["n4"].astype(str), alpha=args.alpha)
        picks = rank_by_pos0(probs, top_n=args.top, seed=int(args.seed) + int(d.toordinal()))

        winners = winners_by_date.get(d, [])
        hits = sorted(set(picks).intersection(set(winners)))

        rows.append(
            {
                "date": d,
                "n_train": int(len(train)),
                "winners": ",".join(winners),
                "n_winners": int(len(winners)),
                "picks": ",".join(picks),
                "hits": ",".join(hits),
                "hits_count": int(len(hits)),
                "any_hit": int(len(hits) > 0),
            }
        )

    bt = pd.DataFrame(rows)
    if bt.empty:
        raise ValueError("Backtest produced no rows (maybe too strict windows?).")

    # add list for MC
    bt["winners_list"] = bt["winners"].apply(lambda s: s.split(",") if s else [])

    bt.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    summary = {
        "days_tested": int(len(bt)),
        "any_hit_rate": float(bt["any_hit"].mean()),
        "avg_hits_per_day": float(bt["hits_count"].mean()),
        "avg_hit_rate_per_pick": float(bt["hits_count"].sum() / (len(bt) * args.top)),
        "hits_count_p50": float(np.quantile(bt["hits_count"], 0.50)),
        "hits_count_p90": float(np.quantile(bt["hits_count"], 0.90)),
        "p_hits_ge2": float((bt["hits_count"] >= 2).mean()),
        "p_hits_ge3": float((bt["hits_count"] >= 3).mean()),
    }

    print("\nSummary (TOP3 only):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    pv = p_values_vs_random(bt, top_n=args.top, trials=args.random_trials, seed=args.seed + 999)
    print("\nP-values vs random (>= observed):")
    for k, v in pv.items():
        print(f"  {k}: {v}")

    rb = monte_carlo_any_hit_rate(bt, top_n=args.top, trials=args.random_trials, seed=args.seed + 12345)
    print("\nRandom baseline (TOP3 only):")
    for k, v in rb.items():
        print(f"  {k}: {v}")

    print("\nLast 10 days:")
    print(bt.tail(10)[["date", "hits_count", "any_hit", "hits", "picks"]].to_string(index=False))

if __name__ == "__main__":
    main()
