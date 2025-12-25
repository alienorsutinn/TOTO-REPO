#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def fmt4(x: int) -> str:
    return f"{int(x):04d}"


def digits_of_str(s: str) -> List[int]:
    return [ord(c) - 48 for c in s]


def chi2_vs_uniform_digit(counts_10: np.ndarray) -> float:
    n = counts_10.sum()
    if n <= 0:
        return 0.0
    exp = n / 10.0
    return float(((counts_10 - exp) ** 2 / exp).sum())


def smoothed_digit_probs(train_n4: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.zeros((4, 10), dtype=np.float64)
    for s in train_n4:
        ds = digits_of_str(s)
        for p in range(4):
            counts[p, ds[p]] += 1.0
    n = counts.sum(axis=1, keepdims=True)
    probs = (counts + alpha) / (n + 10.0 * alpha)
    return probs


def score_universe(probs: np.ndarray) -> np.ndarray:
    lp = np.log(probs + 1e-300)
    u = np.arange(10000, dtype=np.int32)
    d0 = (u // 1000) % 10
    d1 = (u // 100) % 10
    d2 = (u // 10) % 10
    d3 = u % 10
    return lp[0, d0] + lp[1, d1] + lp[2, d2] + lp[3, d3]


def top_n_from_scores(log_scores: np.ndarray, top_n: int) -> List[str]:
    idx = np.argpartition(-log_scores, top_n - 1)[:top_n]
    idx = idx[np.lexsort((idx, -log_scores[idx]))]
    return [f"{i:04d}" for i in idx.tolist()]


@dataclass
class DayRow:
    date: str
    traded: int
    gate_score: float
    gate_thresh: float
    hits_count: int
    any_hit: int
    hits: str
    picks: str


def build_gate_and_picks(
    df: pd.DataFrame,
    dates: List[pd.Timestamp],
    train_window_days: int,
    top_n: int,
    alpha: float,
    gate_quantile: float,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Returns:
      out_df: per-day rows with traded/gate/picks (hits empty for now)
      picks_by_day: list aligned to dates; picks list if traded else []
    """
    past_gate_scores: List[float] = []
    rows: List[DayRow] = []
    picks_by_day: List[List[str]] = []

    for dt in dates:
        train_start = dt - pd.Timedelta(days=train_window_days)
        train_end = dt - pd.Timedelta(days=1)
        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        train_n4 = train_df["n4"].to_numpy(dtype=object)

        if len(train_n4) < 50:
            rows.append(DayRow(
                date=str(dt.date()),
                traded=0,
                gate_score=float("nan"),
                gate_thresh=float("nan"),
                hits_count=0,
                any_hit=0,
                hits="",
                picks="",
            ))
            picks_by_day.append([])
            continue

        pos0 = np.fromiter((digits_of_str(s)[0] for s in train_n4), dtype=np.int32)
        counts0 = np.bincount(pos0, minlength=10).astype(np.float64)
        gate_score = chi2_vs_uniform_digit(counts0)

        if len(past_gate_scores) >= 30:
            gate_thresh = float(np.quantile(np.array(past_gate_scores, dtype=np.float64), gate_quantile))
        else:
            gate_thresh = gate_score

        traded = int(gate_score >= gate_thresh)

        if traded:
            probs = smoothed_digit_probs(train_n4, alpha=alpha)
            log_scores = score_universe(probs)
            picks = top_n_from_scores(log_scores, top_n=top_n)
            picks_by_day.append(picks)
            rows.append(DayRow(
                date=str(dt.date()),
                traded=1,
                gate_score=gate_score,
                gate_thresh=gate_thresh,
                hits_count=0,
                any_hit=0,
                hits="",
                picks=",".join(picks),
            ))
        else:
            picks_by_day.append([])
            rows.append(DayRow(
                date=str(dt.date()),
                traded=0,
                gate_score=gate_score,
                gate_thresh=gate_thresh,
                hits_count=0,
                any_hit=0,
                hits="",
                picks="",
            ))

        past_gate_scores.append(gate_score)

    return pd.DataFrame([r.__dict__ for r in rows]), picks_by_day


def eval_hits(picks_by_day: List[List[str]], actual_by_day: List[List[str]]) -> Tuple[int, int, float, float]:
    # returns: traded_days, traded_any_hits, any_hit_rate_when_traded, avg_hits_when_traded
    traded_days = 0
    traded_any = 0
    traded_hits_total = 0

    for picks, actual in zip(picks_by_day, actual_by_day):
        if not picks:
            continue
        traded_days += 1
        hitset = set(actual)
        hits = sum(1 for x in picks if x in hitset)
        traded_hits_total += hits
        if hits > 0:
            traded_any += 1

    any_rate = (traded_any / traded_days) if traded_days else 0.0
    avg_hits = (traded_hits_total / traded_days) if traded_days else 0.0
    return traded_days, traded_any, any_rate, avg_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="top3", choices=["top3", "starter", "consolation", "all"])
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-24")
    ap.add_argument("--train-window-days", type=int, default=365)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gate-quantile", type=float, default=0.85)
    ap.add_argument("--random-trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/regime_gate_backtest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["n4"] = df["num"].astype(int).map(fmt4)

    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket]

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    by_date = df.groupby("date")["n4"].apply(list).sort_index()
    dates = by_date.index.to_list()
    actual_by_day = [sorted(by_date.loc[d]) for d in dates]

    if not dates:
        raise SystemExit("No data after filtering; check bucket/start/end.")

    out, picks_by_day = build_gate_and_picks(
        df=df,
        dates=dates,
        train_window_days=args.train_window_days,
        top_n=args.top,
        alpha=args.alpha,
        gate_quantile=args.gate_quantile,
    )

    # Fill observed hits into output
    hits_count = []
    any_hit = []
    hits_str = []
    for picks, actual in zip(picks_by_day, actual_by_day):
        if not picks:
            hits_count.append(0)
            any_hit.append(0)
            hits_str.append("")
            continue
        hitset = set(actual)
        hits = [x for x in picks if x in hitset]
        hits_count.append(len(hits))
        any_hit.append(int(len(hits) > 0))
        hits_str.append(",".join(hits))

    out["hits_count"] = hits_count
    out["any_hit"] = any_hit
    out["hits"] = hits_str

    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    days_total = len(out)
    days_traded = int(out["traded"].sum())
    any_hit_overall = float(out["any_hit"].mean()) if days_total else 0.0

    traded_days, traded_any, traded_any_rate, traded_avg_hits = eval_hits(picks_by_day, actual_by_day)

    print("\nSummary:")
    print(f"  days_total: {days_total}")
    print(f"  days_traded: {days_traded} ({(days_traded/days_total if days_total else 0.0):.3f})")
    print(f"  any_hit_rate_overall: {any_hit_overall:.6f}")
    print(f"  any_hit_rate_when_traded: {traded_any_rate:.6f}")
    print(f"  avg_hits_per_day_when_traded: {traded_avg_hits:.6f}")

    # Random baseline p-values
    if args.random_trials and args.random_trials > 0:
        rng = np.random.default_rng(args.seed)
        ge_any = 0
        ge_avg = 0

        # Under null: for each day, TOP3 are 3 iid uniform numbers (duplicates allowed; close enough)
        # Evaluate using same picks_by_day (so model picks are held fixed) -> fair test of "do picks align with reality?"
        rand_any_rates = []
        rand_avg_hits = []

        for _ in range(args.random_trials):
            rand_actual = []
            for _dt in dates:
                xs = rng.integers(0, 10000, size=3)
                rand_actual.append([f"{x:04d}" for x in xs])

            _tdays, _tany, _any_rate, _avg_hits = eval_hits(picks_by_day, rand_actual)
            rand_any_rates.append(_any_rate)
            rand_avg_hits.append(_avg_hits)
            if _any_rate >= traded_any_rate:
                ge_any += 1
            if _avg_hits >= traded_avg_hits:
                ge_avg += 1

        p_any = (ge_any + 1) / (args.random_trials + 1)
        p_avg = (ge_avg + 1) / (args.random_trials + 1)

        r_any = np.array(rand_any_rates, dtype=np.float64)
        r_avg = np.array(rand_avg_hits, dtype=np.float64)

        print("\nP-values vs random (>= observed) WHEN TRADED:")
        print(f"  p_any_hit_rate_when_traded: {p_any:.6f}")
        print(f"  p_avg_hits_when_traded:     {p_avg:.6f}")
        print(f"  trials: {args.random_trials}")

        print("\nRandom baseline WHEN TRADED:")
        print(f"  rand_any_hit_rate_mean: {float(r_any.mean()):.6f}")
        print(f"  rand_any_hit_rate_p10:  {float(np.quantile(r_any, 0.10)):.6f}")
        print(f"  rand_any_hit_rate_p50:  {float(np.quantile(r_any, 0.50)):.6f}")
        print(f"  rand_any_hit_rate_p90:  {float(np.quantile(r_any, 0.90)):.6f}")
        print(f"  rand_avg_hits_mean:     {float(r_avg.mean()):.6f}")
        print(f"  rand_avg_hits_p10:      {float(np.quantile(r_avg, 0.10)):.6f}")
        print(f"  rand_avg_hits_p50:      {float(np.quantile(r_avg, 0.50)):.6f}")
        print(f"  rand_avg_hits_p90:      {float(np.quantile(r_avg, 0.90)):.6f}")

    print("\nLast 10 rows:")
    show = out.tail(10)[["date","traded","gate_score","gate_thresh","hits_count","any_hit","hits","picks"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
