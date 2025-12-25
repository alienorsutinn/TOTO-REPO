#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

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


def build_last2_probs(train_n4: np.ndarray, alpha: float) -> np.ndarray:
    # last2 from 00..99
    counts = np.zeros(100, dtype=np.float64)
    for s in train_n4:
        counts[int(s[-2:])] += 1.0
    n = counts.sum()
    probs = (counts + alpha) / (n + 100.0 * alpha)
    return probs


def sample_matched_top3(
    rng: np.random.Generator,
    digit_probs: np.ndarray,
    last2_probs: Optional[np.ndarray],
    use_last2: bool,
) -> List[str]:
    """
    Generates 3 outcomes (strings) matching:
      - per-position digit marginals (always)
      - optionally last2 marginals (approximately, via rejection-lite)
    """
    out = []
    tries = 0
    while len(out) < 3:
        tries += 1
        if tries > 20000:
            # fail-safe: return whatever we have + fill with digit-only
            while len(out) < 3:
                d0 = rng.choice(10, p=digit_probs[0])
                d1 = rng.choice(10, p=digit_probs[1])
                d2 = rng.choice(10, p=digit_probs[2])
                d3 = rng.choice(10, p=digit_probs[3])
                out.append(f"{d0}{d1}{d2}{d3}")
            break

        # sample digits
        d0 = rng.choice(10, p=digit_probs[0])
        d1 = rng.choice(10, p=digit_probs[1])
        d2 = rng.choice(10, p=digit_probs[2])
        d3 = rng.choice(10, p=digit_probs[3])
        s = f"{d0}{d1}{d2}{d3}"

        if not use_last2 or last2_probs is None:
            out.append(s)
            continue

        # accept/reject based on last2 probability ratio vs uniform-ish proposal.
        # Proposal already has induced last2 distribution from digit_probs; we "tilt" toward observed last2.
        last2 = int(s[-2:])
        p_tgt = last2_probs[last2]
        # scale factor to keep acceptance sane
        # (max p_tgt is at most ~0.05-ish typically, but we normalize against median)
        # Use a conservative cap:
        a = min(1.0, p_tgt * 120.0)  # acceptance roughly proportional to p_tgt
        if rng.random() < a:
            out.append(s)

    return out


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


def eval_hits(picks_by_day: List[List[str]], actual_by_day: List[List[str]]) -> Tuple[int, float, float]:
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
    return traded_days, any_rate, avg_hits


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
    ap.add_argument("--matched-trials", type=int, default=20000)
    ap.add_argument("--use-last2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/regime_gate_backtest_matched_null.csv")
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

    # Build picks + gate using real history (same as your gating approach)
    past_gate_scores: List[float] = []
    picks_by_day: List[List[str]] = []
    rows: List[DayRow] = []

    # store per-day training digit_probs (+ last2_probs) so matched null can condition on the *same* info
    train_digit_probs_by_day: List[Optional[np.ndarray]] = []
    train_last2_probs_by_day: List[Optional[np.ndarray]] = []

    for dt in dates:
        train_start = dt - pd.Timedelta(days=args.train_window_days)
        train_end = dt - pd.Timedelta(days=1)
        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        train_n4 = train_df["n4"].to_numpy(dtype=object)

        if len(train_n4) < 50:
            picks_by_day.append([])
            rows.append(DayRow(str(dt.date()), 0, float("nan"), float("nan"), 0, 0, "", ""))
            train_digit_probs_by_day.append(None)
            train_last2_probs_by_day.append(None)
            continue

        # gate score uses pos0 chi2 vs uniform
        pos0 = np.fromiter((digits_of_str(s)[0] for s in train_n4), dtype=np.int32)
        counts0 = np.bincount(pos0, minlength=10).astype(np.float64)
        gate_score = chi2_vs_uniform_digit(counts0)

        if len(past_gate_scores) >= 30:
            gate_thresh = float(np.quantile(np.array(past_gate_scores, dtype=np.float64), args.gate_quantile))
        else:
            gate_thresh = gate_score

        traded = int(gate_score >= gate_thresh)

        digit_probs = smoothed_digit_probs(train_n4, alpha=args.alpha)
        last2_probs = build_last2_probs(train_n4, alpha=args.alpha) if args.use_last2 else None

        train_digit_probs_by_day.append(digit_probs)
        train_last2_probs_by_day.append(last2_probs)

        if traded:
            log_scores = score_universe(digit_probs)
            picks = top_n_from_scores(log_scores, top_n=args.top)
            picks_by_day.append(picks)
            rows.append(DayRow(str(dt.date()), 1, gate_score, gate_thresh, 0, 0, "", ",".join(picks)))
        else:
            picks_by_day.append([])
            rows.append(DayRow(str(dt.date()), 0, gate_score, gate_thresh, 0, 0, "", ""))

        past_gate_scores.append(gate_score)

    # Observed hits
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

    out = pd.DataFrame([r.__dict__ for r in rows])
    out["hits_count"] = hits_count
    out["any_hit"] = any_hit
    out["hits"] = hits_str
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    days_total = len(out)
    days_traded = int(out["traded"].sum())
    traded_days, obs_any_rate, obs_avg_hits = eval_hits(picks_by_day, actual_by_day)

    print("\nObserved:")
    print(f"  days_total: {days_total}")
    print(f"  days_traded: {days_traded} ({(days_traded/days_total if days_total else 0.0):.3f})")
    print(f"  any_hit_rate_when_traded: {obs_any_rate:.6f}")
    print(f"  avg_hits_per_day_when_traded: {obs_avg_hits:.6f}")

    # Matched-null trials:
    # For each day, generate fake TOP3 outcomes from that day's TRAINING digit_probs (and optional last2),
    # then evaluate hits using the same picks_by_day.
    rng = np.random.default_rng(args.seed)
    ge_any = 0
    ge_avg = 0
    null_any = []
    null_avg = []

    for _ in range(args.matched_trials):
        fake_actual_by_day = []
        for dp, l2, picks in zip(train_digit_probs_by_day, train_last2_probs_by_day, picks_by_day):
            if dp is None:
                fake_actual_by_day.append([])
                continue
            # only generate "outcomes" for days that exist; even if not traded, it's harmless
            xs = sample_matched_top3(rng, dp, l2, use_last2=args.use_last2)
            fake_actual_by_day.append(xs)

        _tdays, any_rate, avg_hits = eval_hits(picks_by_day, fake_actual_by_day)
        null_any.append(any_rate)
        null_avg.append(avg_hits)
        if any_rate >= obs_any_rate:
            ge_any += 1
        if avg_hits >= obs_avg_hits:
            ge_avg += 1

    p_any = (ge_any + 1) / (args.matched_trials + 1)
    p_avg = (ge_avg + 1) / (args.matched_trials + 1)

    null_any = np.array(null_any, dtype=np.float64)
    null_avg = np.array(null_avg, dtype=np.float64)

    print("\nMatched-null p-values (>= observed) WHEN TRADED:")
    print(f"  p_any_hit_rate_when_traded: {p_any:.6f}")
    print(f"  p_avg_hits_when_traded:     {p_avg:.6f}")
    print(f"  trials: {args.matched_trials}")
    print("\nMatched-null distribution WHEN TRADED:")
    print(f"  null_any_mean: {float(null_any.mean()):.6f}")
    print(f"  null_any_p10:  {float(np.quantile(null_any, 0.10)):.6f}")
    print(f"  null_any_p50:  {float(np.quantile(null_any, 0.50)):.6f}")
    print(f"  null_any_p90:  {float(np.quantile(null_any, 0.90)):.6f}")
    print(f"  null_avg_mean: {float(null_avg.mean()):.6f}")
    print(f"  null_avg_p10:  {float(np.quantile(null_avg, 0.10)):.6f}")
    print(f"  null_avg_p50:  {float(np.quantile(null_avg, 0.50)):.6f}")
    print(f"  null_avg_p90:  {float(np.quantile(null_avg, 0.90)):.6f}")

    print("\nLast 10 rows:")
    show = out.tail(10)[["date","traded","gate_score","gate_thresh","hits_count","any_hit","hits","picks"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
