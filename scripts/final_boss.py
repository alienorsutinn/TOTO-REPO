#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

def fmt4(x) -> str:
    try:
        return f"{int(x):04d}"
    except Exception:
        s = str(x).strip()
        if s.isdigit():
            return f"{int(s):04d}"
        return s.zfill(4)

def load_numbers_long(path: str, dedupe: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["n4"] = df["num"].apply(fmt4)
    df = df[df["n4"].str.len() == 4].copy()

    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["date", "bucket", "n4"])
        dropped = before - len(df)
        if dropped:
            print(f"[dedupe] dropped {dropped} duplicate rows")
    return df

def digit_marginals(train_df: pd.DataFrame):
    arr = np.array(train_df["n4"].tolist(), dtype=object)
    d0 = np.array([int(s[0]) for s in arr])
    d1 = np.array([int(s[1]) for s in arr])
    d2 = np.array([int(s[2]) for s in arr])
    d3 = np.array([int(s[3]) for s in arr])

    def probs(d):
        c = np.bincount(d, minlength=10).astype(float)
        c = c / c.sum()
        return c

    return probs(d0), probs(d1), probs(d2), probs(d3)

def last2_probs(train_df: pd.DataFrame):
    arr = np.array(train_df["n4"].tolist(), dtype=object)
    l2 = np.array([int(s[2:4]) for s in arr])
    c = np.bincount(l2, minlength=100).astype(float)
    c = c / c.sum()
    return c

def score_digits_only(train_df: pd.DataFrame, alpha: float = 1.0):
    """
    Simple Dirichlet-smoothed digit model:
      P(n4) = Π_k (count_posk[d] + alpha) / (N + 10*alpha)
    """
    d0p, d1p, d2p, d3p = digit_marginals(train_df)
    # with alpha smoothing baked in if you want: here we already used empirical probs;
    # we can optionally temper probabilities with alpha
    # (alpha > 1 makes it flatter; alpha < 1 makes it peakier)
    eps = 1e-12
    d0p = np.maximum(d0p, eps) ** (1.0 / alpha)
    d1p = np.maximum(d1p, eps) ** (1.0 / alpha)
    d2p = np.maximum(d2p, eps) ** (1.0 / alpha)
    d3p = np.maximum(d3p, eps) ** (1.0 / alpha)
    # renormalize
    d0p /= d0p.sum(); d1p /= d1p.sum(); d2p /= d2p.sum(); d3p /= d3p.sum()

    # return a function that scores any 4-digit string
    def p(n4: str) -> float:
        return float(d0p[int(n4[0])] * d1p[int(n4[1])] * d2p[int(n4[2])] * d3p[int(n4[3])])
    return p

def universe_0000_9999():
    return np.array([f"{i:04d}" for i in range(10000)], dtype=object)

def pick_topn_by_score(score_fn, top_n: int, diversity_last2: int = None, diversity_pos0: int = None):
    U = universe_0000_9999()
    scores = np.array([score_fn(x) for x in U], dtype=float)

    # deterministic tie-break: sort by (-score, number)
    order = np.lexsort((U, -scores))

    picks = []
    cnt_last2 = {}
    cnt_pos0 = {}

    for idx in order:
        s = U[idx]
        if diversity_last2 is not None:
            l2 = s[2:4]
            if cnt_last2.get(l2, 0) >= diversity_last2:
                continue
        if diversity_pos0 is not None:
            p0 = s[0]
            if cnt_pos0.get(p0, 0) >= diversity_pos0:
                continue

        picks.append(s)
        if diversity_last2 is not None:
            cnt_last2[l2] = cnt_last2.get(l2, 0) + 1
        if diversity_pos0 is not None:
            cnt_pos0[p0] = cnt_pos0.get(p0, 0) + 1

        if len(picks) >= top_n:
            break
    return picks

def sample_unique_uniform(rng, top_n: int):
    picks = set()
    while len(picks) < top_n:
        picks.add(f"{int(rng.integers(0, 10000)):04d}")
    return list(picks)

def sample_unique_digit_marginal(rng, d0p, d1p, d2p, d3p, top_n: int):
    picks = set()
    while len(picks) < top_n:
        a = rng.choice(10, p=d0p)
        b = rng.choice(10, p=d1p)
        c = rng.choice(10, p=d2p)
        d = rng.choice(10, p=d3p)
        picks.add(f"{a}{b}{c}{d}")
    return list(picks)

def sample_unique_last2_marginal(rng, last2p, top_n: int):
    """
    Keep last2 distribution realistic; sample first2 uniform.
    """
    picks = set()
    while len(picks) < top_n:
        first2 = int(rng.integers(0, 100))
        last2 = rng.choice(100, p=last2p)
        picks.add(f"{first2:02d}{last2:02d}")
    return list(picks)

@dataclass
class Metrics:
    any_hit_rate: float
    avg_hits_per_day: float
    p_hits_ge2: float

def eval_strategy(dates, picks_by_date, targets_by_date):
    hits_count = []
    for dt in dates:
        picks = set(picks_by_date[dt])
        targets = set(targets_by_date[dt])
        h = len(picks.intersection(targets))
        hits_count.append(h)
    hits_count = np.array(hits_count, dtype=int)
    any_hit = (hits_count >= 1).mean()
    avg_hits = hits_count.mean()
    p_ge2 = (hits_count >= 2).mean()
    return Metrics(any_hit, avg_hits, p_ge2), hits_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-window-days", type=int, default=365)
    ap.add_argument("--train-bucket", default="top3")
    ap.add_argument("--target-buckets", default="top3")  # comma-separated or "all"
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--random-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--diversity-last2", type=int, default=3)  # set 0 to disable
    ap.add_argument("--diversity-pos0", type=int, default=5)   # set 0 to disable
    args = ap.parse_args()

    df = load_numbers_long(args.numbers_long, dedupe=args.dedupe)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    # target buckets
    if args.target_buckets.strip().lower() == "all":
        target_buckets = None
    else:
        target_buckets = [b.strip().lower() for b in args.target_buckets.split(",") if b.strip()]

    # build targets per date
    df_date = df[(pd.to_datetime(df["date"]).dt.date >= start) & (pd.to_datetime(df["date"]).dt.date <= end)].copy()
    all_dates = sorted(df_date["date"].unique().tolist())

    targets_by_date = {}
    for dt in all_dates:
        g = df_date[df_date["date"] == dt]
        if target_buckets is not None:
            g = g[g["bucket"].isin(target_buckets)]
        targets_by_date[dt] = g["n4"].tolist()

    # test dates are those with enough history
    test_dates = []
    for dt in all_dates:
        d = pd.to_datetime(dt).date()
        train_start = d - pd.Timedelta(days=args.train_window_days)
        if train_start < start:
            continue
        test_dates.append(dt)

    if not test_dates:
        raise SystemExit("No test dates after applying train window.")

    # build model picks per date (digits-only)
    picks_model = {}
    for dt in test_dates:
        d = pd.to_datetime(dt).date()
        train_start = d - pd.Timedelta(days=args.train_window_days)
        train = df[(pd.to_datetime(df["date"]).dt.date >= train_start) &
                   (pd.to_datetime(df["date"]).dt.date < d) &
                   (df["bucket"] == args.train_bucket.lower())].copy()
        if len(train) == 0:
            continue
        score_fn = score_digits_only(train, alpha=args.alpha)

        div_l2 = None if args.diversity_last2 <= 0 else args.diversity_last2
        div_p0 = None if args.diversity_pos0 <= 0 else args.diversity_pos0

        picks_model[dt] = pick_topn_by_score(score_fn, args.top, diversity_last2=div_l2, diversity_pos0=div_p0)

    # align dates where we have picks
    dates = [dt for dt in test_dates if dt in picks_model]

    # evaluate model
    m_model, hits_model = eval_strategy(dates, picks_model, targets_by_date)
    print("\nModel (digits-only) metrics:")
    print(m_model)

    # random baselines (matched)
    rng = np.random.default_rng(args.seed)

    # Precompute marginal distributions from the *same training bucket window* per day (fair)
    # For speed: approximate by using the most recent global window across the test span.
    # If you want exact per-day matching, we can do it; but this is already much fairer than uniform.
    last_day = pd.to_datetime(dates[-1]).date()
    train_start = last_day - pd.Timedelta(days=args.train_window_days)
    train_global = df[(pd.to_datetime(df["date"]).dt.date >= train_start) &
                      (pd.to_datetime(df["date"]).dt.date < last_day) &
                      (df["bucket"] == args.train_bucket.lower())].copy()

    d0p, d1p, d2p, d3p = digit_marginals(train_global)
    l2p = last2_probs(train_global)

    def simulate_baseline(which: str):
        any_hits = []
        avg_hits = []
        p_ge2s = []
        for _ in range(args.random_trials):
            picks = {}
            for dt in dates:
                if which == "uniform":
                    picks[dt] = sample_unique_uniform(rng, args.top)
                elif which == "digit_marginal":
                    picks[dt] = sample_unique_digit_marginal(rng, d0p, d1p, d2p, d3p, args.top)
                elif which == "last2_marginal":
                    picks[dt] = sample_unique_last2_marginal(rng, l2p, args.top)
                else:
                    raise ValueError(which)

            m, _ = eval_strategy(dates, picks, targets_by_date)
            any_hits.append(m.any_hit_rate)
            avg_hits.append(m.avg_hits_per_day)
            p_ge2s.append(m.p_hits_ge2)

        any_hits = np.array(any_hits)
        avg_hits = np.array(avg_hits)
        p_ge2s = np.array(p_ge2s)

        return {
            "any_mean": float(any_hits.mean()),
            "any_p": float((np.sum(any_hits >= m_model.any_hit_rate) + 1) / (len(any_hits) + 1)),
            "avg_mean": float(avg_hits.mean()),
            "avg_p": float((np.sum(avg_hits >= m_model.avg_hits_per_day) + 1) / (len(avg_hits) + 1)),
            "ge2_mean": float(p_ge2s.mean()),
            "ge2_p": float((np.sum(p_ge2s >= m_model.p_hits_ge2) + 1) / (len(p_ge2s) + 1)),
        }

    for which in ["uniform", "digit_marginal", "last2_marginal"]:
        res = simulate_baseline(which)
        print(f"\nBaseline: {which}")
        print(res)

    # write detailed output for inspection
    out = pd.DataFrame({
        "date": dates,
        "hits_count": hits_model,
        "any_hit": (hits_model >= 1).astype(int),
        "hits": [",".join(targets_by_date[d]) for d in dates],
        "picks": [",".join(picks_model[d]) for d in dates],
    })
    out_path = "data/processed/final_boss_backtest.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    main()
