#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def fmt4(x: int) -> str:
    return f"{int(x):04d}"

def load_long(path: str, bucket: str, dedupe: bool):
    df = pd.read_csv(path)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    if bucket != "all":
        df = df[df["bucket"] == bucket.lower()]
    if dedupe:
        before = len(df)
        df["n4"] = df["num"].astype(int).map(lambda x: f"{x:04d}")
        df = df.drop_duplicates(subset=["date","bucket","n4"])
        print(f"[dedupe] dropped {before-len(df)} duplicate rows")
    df = df.sort_values("date")
    df["n4"] = df["num"].apply(fmt4)
    df = df[df["n4"].str.len() == 4]
    return df

def build_empirical_marginals(train_numbers):
    # train_numbers: array of 4-char strings
    pos_counts = [np.zeros(10, dtype=float) for _ in range(4)]
    last2_counts = np.zeros(100, dtype=float)

    for s in train_numbers:
        for i,ch in enumerate(s):
            pos_counts[i][ord(ch)-48] += 1
        last2_counts[int(s[-2:])] += 1

    pos_probs = [c / c.sum() if c.sum() else np.ones(10)/10 for c in pos_counts]
    last2_probs = last2_counts / last2_counts.sum() if last2_counts.sum() else np.ones(100)/100
    return pos_probs, last2_probs

def sample_digits_matched(rng, pos_probs, n):
    # independent by position, matches per-position digit distribution
    out = []
    for _ in range(n):
        digs = [str(rng.choice(10, p=pos_probs[i])) for i in range(4)]
        out.append("".join(digs))
    return np.array(out, dtype=object)

def sample_last2_matched(rng, pos_probs_01, last2_probs, n):
    # sample first2 digits from empirical (pos0,pos1), last2 from empirical last2.
    out = []
    for _ in range(n):
        d0 = str(rng.choice(10, p=pos_probs_01[0]))
        d1 = str(rng.choice(10, p=pos_probs_01[1]))
        l2 = int(rng.choice(100, p=last2_probs))
        out.append(d0 + d1 + f"{l2:02d}")
    return np.array(out, dtype=object)

def score_candidates(train_numbers, universe, mode):
    """
    Simple score used for ranking picks:
    - mode='pos0' : empirical P(pos0 digit)
    - mode='multifeature' : digits independent (pos0..pos3)
    """
    pos_probs, last2_probs = build_empirical_marginals(train_numbers)
    scores = {}
    for s in universe:
        if mode == "pos0":
            scores[s] = pos_probs[0][int(s[0])]
        elif mode == "digits":
            scores[s] = (
                pos_probs[0][int(s[0])] *
                pos_probs[1][int(s[1])] *
                pos_probs[2][int(s[2])] *
                pos_probs[3][int(s[3])]
            )
        elif mode == "digits_plus_last2":
            scores[s] = (
                pos_probs[0][int(s[0])] *
                pos_probs[1][int(s[1])] *
                pos_probs[2][int(s[2])] *
                pos_probs[3][int(s[3])] *
                last2_probs[int(s[-2:])]
            )
        else:
            raise ValueError("unknown mode")
    return scores

def deterministic_topk(scores, k):
    # stable tie-break: score desc, then numeric asc
    items = sorted(scores.items(), key=lambda kv: (-kv[1], int(kv[0])))
    return [s for s,_ in items[:k]]

def hits_for_day(picks, actual_numbers):
    aset = set(actual_numbers)
    hits = [p for p in picks if p in aset]
    return hits

def build_universe():
    return np.array([f"{i:04d}" for i in range(10000)], dtype=object)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="top3")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-24")
    ap.add_argument("--train-window-days", type=int, default=365)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--mode", choices=["pos0","digits","digits_plus_last2"], default="digits")
    ap.add_argument("--null", choices=["uniform","digits_matched","last2_matched"], default="uniform")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--out", default="data/processed/backtest_matched_nulls.csv")
    args = ap.parse_args()

    df = load_long(args.numbers_long, args.bucket, args.dedupe)
    df = df[(df["date"] >= args.start) & (df["date"] <= args.end)]
    df["date"] = pd.to_datetime(df["date"])

    # group actuals by date
    by_date = df.groupby("date")["n4"].apply(list).sort_index()
    dates = by_date.index.to_list()

    universe = build_universe()
    rng = np.random.default_rng(args.seed)

    rows = []
    any_hit = []
    hits_ct = []

    # Build per-day backtest (rolling train window)
    for i, dt in enumerate(dates):
        train_start = dt - pd.Timedelta(days=args.train_window_days)
        train_dates = [d for d in dates if (d >= train_start and d < dt)]
        if len(train_dates) < 30:
            continue

        train_numbers = []
        for d in train_dates:
            train_numbers.extend(by_date.loc[d])
        train_numbers = np.array(train_numbers, dtype=object)

        scores = score_candidates(train_numbers, universe, args.mode)
        picks = deterministic_topk(scores, args.top)

        actual = by_date.loc[dt]
        h = hits_for_day(picks, actual)
        any_hit.append(1 if len(h) else 0)
        hits_ct.append(len(h))

        rows.append({
            "date": str(dt.date()),
            "train_start": str(train_start.date()),
            "train_window_days": args.train_window_days,
            "bucket": args.bucket,
            "mode": args.mode,
            "top": args.top,
            "hits_count": len(h),
            "any_hit": 1 if len(h) else 0,
            "hits": ",".join(h),
            "picks": ",".join(picks)
        })

    bt = pd.DataFrame(rows)
    bt.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    if len(bt) == 0:
        print("No backtest rows (check dates / window).")
        return

    obs_any = float(bt["any_hit"].mean())
    obs_avg_hits = float(bt["hits_count"].mean())
    days = int(len(bt))

    # Monte Carlo under selected null
    ge_any = 0
    ge_avg = 0

    # Precompute for each day: actual count and training marginals (for matched nulls)
    day_train_cache = []
    for _, r in bt.iterrows():
        dt = pd.to_datetime(r["date"])
        train_start = pd.to_datetime(r["train_start"])
        train_dates = [d for d in dates if (d >= train_start and d < dt)]
        train_numbers = []
        for d in train_dates:
            train_numbers.extend(by_date.loc[d])
        train_numbers = np.array(train_numbers, dtype=object)

        pos_probs, last2_probs = build_empirical_marginals(train_numbers)
        day_train_cache.append((pos_probs, last2_probs, len(by_date.loc[dt])))

    for _ in range(args.trials):
        trial_any = 0
        trial_hits = 0
        for (pos_probs, last2_probs, m) in day_train_cache:
            # sample m "winning numbers" under null
            if args.null == "uniform":
                u = rng.integers(0, 10000, size=m)
                s = np.array([f"{x:04d}" for x in u], dtype=object)
            elif args.null == "digits_matched":
                s = sample_digits_matched(rng, pos_probs, m)
            elif args.null == "last2_matched":
                s = sample_last2_matched(rng, pos_probs[:2], last2_probs, m)
            else:
                raise ValueError("bad null")

            # evaluate using the *same picks count* but under null outcomes:
            # approximate any_hit by sampling overlap probability via set intersection:
            # to do it exactly, we'd need picks for each day; we can reuse from bt.
            # (exact)
            # NOTE: reading picks each time is slower but ok for 2k-10k trials with 600-700 days.
            # We'll pre-split picks sets:
        # We'll compute in second pass to avoid overhead

    # exact MC with cached picks
    pick_sets = [set(p.split(",")) for p in bt["picks"].tolist()]
    m_list = [m for (_,_,m) in day_train_cache]
    pos_list = [pos for (pos,_,_) in day_train_cache]
    last2_list = [l2 for (_,l2,_) in day_train_cache]

    ge_any = 0
    ge_avg = 0
    for _ in range(args.trials):
        trial_any_hits = 0
        trial_hits_sum = 0
        for day_idx in range(days):
            m = m_list[day_idx]
            if args.null == "uniform":
                u = rng.integers(0, 10000, size=m)
                s = set(f"{x:04d}" for x in u)
            elif args.null == "digits_matched":
                s = set(sample_digits_matched(rng, pos_list[day_idx], m).tolist())
            elif args.null == "last2_matched":
                s = set(sample_last2_matched(rng, pos_list[day_idx][:2], last2_list[day_idx], m).tolist())
            else:
                raise ValueError("bad null")

            hits = len(pick_sets[day_idx] & s)
            if hits > 0:
                trial_any_hits += 1
            trial_hits_sum += hits

        trial_any = trial_any_hits / days
        trial_avg = trial_hits_sum / days
        if trial_any >= obs_any: ge_any += 1
        if trial_avg >= obs_avg_hits: ge_avg += 1

    p_any = (ge_any + 1) / (args.trials + 1)
    p_avg = (ge_avg + 1) / (args.trials + 1)

    print("\nObserved:")
    print("  days_tested:", days)
    print("  any_hit_rate:", obs_any)
    print("  avg_hits_per_day:", obs_avg_hits)
    print("\nP-values vs matched null (>= observed):")
    print("  null:", args.null)
    print("  p_any_hit_rate:", p_any)
    print("  p_avg_hits_per_day:", p_avg)
    print("  trials:", args.trials)

if __name__ == "__main__":
    main()
