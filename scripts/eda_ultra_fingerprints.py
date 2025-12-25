#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def fmt4(x: int) -> str:
    return f"{int(x):04d}"


def digits(s: str):
    return [ord(c) - 48 for c in s]


def rep_score(ds):
    # 0..3 : number of equalities among digits (rough)
    return int(ds[0]==ds[1]) + int(ds[1]==ds[2]) + int(ds[2]==ds[3]) + int(ds[0]==ds[2]) + int(ds[1]==ds[3]) + int(ds[0]==ds[3])


def is_run(ds):
    inc = all(ds[i+1] == ds[i] + 1 for i in range(3))
    dec = all(ds[i+1] == ds[i] - 1 for i in range(3))
    return inc, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="all", choices=["top3","starter","consolation","all"])
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket]

    n4 = df["num"].astype(int).map(fmt4).to_numpy(dtype=object)
    n = len(n4)
    if n == 0:
        raise SystemExit("No rows after filtering.")

    # Observed metrics
    sums = np.array([sum(digits(s)) for s in n4], dtype=np.int32)
    rep = np.array([rep_score(digits(s)) for s in n4], dtype=np.int32)
    inc = np.array([is_run(digits(s))[0] for s in n4], dtype=bool)
    dec = np.array([is_run(digits(s))[1] for s in n4], dtype=bool)

    # digit frequency (all digits pooled)
    all_digits = np.concatenate([np.array(digits(s), dtype=np.int32) for s in n4])
    digit_counts = np.bincount(all_digits, minlength=10)

    # "lucky" tests (heuristic)
    obs_share_8 = digit_counts[8] / all_digits.size
    obs_share_4 = digit_counts[4] / all_digits.size

    # plate-ish bands (0xxx..9xxx)
    first = np.array([digits(s)[0] for s in n4], dtype=np.int32)
    band_counts = np.bincount(first, minlength=10)

    obs = {
        "sum_mean": float(sums.mean()),
        "sum_p90": float(np.quantile(sums, 0.90)),
        "rep_mean": float(rep.mean()),
        "rep_ge2_rate": float((rep >= 2).mean()),
        "inc_run_rate": float(inc.mean()),
        "dec_run_rate": float(dec.mean()),
        "share_digit_8": float(obs_share_8),
        "share_digit_4": float(obs_share_4),
        "band_entropy": float(-(band_counts/band_counts.sum() * np.log((band_counts+1e-12)/band_counts.sum())).sum()),
    }

    # Null: iid uniform 0000-9999
    rng = np.random.default_rng(args.seed)
    ge = {k: 0 for k in obs.keys()}

    for _ in range(args.trials):
        u = rng.integers(0, 10000, size=n)
        s = np.array([f"{x:04d}" for x in u], dtype=object)

        sums0 = np.array([sum(digits(x)) for x in s], dtype=np.int32)
        rep0 = np.array([rep_score(digits(x)) for x in s], dtype=np.int32)
        inc0 = np.array([is_run(digits(x))[0] for x in s], dtype=bool)
        dec0 = np.array([is_run(digits(x))[1] for x in s], dtype=bool)

        all0 = np.concatenate([np.array(digits(x), dtype=np.int32) for x in s])
        cnt0 = np.bincount(all0, minlength=10)
        share8 = cnt0[8] / all0.size
        share4 = cnt0[4] / all0.size

        first0 = np.array([digits(x)[0] for x in s], dtype=np.int32)
        band0 = np.bincount(first0, minlength=10)
        ent0 = float(-(band0/band0.sum() * np.log((band0+1e-12)/band0.sum())).sum())

        null = {
            "sum_mean": float(sums0.mean()),
            "sum_p90": float(np.quantile(sums0, 0.90)),
            "rep_mean": float(rep0.mean()),
            "rep_ge2_rate": float((rep0 >= 2).mean()),
            "inc_run_rate": float(inc0.mean()),
            "dec_run_rate": float(dec0.mean()),
            "share_digit_8": float(share8),
            "share_digit_4": float(share4),
            "band_entropy": float(ent0),
        }

        for k in obs.keys():
            if null[k] >= obs[k]:
                ge[k] += 1

    pvals = {k: (ge[k] + 1) / (args.trials + 1) for k in obs.keys()}

    print(f"Bucket={args.bucket} n={n} trials={args.trials}")
    print("\nObserved + p_ge_obs (>=):")
    for k in sorted(obs.keys()):
        print(f"  {k:14s}  obs={obs[k]:.6f}  p={pvals[k]:.6f}")

