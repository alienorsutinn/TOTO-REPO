#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def fmt4(x):
    try:
        return f"{int(x)%10000:04d}"
    except:
        s="".join([c for c in str(x) if c.isdigit()])
        if not s: return ""
        return s[-4:].zfill(4)

def is_mmdd(s):
    mm = int(s[:2]); dd = int(s[2:])
    return (1 <= mm <= 12) and (1 <= dd <= 31)

def is_ddmm(s):
    dd = int(s[:2]); mm = int(s[2:])
    return (1 <= mm <= 12) and (1 <= dd <= 31)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="all")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket.lower()]
    n4 = df["num"].apply(fmt4)
    n4 = n4[n4.str.len()==4].to_numpy()

    obs_mmdd = sum(is_mmdd(s) for s in n4)
    obs_ddmm = sum(is_ddmm(s) for s in n4)
    n = len(n4)

    rng = np.random.default_rng(args.seed)
    ge_mmdd = 0
    ge_ddmm = 0

    # Null: iid uniform 0000-9999
    for _ in range(args.trials):
        u = rng.integers(0, 10000, size=n)
        s = np.array([f"{x:04d}" for x in u], dtype=object)
        mmdd = sum(is_mmdd(x) for x in s)
        ddmm = sum(is_ddmm(x) for x in s)
        if mmdd >= obs_mmdd: ge_mmdd += 1
        if ddmm >= obs_ddmm: ge_ddmm += 1

    p_mmdd = (ge_mmdd + 1) / (args.trials + 1)
    p_ddmm = (ge_ddmm + 1) / (args.trials + 1)

    print(f"n={n} bucket={args.bucket}")
    print(f"obs_mmdd={obs_mmdd} rate={obs_mmdd/n:.4f}  p_perm={p_mmdd:.4f}")
    print(f"obs_ddmm={obs_ddmm} rate={obs_ddmm/n:.4f}  p_perm={p_ddmm:.4f}")

if __name__ == "__main__":
    main()
