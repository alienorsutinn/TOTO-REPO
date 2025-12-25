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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="top3")
    ap.add_argument("--out", default="data/processed/eda_within_day.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df = df[df["bucket"] == args.bucket.lower()]
    df["n4"] = df["num"].apply(fmt4)
    df = df[df["n4"].str.len()==4]
    df["n_int"] = df["n4"].astype(int)

    rows = []
    for dt, g in df.groupby("date"):
        xs = np.sort(g["n_int"].to_numpy())
        if len(xs) < 2:
            continue
        diffs = np.diff(xs)
        # Near-collision metrics
        near10 = int(np.sum(diffs <= 10))
        near50 = int(np.sum(diffs <= 50))
        near100 = int(np.sum(diffs <= 100))
        # last2 / last3 collisions within day
        last2 = g["n4"].str[-2:]
        last3 = g["n4"].str[-3:]
        c2 = int(last2.duplicated().sum())
        c3 = int(last3.duplicated().sum())
        rows.append({
            "date": str(dt),
            "n": int(len(xs)),
            "min_gap": int(diffs.min()),
            "p10_gaps": near10,
            "p50_gaps": near50,
            "p100_gaps": near100,
            "dup_last2": c2,
            "dup_last3": c3,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    if len(out):
        print("\nSummary:")
        print(out.describe(include="all").to_string())

        # highlight the most "structured" days
        print("\nTop 15 smallest min_gap:")
        print(out.nsmallest(15, "min_gap")[["date","n","min_gap","p10_gaps","p50_gaps","dup_last2","dup_last3"]].to_string(index=False))

        print("\nTop 15 largest min_gap (too separated?):")
        print(out.nlargest(15, "min_gap")[["date","n","min_gap","p10_gaps","p50_gaps","dup_last2","dup_last3"]].to_string(index=False))

if __name__ == "__main__":
    main()
