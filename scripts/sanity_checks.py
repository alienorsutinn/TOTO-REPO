from __future__ import annotations

import argparse
from collections import Counter
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--numbers-long", default="data/processed/numbers_long.csv")
    p.add_argument("--bucket", default="all", help="all|top3|starter|consolation")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.strip().str.lower()
    df["num"] = df["num"].astype(str).str.strip().str.zfill(4)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["year"] = df["date"].dt.year

    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket].copy()

    nums = df["num"].tolist()

    # position digit frequencies
    pos_counts = [Counter() for _ in range(4)]
    last_counts = Counter()
    for s in nums:
        for i, ch in enumerate(s):
            pos_counts[i][ch] += 1
        last_counts[s[-1]] += 1

    print(f"Rows: {len(df)} (bucket={args.bucket})")
    print("\nDigit frequency by position (0=thousands ... 3=ones):")
    for i in range(4):
        total = sum(pos_counts[i].values())
        top = pos_counts[i].most_common(10)
        print(f"  pos{i} total={total} top10={[(d, round(c/total,4)) for d,c in top]}")

    print("\nLast digit distribution:")
    total = sum(last_counts.values())
    print([(d, round(c/total,4)) for d,c in last_counts.most_common()])

    # repeated-digit rate
    rep = 0
    for s in nums:
        if len(set(s)) < 4:
            rep += 1
    print(f"\nRepeated-digit share (any repeat): {rep/len(nums):.4f}")

    # quick drift check (yearly last digit)
    print("\nYearly last-digit top3:")
    for y, g in df.groupby("year"):
        c = Counter([x[-1] for x in g["num"].tolist()])
        tot = sum(c.values())
        top3 = [(d, round(v/tot,4)) for d, v in c.most_common(3)]
        print(f"  {y}: {top3}")


if __name__ == "__main__":
    main()
