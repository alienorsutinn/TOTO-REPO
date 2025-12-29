# scripts/compare_pick_sets.py
from __future__ import annotations

import argparse
import pandas as pd


def to_set(s: str) -> set[str]:
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    args = ap.parse_args()

    a = pd.read_csv(args.csv_a)
    b = pd.read_csv(args.csv_b)

    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])

    a = a[["date", "picks"]].rename(columns={"picks": "picks_a"})
    b = b[["date", "picks"]].rename(columns={"picks": "picks_b"})
    m = a.merge(b, on="date", how="inner").sort_values("date")

    if len(m) == 0:
        raise SystemExit("No overlapping dates between the two CSVs.")

    jac = []
    for _, row in m.iterrows():
        sa, sb = to_set(row["picks_a"]), to_set(row["picks_b"])
        denom = len(sa | sb)
        jac.append((len(sa & sb) / denom) if denom else 0.0)

    m["jaccard"] = jac
    print(f"[overlap] days={len(m)} mean_jaccard={m['jaccard'].mean():.4f} median={m['jaccard'].median():.4f}")
    print("\nLast 10 days:")
    print(m.tail(10)[["date", "jaccard"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
