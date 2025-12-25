from __future__ import annotations
import argparse
import math
import pandas as pd
import numpy as np

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

    # hard dedupe (same date+bucket+number should be unique)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "bucket", "n4"], keep="first").copy()
    after = len(df)
    if after != before:
        print(f"[dedupe] dropped {before-after} duplicate rows")

    return df

def summarize_pos0(nums: pd.Series, title: str) -> None:
    d0 = nums.str[0]
    counts = d0.value_counts().reindex(list("0123456789"), fill_value=0)
    n = int(counts.sum())
    freq = (counts / n).round(6)

    # expected uniform
    exp = n / 10.0
    z = ((counts - exp) / math.sqrt(exp)).round(3)  # approx z-score per digit

    # print sorted by deviation
    dev = (freq - 0.1).round(6)
    out = pd.DataFrame({"count": counts, "freq": freq, "dev_from_0.1": dev, "z_approx": z})
    out_sorted = out.sort_values("dev_from_0.1", ascending=False)

    print(f"\n=== {title} ===")
    print(f"n={n}")
    print(out_sorted.to_string())

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--window-end", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--window-days", type=int, default=180)
    args = ap.parse_args()

    df = read_long(args.numbers_long)
    top3 = df[df["bucket"] == "top3"].copy()
    if top3.empty:
        raise ValueError("No top3 rows found.")

    summarize_pos0(top3["n4"].astype(str), "TOP3 pos0 overall (all dates)")

    if args.window_end:
        end = pd.to_datetime(args.window_end).date()
        start = (pd.to_datetime(end) - pd.Timedelta(days=args.window_days - 1)).date()
        w = top3[(top3["date"] >= start) & (top3["date"] <= end)].copy()
        summarize_pos0(w["n4"].astype(str), f"TOP3 pos0 window {start}..{end} (days={args.window_days})")

        # Also show per-year in that window (if it spans years)
        w["year"] = pd.to_datetime(w["date"]).dt.year
        for yr, g in w.groupby("year"):
            summarize_pos0(g["n4"].astype(str), f"TOP3 pos0 within window, year={yr}")

if __name__ == "__main__":
    main()
