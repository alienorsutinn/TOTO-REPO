from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- utilities ----------

def _zfill4(x: str) -> str:
    s = str(x).strip()
    if s == "":
        return "0000"
    return s.zfill(4)


def _read_numbers_long(path: str, bucket: str = "all") -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date'. Have: {list(df.columns)}")

    num_col = None
    for c in ["number", "num", "n4", "value"]:
        if c in df.columns:
            num_col = c
            break
    if num_col is None:
        raise ValueError(f"Missing number column (number|num|n4|value). Have: {list(df.columns)}")

    if "bucket" not in df.columns:
        df["bucket"] = "all"
    if "operator" not in df.columns:
        df["operator"] = "unknown"

    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["operator"] = df["operator"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["number_str"] = df[num_col].astype(str).map(_zfill4)

    if bucket != "all":
        df = df[df["bucket"] == bucket].copy()

    return df


def chi2_uniform_digit(counts: np.ndarray) -> Tuple[float, int, float]:
    """
    counts: length-10 array of digit counts.
    Returns: (chi2, df, approx_p) using Wilson-Hilferty normal approx (no scipy dependency).
    """
    n = counts.sum()
    if n <= 0:
        return (float("nan"), 9, float("nan"))

    expected = np.ones(10) * (n / 10.0)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    df = 9

    # Approximate chi-square p-value via Wilson-Hilferty transform to normal
    # z = [(chi2/df)^(1/3) - (1 - 2/(9df))] / sqrt(2/(9df))
    a = (chi2 / df) ** (1.0 / 3.0)
    mu = 1.0 - 2.0 / (9.0 * df)
    sigma = math.sqrt(2.0 / (9.0 * df))
    z = (a - mu) / sigma

    # p = 1 - Phi(z)
    p = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return chi2, df, p


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """
    Returns q-values (FDR-adjusted p-values).
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * m) / rank
        prev = min(prev, val)
        q[i] = prev
    # unpermute
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def digit_counts(series: pd.Series, pos: int) -> np.ndarray:
    digs = series.str[pos]
    counts = np.zeros(10, dtype=int)
    for d, c in digs.value_counts().items():
        if d.isdigit():
            counts[int(d)] = int(c)
    return counts


def permutation_pvalue_top3_pos0(df: pd.DataFrame, trials: int, seed: int) -> float:
    """
    Permute digits within each number (breaks position structure but preserves digit multiset),
    recompute chi2 for pos0, return p-value >= observed.
    """
    rng = np.random.default_rng(seed)
    s = df["number_str"].astype(str)
    obs_counts = digit_counts(s, pos=0)
    obs_chi2, _, _ = chi2_uniform_digit(obs_counts)

    chi2s = []
    arr = np.array([list(x) for x in s], dtype="<U1")  # shape (n,4)
    for _ in range(trials):
        perm = arr.copy()
        # permute columns per row
        for i in range(perm.shape[0]):
            rng.shuffle(perm[i])
        perm_s = pd.Series(["".join(row) for row in perm])
        c = digit_counts(perm_s, pos=0)
        chi2, _, _ = chi2_uniform_digit(c)
        chi2s.append(chi2)

    chi2s = np.array(chi2s)
    p = float((chi2s >= obs_chi2).mean())
    return p


def rolling_chi2(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.sort_values("date")
    # group by date: list of numbers
    g = dfx.groupby("date")["number_str"].apply(list).reset_index()

    out_rows = []
    for i in range(len(g)):
        end = g.loc[i, "date"]
        start = end - pd.Timedelta(days=window_days - 1)
        mask = (g["date"] >= start) & (g["date"] <= end)
        nums = sum(g.loc[mask, "number_str"].tolist(), [])
        if len(nums) < 200:  # avoid tiny windows
            continue
        s = pd.Series(nums)
        row = {"end_date": end.date(), "n": len(nums)}
        for pos in range(4):
            c = digit_counts(s, pos)
            chi2, _, p = chi2_uniform_digit(c)
            row[f"pos{pos}_chi2"] = chi2
            row[f"pos{pos}_p"] = p
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ---------- main report ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="all")
    ap.add_argument("--perm-trials", type=int, default=5000)
    ap.add_argument("--perm-seed", type=int, default=1337)
    ap.add_argument("--rolling-days", type=int, default=180)
    ap.add_argument("--out-prefix", default="data/processed/eda_ultra")
    args = ap.parse_args()

    df = _read_numbers_long(args.numbers_long, bucket=args.bucket)
    print(f"Rows: {len(df)} (bucket={args.bucket})")
    print(f"Unique numbers: {df['number_str'].nunique()} / {len(df)}")
    print(f"Unique dates:   {df['date'].nunique()} (min={df['date'].min()}, max={df['date'].max()})")

    # duplicates
    dup = df.duplicated(subset=["date", "bucket", "number_str"], keep=False)
    n_dup = int(dup.sum())
    print(f"Duplicate (date,bucket,number) rows: {n_dup}")
    if n_dup:
        print("First few dup rows:")
        print(df.loc[dup, ["date", "operator", "bucket", "number_str"]].head(10).to_string(index=False))

    # operator × position chi2
    rows = []
    for (op, buck), g in df.groupby(["operator", "bucket"]):
        s = g["number_str"].astype(str)
        for pos in range(4):
            c = digit_counts(s, pos)
            chi2, dff, p = chi2_uniform_digit(c)
            rows.append(
                {
                    "operator": op,
                    "bucket": buck,
                    "pos": pos,
                    "n": int(len(g)),
                    "chi2": chi2,
                    "df": dff,
                    "p": p,
                }
            )

    res = pd.DataFrame(rows).sort_values("p")
    # FDR across ALL these tests
    res["q_fdr"] = benjamini_hochberg(res["p"].tolist())
    out_csv = f"{args.out_prefix}_{args.bucket}_chi2_by_operator.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
    print("\nTop 20 smallest p-values (with FDR q):")
    print(res.head(20).to_string(index=False))

    # permutation test for top3 pos0 anomaly (only if bucket==top3 or bucket==all with bucket column)
    if args.bucket == "top3":
        p_perm = permutation_pvalue_top3_pos0(df, trials=args.perm_trials, seed=args.perm_seed)
        print(f"\nPermutation test (top3 pos0) p_perm: {p_perm:.4f}  (trials={args.perm_trials})")

    # rolling stability (only meaningful when bucket != all? still okay)
    roll = rolling_chi2(df, window_days=args.rolling_days)
    out_roll = f"{args.out_prefix}_{args.bucket}_rolling_{args.rolling_days}d.csv"
    roll.to_csv(out_roll, index=False)
    print(f"\nWrote: {out_roll}")
    if len(roll):
        # show worst (smallest p) window for pos0
        worst = roll.sort_values("pos0_p").head(5)[["end_date", "n", "pos0_chi2", "pos0_p"]]
        print("\nWorst 5 rolling windows for pos0 (smallest p):")
        print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
