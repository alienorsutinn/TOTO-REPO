#!/usr/bin/env python3
"""
eda_pairs.py — "next level" EDA for 4D draw data in long format.

Input expected: numbers_long.csv with at least:
  date, bucket, num
Optional:
  operator, draw_no

Works with your current schema:
  ['date','draw_no','operator','bucket','n','num']

What it does:
- integrity checks: duplicates, per-date counts, overlaps between buckets
- digit marginals (per position) + chi2 vs uniform
- pair distributions: pos0pos1 and pos2pos3 (00-99) + chi2 + permutation p
- rolling window tests for pair chi2 + KL divergence drift
- cross-bucket coupling: correlation of per-date digit histograms between buckets
- outputs multiple CSVs into data/processed/

No heavy deps. Uses pandas + numpy only.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _as_date(s: pd.Series) -> pd.Series:
    # Robust parse: allow "YYYY-MM-DD" strings
    return pd.to_datetime(s, errors="coerce").dt.date


def _format_num4(x) -> str:
    # Convert input like 7 -> "0007"; also handle "0007" already
    try:
        i = int(x)
        if i < 0:
            return ""
        if i > 9999:
            # If data contains >4 digits, keep last 4 (defensive)
            i = i % 10000
        return f"{i:04d}"
    except Exception:
        s = str(x).strip()
        if not s:
            return ""
        # Keep digits only
        s2 = "".join(ch for ch in s if ch.isdigit())
        if not s2:
            return ""
        if len(s2) > 4:
            s2 = s2[-4:]
        return s2.zfill(4)


def _read_numbers_long(path: str, bucket: str, operator: Optional[str], dedupe: bool) -> pd.DataFrame:
    df = pd.read_csv(path)

    req = {"date", "bucket", "num"}
    if not req.issubset(df.columns):
        raise ValueError(f"numbers_long missing required columns: {sorted(req)}. Have: {list(df.columns)}")

    df["date"] = _as_date(df["date"])
    df = df.dropna(subset=["date"])
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()

    if bucket != "all":
        df = df[df["bucket"] == bucket.lower()]

    if operator is not None and "operator" in df.columns:
        df["operator"] = df["operator"].astype(str).str.lower().str.strip()
        df = df[df["operator"] == operator.lower()]

    df["number_str"] = df["num"].apply(_format_num4)
    df = df[df["number_str"].str.len() == 4]

    # Dedupe within (date,bucket,number_str) by default
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["date", "bucket", "number_str"])
        dropped = before - len(df)
        if dropped:
            print(f"[dedupe] dropped {dropped} duplicate rows")

    # Derived digits / pairs
    df["pos0"] = df["number_str"].str[0]
    df["pos1"] = df["number_str"].str[1]
    df["pos2"] = df["number_str"].str[2]
    df["pos3"] = df["number_str"].str[3]
    df["p01"] = df["pos0"] + df["pos1"]  # 00-99
    df["p23"] = df["pos2"] + df["pos3"]  # 00-99

    return df


def _chisq_stat_from_counts(counts: np.ndarray, expected: float) -> float:
    # counts shape [k], expected scalar
    # avoid div-by-zero
    expected = max(expected, 1e-12)
    return float(np.sum((counts - expected) ** 2 / expected))


def _chisq_p_approx(chi2: float, df: int) -> float:
    """
    Approx p-value without scipy.
    Uses Wilson-Hilferty normal approximation for chi-square.
    Good enough for ranking and coarse significance.
    """
    if chi2 <= 0:
        return 1.0
    if df <= 0:
        return 1.0
    # Wilson-Hilferty transform
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    # p = 1 - Phi(z)
    p = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return float(min(max(p, 0.0), 1.0))


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        q = pvals[idx] * n / rank
        prev = min(prev, q)
        ranked[idx] = prev
    return np.clip(ranked, 0.0, 1.0)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) with smoothing; p,q sum to 1."""
    eps = 1e-12
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.sum(p2 * np.log(p2 / q2)))


def _vector_digit_hist(df: pd.DataFrame, col: str) -> np.ndarray:
    # returns length-10 probability vector for digits 0..9
    vc = df[col].value_counts()
    counts = np.array([vc.get(str(d), 0) for d in range(10)], dtype=float)
    s = counts.sum()
    if s <= 0:
        return np.ones(10) / 10
    return counts / s


def _vector_pair_hist(df: pd.DataFrame, col: str) -> np.ndarray:
    # returns length-100 probability vector for 00..99
    vc = df[col].value_counts()
    counts = np.array([vc.get(f"{i:02d}", 0) for i in range(100)], dtype=float)
    s = counts.sum()
    if s <= 0:
        return np.ones(100) / 100
    return counts / s


# -----------------------------
# EDA blocks
# -----------------------------
@dataclass
class Chi2Result:
    label: str
    n: int
    chi2: float
    df: int
    p_approx: float


def digit_marginals_chi2(df: pd.DataFrame) -> List[Chi2Result]:
    out: List[Chi2Result] = []
    n = len(df)
    for pos in ["pos0", "pos1", "pos2", "pos3"]:
        vc = df[pos].value_counts()
        counts = np.array([vc.get(str(d), 0) for d in range(10)], dtype=float)
        exp = n / 10.0
        chi2 = _chisq_stat_from_counts(counts, exp)
        p = _chisq_p_approx(chi2, df=9)
        out.append(Chi2Result(label=pos, n=n, chi2=chi2, df=9, p_approx=p))
    return out


def pair_chi2(df: pd.DataFrame, col: str) -> Chi2Result:
    n = len(df)
    vc = df[col].value_counts()
    counts = np.array([vc.get(f"{i:02d}", 0) for i in range(100)], dtype=float)
    exp = n / 100.0
    chi2 = _chisq_stat_from_counts(counts, exp)
    p = _chisq_p_approx(chi2, df=99)
    return Chi2Result(label=col, n=n, chi2=chi2, df=99, p_approx=p)


def perm_pvalue_pair_chi2(df: pd.DataFrame, col: str, trials: int, seed: int) -> float:
    """
    Permutation test for whether the pair distribution is more non-uniform than random,
    preserving marginal digits per position by shuffling digits independently.

    - For p01: shuffle pos0 and pos1 independently, recombine.
    - For p23: shuffle pos2 and pos3 independently, recombine.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 50 or trials <= 0:
        return float("nan")

    if col == "p01":
        a = df["pos0"].to_numpy()
        b = df["pos1"].to_numpy()
    elif col == "p23":
        a = df["pos2"].to_numpy()
        b = df["pos3"].to_numpy()
    else:
        raise ValueError("col must be p01 or p23")

    # observed
    obs = pair_chi2(df, col=col).chi2

    # Precompute expected
    exp = n / 100.0
    ge = 0
    for _ in range(trials):
        a2 = a.copy()
        b2 = b.copy()
        rng.shuffle(a2)
        rng.shuffle(b2)
        pairs = np.char.add(a2.astype(str), b2.astype(str))  # vectorized string add
        vc = pd.Series(pairs).value_counts()
        counts = np.array([vc.get(f"{i:02d}", 0) for i in range(100)], dtype=float)
        chi2 = _chisq_stat_from_counts(counts, exp)
        if chi2 >= obs:
            ge += 1
    # +1 smoothing
    return float((ge + 1) / (trials + 1))


def rolling_pair_tests(
    df: pd.DataFrame,
    col: str,
    rolling_days: int,
    baseline_hist: np.ndarray,
) -> pd.DataFrame:
    """
    Rolling window (end_date anchored):
      - n in window
      - chi2 + approx p
      - KL(window || baseline)
    """
    if rolling_days <= 0:
        raise ValueError("rolling_days must be > 0")

    df = df.sort_values("date")
    dates = pd.Series(sorted(df["date"].unique()))
    rows = []

    # Use a sliding pointer window over df indices
    dts = pd.to_datetime(df["date"])
    for end in dates:
        end_ts = pd.to_datetime(end)
        start_ts = end_ts - pd.Timedelta(days=rolling_days - 1)
        w = df[(dts >= start_ts) & (dts <= end_ts)]
        n = len(w)
        if n < 50:
            continue
        # chi2
        r = pair_chi2(w, col=col)
        # KL
        w_hist = _vector_pair_hist(w, col=col)
        kl = _kl_divergence(w_hist, baseline_hist)
        rows.append(
            {
                "end_date": str(end),
                "n": n,
                "chi2": r.chi2,
                "p_approx": r.p_approx,
                "kl_vs_baseline": kl,
            }
        )

    return pd.DataFrame(rows)


def overlaps_by_date(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Per date:
    - count overlaps between starter and consolation, top3 and others
    - count same-last2 overlap etc.
    """
    # Ensure number_str exists
    if "number_str" not in df_all.columns:
        raise ValueError("df must include number_str")
    df_all = df_all.copy()
    df_all["last2"] = df_all["number_str"].str[2:]
    df_all["first2"] = df_all["number_str"].str[:2]

    rows = []
    for dt, g in df_all.groupby("date"):
        buckets = {b: gg for b, gg in g.groupby("bucket")}
        top3 = buckets.get("top3")
        starter = buckets.get("starter")
        consolation = buckets.get("consolation")

        def _ov(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame], key: str) -> int:
            if a is None or b is None:
                return 0
            sa = set(a[key])
            sb = set(b[key])
            return len(sa.intersection(sb))

        rows.append(
            {
                "date": str(dt),
                "n_top3": 0 if top3 is None else len(top3),
                "n_starter": 0 if starter is None else len(starter),
                "n_consolation": 0 if consolation is None else len(consolation),
                "overlap_exact_starter_consolation": _ov(starter, consolation, "number_str"),
                "overlap_exact_top3_starter": _ov(top3, starter, "number_str"),
                "overlap_exact_top3_consolation": _ov(top3, consolation, "number_str"),
                "overlap_last2_starter_consolation": _ov(starter, consolation, "last2"),
                "overlap_first2_starter_consolation": _ov(starter, consolation, "first2"),
            }
        )
    return pd.DataFrame(rows)


def cross_bucket_digit_coupling(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, build digit histogram (pos0..pos3) per bucket.
    Then compute correlation across dates between buckets (per position).

    Output:
      bucket_a, bucket_b, pos, corr
    """
    # per date, per bucket, per pos: 10-dim distribution
    df_all = df_all.copy()
    df_all = df_all[df_all["bucket"].isin(["top3", "starter", "consolation"])]

    # Build per-date scalar signals: e.g. share of digit '5' in pos0
    # We'll compute correlation for each digit as well, and summarize by mean abs corr.
    out_rows = []
    buckets = ["top3", "starter", "consolation"]
    positions = ["pos0", "pos1", "pos2", "pos3"]

    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            a, b = buckets[i], buckets[j]
            for pos in positions:
                # Create wide frame: index=date, columns=digit0..digit9, values=share
                def wide(bucket: str) -> pd.DataFrame:
                    gg = df_all[df_all["bucket"] == bucket].groupby(["date", pos])[pos].count()
                    # gg has MultiIndex (date, digit) -> count
                    # Convert to shares
                    wide_counts = gg.unstack(fill_value=0).reindex(columns=[str(d) for d in range(10)], fill_value=0)
                    wide_shares = wide_counts.div(wide_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                    return wide_shares

                wa = wide(a)
                wb = wide(b)
                common = wa.index.intersection(wb.index)
                if len(common) < 30:
                    continue
                wa = wa.loc[common]
                wb = wb.loc[common]

                # per-digit correlation across dates
                corrs = []
                for d in range(10):
                    x = wa[str(d)].to_numpy()
                    y = wb[str(d)].to_numpy()
                    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                        continue
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
                if not corrs:
                    continue

                out_rows.append(
                    {
                        "bucket_a": a,
                        "bucket_b": b,
                        "pos": pos,
                        "n_dates": len(common),
                        "mean_corr": float(np.mean(corrs)),
                        "mean_abs_corr": float(np.mean(np.abs(corrs))),
                        "max_abs_corr": float(np.max(np.abs(corrs))),
                    }
                )

    return pd.DataFrame(out_rows)


def per_date_counts(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Per date, counts by bucket (and operator if present).
    Useful to detect missing rows / weird days.
    """
    cols = ["date", "bucket"]
    if "operator" in df_all.columns:
        cols.insert(1, "operator")
    g = df_all.groupby(cols).size().reset_index(name="n")
    # Pivot to date x bucket
    idx = ["date"]
    if "operator" in g.columns:
        idx = ["date", "operator"]
    pivot = g.pivot_table(index=idx, columns="bucket", values="n", fill_value=0).reset_index()
    # expected? just include columns if present
    for b in ["top3", "starter", "consolation"]:
        if b not in pivot.columns:
            pivot[b] = 0
    return pivot


# -----------------------------
# Main runner
# -----------------------------
def run_eda(
    numbers_long: str,
    bucket: str,
    operator: Optional[str],
    rolling_days: int,
    perm_trials: int,
    seed: int,
    out_prefix: str,
    dedupe: bool,
) -> None:
    df = _read_numbers_long(numbers_long, bucket=bucket, operator=operator, dedupe=dedupe)

    print(f"\nRows: {len(df)} (bucket={bucket}, operator={operator or 'all'})")
    print(f"Unique numbers: {df['number_str'].nunique()} / {len(df)}")
    print(f"Unique dates:   {df['date'].nunique()} (min={df['date'].min()}, max={df['date'].max()})")

    # Always also load all buckets (for overlaps/coupling) if we can
    df_all = _read_numbers_long(numbers_long, bucket="all", operator=operator, dedupe=dedupe)

    # 1) Integrity / overlaps
    dup_count = int(df_all.duplicated(subset=["date", "bucket", "number_str"]).sum())
    print(f"Duplicate (date,bucket,number) rows in ALL: {dup_count}")

    counts = per_date_counts(df_all)
    out_counts = f"{out_prefix}_per_date_counts.csv"
    _ensure_dir(out_counts)
    counts.to_csv(out_counts, index=False)
    print(f"Wrote: {out_counts}")

    ov = overlaps_by_date(df_all)
    out_ov = f"{out_prefix}_overlaps_by_date.csv"
    _ensure_dir(out_ov)
    ov.to_csv(out_ov, index=False)
    print(f"Wrote: {out_ov}")

    # 2) Digit marginals
    dm = digit_marginals_chi2(df)
    dm_df = pd.DataFrame([{"pos": r.label, "n": r.n, "chi2": r.chi2, "df": r.df, "p_approx": r.p_approx} for r in dm])
    dm_df["q_fdr"] = _fdr_bh(dm_df["p_approx"].to_numpy())
    out_dm = f"{out_prefix}_digit_marginals.csv"
    _ensure_dir(out_dm)
    dm_df.to_csv(out_dm, index=False)
    print(f"Wrote: {out_dm}")
    print("\nDigit marginals chi2 (approx):")
    print(dm_df.sort_values("p_approx").to_string(index=False))

    # 3) Pair tests (global)
    pr = []
    for col in ["p01", "p23"]:
        r = pair_chi2(df, col=col)
        pr.append({"pair": col, "n": r.n, "chi2": r.chi2, "df": r.df, "p_approx": r.p_approx})
    pr_df = pd.DataFrame(pr)
    pr_df["q_fdr"] = _fdr_bh(pr_df["p_approx"].to_numpy())
    out_pr = f"{out_prefix}_pair_global.csv"
    _ensure_dir(out_pr)
    pr_df.to_csv(out_pr, index=False)
    print(f"\nWrote: {out_pr}")
    print("\nPair chi2 (approx):")
    print(pr_df.sort_values("p_approx").to_string(index=False))

    if perm_trials > 0:
        for col in ["p01", "p23"]:
            pperm = perm_pvalue_pair_chi2(df, col=col, trials=perm_trials, seed=seed + (1 if col == "p01" else 2))
            print(f"Permutation p (>= chi2 obs) for {col}: {pperm:.6f} (trials={perm_trials})")

    # 4) Rolling windows for pair chi2 + KL
    if rolling_days > 0:
        # Baseline hist from full dataset (same bucket/op selection)
        base_p01 = _vector_pair_hist(df, col="p01")
        base_p23 = _vector_pair_hist(df, col="p23")

        roll_p01 = rolling_pair_tests(df, col="p01", rolling_days=rolling_days, baseline_hist=base_p01)
        roll_p23 = rolling_pair_tests(df, col="p23", rolling_days=rolling_days, baseline_hist=base_p23)

        out_r01 = f"{out_prefix}_rolling_{rolling_days}d_p01.csv"
        out_r23 = f"{out_prefix}_rolling_{rolling_days}d_p23.csv"
        _ensure_dir(out_r01)
        _ensure_dir(out_r23)
        roll_p01.to_csv(out_r01, index=False)
        roll_p23.to_csv(out_r23, index=False)
        print(f"\nWrote: {out_r01}")
        print(f"Wrote: {out_r23}")

        if len(roll_p01):
            worst = roll_p01.nsmallest(10, "p_approx")[["end_date", "n", "chi2", "p_approx", "kl_vs_baseline"]]
            print("\nWorst rolling windows (p01) by p_approx:")
            print(worst.to_string(index=False))
        if len(roll_p23):
            worst = roll_p23.nsmallest(10, "p_approx")[["end_date", "n", "chi2", "p_approx", "kl_vs_baseline"]]
            print("\nWorst rolling windows (p23) by p_approx:")
            print(worst.to_string(index=False))

    # 5) Cross-bucket coupling (always on full)
    cb = cross_bucket_digit_coupling(df_all)
    out_cb = f"{out_prefix}_cross_bucket_digit_coupling.csv"
    _ensure_dir(out_cb)
    cb.to_csv(out_cb, index=False)
    print(f"\nWrote: {out_cb}")
    if len(cb):
        print("\nCross-bucket coupling (higher = more suspicious coupling):")
        print(cb.sort_values("max_abs_corr", ascending=False).head(12).to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True, help="Path to numbers_long.csv")
    ap.add_argument("--bucket", default="all", help="all | top3 | starter | consolation")
    ap.add_argument("--operator", default=None, help="Optional operator filter (if column exists)")
    ap.add_argument("--rolling-days", type=int, default=180, help="Rolling window days for pair tests")
    ap.add_argument("--perm-trials", type=int, default=0, help="Permutation trials for pair chi2 (0 disables)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--out-dir", default="data/processed", help="Output directory")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate (date,bucket,number) rows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    b = args.bucket.lower().strip()
    op = args.operator.lower().strip() if args.operator else None

    tag = b
    if op:
        tag = f"{tag}_{op}"
    out_prefix = str(Path(args.out_dir) / f"eda_pairs_{tag}")

    run_eda(
        numbers_long=args.numbers_long,
        bucket=b,
        operator=op,
        rolling_days=args.rolling_days,
        perm_trials=args.perm_trials,
        seed=args.seed,
        out_prefix=out_prefix,
        dedupe=bool(args.dedupe),
    )


if __name__ == "__main__":
    main()
