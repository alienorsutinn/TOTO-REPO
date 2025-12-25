#!/usr/bin/env python3
"""
Deep EDA + sanity checks for 4D numbers.

What this does:
- Basic shape checks (counts, unique, duplicates)
- Digit frequency by position (0..3), last digit, first digit
- Repeat structure (any repeat, doubles/triples/quads) + compare vs uniform theory
- Chi-square tests for digit uniformity (per position and per bucket)
- Simple serial-dependence checks:
  - last-digit transition matrix
  - "same last digit as previous draw" rate vs expectation
- Bucket comparisons (top3/starter/consolation/all)

Notes:
- This does NOT assume any "true" edge exists. It quantifies deviations + significance.
- Chi-square p-values are approximate (Wilson–Hilferty transform) without scipy.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


BUCKETS = ["all", "top3", "starter", "consolation"]


def _norm_bucket(b: str) -> str:
    b = (b or "").strip().lower()
    if b in ("all", ""):
        return "all"
    if b in ("top3", "top", "1st", "first", "second", "third"):
        return "top3"
    if b in ("starter", "start"):
        return "starter"
    if b in ("consolation", "cons"):
        return "consolation"
    raise ValueError(f"Unknown bucket: {b}. Use one of: {BUCKETS}")


def _read_numbers_long(path: str, bucket: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # expected columns (tolerate schema variants)
    # required: date + number column (number|num)
    if "date" not in df.columns:
        raise ValueError(f"numbers_long missing required column 'date'. Have: {list(df.columns)}")

    num_col = None
    for c in ["number", "num", "n4", "value"]:
        if c in df.columns:
            num_col = c
            break
    if num_col is None:
        raise ValueError(f"numbers_long missing a number column (expected one of: number,num,n4,value). Have: {list(df.columns)}")

    if "bucket" not in df.columns:
        # tolerate old schema by creating bucket=all
        df["bucket"] = "all"

    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # normalize number to 4-digit string (keep leading zeros)
    df["number_str"] = df[num_col].astype(str).str.strip().str.zfill(4)

    if bucket != "all":
        df = df[df["bucket"] == bucket].copy()

    df = df.sort_values(["date", "number_str"]).reset_index(drop=True)
    return df


def _digits_by_pos(number_str: str) -> Tuple[int, int, int, int]:
    return (ord(number_str[0]) - 48, ord(number_str[1]) - 48, ord(number_str[2]) - 48, ord(number_str[3]) - 48)


def _count_digits(df: pd.DataFrame) -> Dict[int, List[int]]:
    # returns {pos: [counts for digit 0..9]}
    counts = {0: [0] * 10, 1: [0] * 10, 2: [0] * 10, 3: [0] * 10}
    for s in df["number_str"].tolist():
        d0, d1, d2, d3 = _digits_by_pos(s)
        counts[0][d0] += 1
        counts[1][d1] += 1
        counts[2][d2] += 1
        counts[3][d3] += 1
    return counts


def _topk_dist(counts: List[int], k: int = 10) -> List[Tuple[str, float]]:
    total = sum(counts) or 1
    pairs = [(str(d), c / total) for d, c in enumerate(counts)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def _repeat_profile(s: str) -> Tuple[bool, int]:
    # returns (any_repeat, max_multiplicity)
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    mx = max(freq.values())
    return (mx >= 2, mx)


def _theory_repeat_probs() -> Dict[str, float]:
    # For 4 iid uniform digits 0..9
    # P(all distinct) = 10*9*8*7 / 10^4 = 5040/10000 = 0.504
    p_all_distinct = 5040 / 10000.0
    p_any_repeat = 1.0 - p_all_distinct

    # Exactly one pair (AABC with all digits not all distinct, not triple/quad):
    # Count = choose digit for pair (10) * choose 2 other distinct digits (C(9,2)=36) * permutations (4!/2!=12) = 4320
    p_exactly_one_pair = 4320 / 10000.0

    # Two pairs (AABB):
    # choose 2 digits for pairs C(10,2)=45 * permutations 4!/(2!2!)=6 => 270
    p_two_pairs = 270 / 10000.0

    # Triple (AAAB):
    # choose digit for triple (10) * choose other digit (9) * permutations 4!/3!=4 => 360
    p_triple = 360 / 10000.0

    # Quad (AAAA): 10 / 10000 = 0.001
    p_quad = 10 / 10000.0

    # sanity: all distinct + one pair + two pairs + triple + quad = 1
    return {
        "p_all_distinct": p_all_distinct,
        "p_any_repeat": p_any_repeat,
        "p_exactly_one_pair": p_exactly_one_pair,
        "p_two_pairs": p_two_pairs,
        "p_triple": p_triple,
        "p_quad": p_quad,
    }


def _chi_square_uniform(counts: List[int]) -> Tuple[float, float]:
    """
    Chi-square vs uniform over 10 digits.
    Returns (chi2_stat, approx_p_value).
    We approximate p-value for df=9 via Wilson–Hilferty transform to normal.
    """
    n = sum(counts)
    if n == 0:
        return (0.0, 1.0)
    exp = n / 10.0
    chi2 = 0.0
    for c in counts:
        chi2 += (c - exp) ** 2 / exp

    df = 9
    # Wilson–Hilferty transform:
    # If X ~ chi2(k), then Z = ((X/k)^(1/3) - (1 - 2/(9k))) / sqrt(2/(9k)) approx N(0,1)
    k = df
    z = ((chi2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    # p = P(Chi2 >= chi2) = P(Z >= z)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return (chi2, float(p))


def _transition_counts(values: List[int], n_states: int = 10) -> List[List[int]]:
    m = [[0] * n_states for _ in range(n_states)]
    for a, b in zip(values[:-1], values[1:]):
        m[a][b] += 1
    return m


def _same_as_prev_rate(values: List[int]) -> float:
    if len(values) < 2:
        return 0.0
    same = sum(1 for a, b in zip(values[:-1], values[1:]) if a == b)
    return same / (len(values) - 1)


def _print_transition_matrix(m: List[List[int]]) -> None:
    # show row-normalized top transitions per digit
    for a in range(10):
        row = m[a]
        total = sum(row)
        if total == 0:
            continue
        pairs = [(b, row[b] / total) for b in range(10)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top3 = ", ".join([f"{b}:{p:.3f}" for b, p in pairs[:3]])
        print(f"  from {a}: {top3}")


def _bucket_label(df: pd.DataFrame) -> str:
    b = df["bucket"].iloc[0] if len(df) else "all"
    return str(b)


@dataclass
class BucketReport:
    bucket: str
    n_rows: int
    n_unique_numbers: int
    n_unique_dates: int
    repeats_any: float
    repeats_max2: float
    repeats_max3: float
    repeats_max4: float


def _bucket_repeat_report(df: pd.DataFrame) -> BucketReport:
    reps_any = 0
    max2 = 0
    max3 = 0
    max4 = 0
    for s in df["number_str"].tolist():
        any_rep, mx = _repeat_profile(s)
        if any_rep:
            reps_any += 1
        if mx == 2:
            max2 += 1
        elif mx == 3:
            max3 += 1
        elif mx == 4:
            max4 += 1
    n = len(df) or 1
    return BucketReport(
        bucket=_bucket_label(df),
        n_rows=len(df),
        n_unique_numbers=int(df["number_str"].nunique()) if len(df) else 0,
        n_unique_dates=int(df["date"].nunique()) if len(df) else 0,
        repeats_any=reps_any / n,
        repeats_max2=max2 / n,
        repeats_max3=max3 / n,
        repeats_max4=max4 / n,
    )


def run_one(numbers_long: str, bucket: str) -> None:
    df = _read_numbers_long(numbers_long, bucket=bucket)
    print(f"Rows: {len(df)} (bucket={bucket})")
    if len(df) == 0:
        return

    # Basic dataset checks
    dup_rows = df.duplicated(subset=["date", "bucket", "number_str"]).sum()
    print(f"Unique numbers: {df['number_str'].nunique()} / {len(df)}")
    print(f"Unique dates:   {df['date'].nunique()} (min={df['date'].min()}, max={df['date'].max()})")
    print(f"Duplicate (date,bucket,number) rows: {int(dup_rows)}")

    # Digit frequency by position
    counts = _count_digits(df)
    print("\nDigit frequency by position (0=thousands ... 3=ones):")
    for pos in range(4):
        top10 = _topk_dist(counts[pos], 10)
        total = sum(counts[pos])
        print(f"  pos{pos} total={total} top10={[(d, round(p,4)) for d,p in top10]}")

    # Chi-square tests for uniformity by position
    print("\nChi-square (uniform digits) by position:")
    for pos in range(4):
        chi2, p = _chi_square_uniform(counts[pos])
        print(f"  pos{pos}: chi2={chi2:.2f}, df=9, approx_p={p:.4f}")

    # Last digit
    last_counts = counts[3]
    print("\nLast digit distribution:")
    print([(d, round(p, 4)) for d, p in _topk_dist(last_counts, 10)])

    # Repeat structure + compare to theory
    rep = _bucket_repeat_report(df)
    theory = _theory_repeat_probs()
    print("\nRepeated-digit structure:")
    print(f"  any repeat: {rep.repeats_any:.4f} (theory ~{theory['p_any_repeat']:.4f})")
    print(f"  max multiplicity ==2: {rep.repeats_max2:.4f} (theory one-pair+two-pairs ~{theory['p_exactly_one_pair']+theory['p_two_pairs']:.4f})")
    print(f"  max multiplicity ==3: {rep.repeats_max3:.4f} (theory triple ~{theory['p_triple']:.4f})")
    print(f"  max multiplicity ==4: {rep.repeats_max4:.4f} (theory quad ~{theory['p_quad']:.4f})")

    # Serial dependence checks (requires stable "draw order" proxy)
    # We'll use each day's numbers sorted lexicographically; then flatten by date.
    # This is not perfect (true published order may differ), but still catches large effects.
    flat_last_digits: List[int] = []
    for dt, g in df.groupby("date", sort=True):
        # keep stable within-day order
        for s in g["number_str"].sort_values().tolist():
            flat_last_digits.append(ord(s[3]) - 48)

    same_rate = _same_as_prev_rate(flat_last_digits)
    # For iid uniform digits, P(same as previous) = 0.1
    print("\nSerial checks (last digit, flattened by date + within-day sort):")
    print(f"  P(last_digit[t] == last_digit[t-1]): {same_rate:.4f} (iid expectation ~0.1000)")
    tm = _transition_counts(flat_last_digits, 10)
    print("  Top transitions per digit (row-normalized):")
    _print_transition_matrix(tm)

    # Year-by-year last digit top3
    print("\nYearly last-digit top3:")
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    for y, g in df2.groupby("year"):
        c = [0] * 10
        for s in g["number_str"].tolist():
            c[ord(s[3]) - 48] += 1
        top3 = _topk_dist(c, 3)
        print(f"  {y}: {[(d, round(p,4)) for d,p in top3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True, help="data/processed/numbers_long.csv")
    ap.add_argument("--bucket", default="all", help="all|top3|starter|consolation")
    args = ap.parse_args()

    bucket = _norm_bucket(args.bucket)
    run_one(args.numbers_long, bucket=bucket)


if __name__ == "__main__":
    main()
