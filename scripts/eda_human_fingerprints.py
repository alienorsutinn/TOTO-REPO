#!/usr/bin/env python3
import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def fmt4(x: int) -> str:
    return f"{int(x):04d}"


def digits(s: str) -> Tuple[int,int,int,int]:
    return (ord(s[0])-48, ord(s[1])-48, ord(s[2])-48, ord(s[3])-48)


def valid_mmdd(s: str) -> bool:
    mm = int(s[:2]); dd = int(s[2:])
    if mm < 1 or mm > 12: return False
    if dd < 1: return False
    # days per month (ignore leap; effect is tiny, and mmdd test is heuristic anyway)
    dpm = [31,28,31,30,31,30,31,31,30,31,30,31]
    return dd <= dpm[mm-1]


def valid_ddmm(s: str) -> bool:
    dd = int(s[:2]); mm = int(s[2:])
    if mm < 1 or mm > 12: return False
    if dd < 1: return False
    dpm = [31,28,31,30,31,30,31,31,30,31,30,31]
    return dd <= dpm[mm-1]


def count_mmdd_space() -> int:
    c = 0
    for x in range(10000):
        s = f"{x:04d}"
        if valid_mmdd(s): c += 1
    return c


def count_ddmm_space() -> int:
    c = 0
    for x in range(10000):
        s = f"{x:04d}"
        if valid_ddmm(s): c += 1
    return c


def norm_sf(z: float) -> float:
    # survival function 1 - Phi(z)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def p_ge_binom_normal_approx(n: int, p: float, k_obs: int) -> float:
    # P(X >= k_obs) using normal approx w/ continuity correction
    mu = n * p
    var = n * p * (1 - p)
    if var <= 0:
        return 1.0 if k_obs <= mu else 0.0
    # continuity correction: k_obs - 0.5
    z = ( (k_obs - 0.5) - mu ) / math.sqrt(var)
    return float(norm_sf(z))


def p_ge_poisson_approx(lam: float, k_obs: int) -> float:
    # P(X >= k_obs) for Poisson(lam)
    # compute tail via summing pmf up to k_obs-1
    if k_obs <= 0:
        return 1.0
    # cumulative up to k_obs-1
    s = 0.0
    term = math.exp(-lam)
    s += term
    for k in range(1, k_obs):
        term *= lam / k
        s += term
    return float(max(0.0, 1.0 - s))


def p_ge(n: int, p: float, k_obs: int) -> float:
    # choose poisson for small expectation, else normal
    lam = n * p
    if lam < 20:
        return p_ge_poisson_approx(lam, k_obs)
    return p_ge_binom_normal_approx(n, p, k_obs)


def fingerprint_counts(n4: np.ndarray) -> Dict[str, int]:
    c = {
        "all_same": 0,
        "palindrome_abba": 0,
        "aabb": 0,
        "abab": 0,
        "three_of_kind": 0,
        "ascending_run": 0,
        "descending_run": 0,
        "year_19xx": 0,
        "year_20xx": 0,
        "mmdd": 0,
        "ddmm": 0,
    }
    for s in n4:
        a,b,c2,d = digits(s)
        # all same
        if a==b==c2==d:
            c["all_same"] += 1
            continue

        # palindrome ABBA
        if a==d and b==c2:
            c["palindrome_abba"] += 1

        # AABB
        if a==b and c2==d and a!=c2:
            c["aabb"] += 1

        # ABAB
        if a==c2 and b==d and a!=b:
            c["abab"] += 1

        # three of a kind (exactly 3 same + 1 different)
        vals = [a,b,c2,d]
        uniq = {}
        for v in vals:
            uniq[v] = uniq.get(v, 0) + 1
        if 3 in uniq.values():
            c["three_of_kind"] += 1

        # ascending / descending runs (0123..6789 and 9876..3210)
        if (b==a+1) and (c2==b+1) and (d==c2+1):
            c["ascending_run"] += 1
        if (b==a-1) and (c2==b-1) and (d==c2-1):
            c["descending_run"] += 1

        # year-like
        if a==1 and b==9:
            c["year_19xx"] += 1
        if a==2 and b==0:
            c["year_20xx"] += 1

        # birthdayish
        if valid_mmdd(s):
            c["mmdd"] += 1
        if valid_ddmm(s):
            c["ddmm"] += 1

    return c


def last2_collision_stats(df: pd.DataFrame) -> Tuple[int, int, float, float]:
    """
    Returns:
      n_dates, obs_anydup_days, expected_anydup_days, p_ge_obs
    Assumes within a date, we look at last2 collisions.
    """
    g = df.groupby("date")["n4"].apply(list)
    n_dates = len(g)

    obs_anydup = 0
    exp_sum = 0.0
    var_sum = 0.0

    for _, lst in g.items():
        last2 = [x[-2:] for x in lst]
        anydup = int(len(set(last2)) < len(last2))
        obs_anydup += anydup

        n = len(last2)
        # under null: last2 iid uniform over 100 possibilities
        # P(no dup) = falling_factorial(100, n) / 100^n
        if n <= 1:
            p_any = 0.0
        else:
            ff = 1.0
            for k in range(n):
                ff *= (100 - k) / 100.0
            p_no = ff
            p_any = 1.0 - p_no

        exp_sum += p_any
        var_sum += p_any * (1.0 - p_any)

    # normal approx for sum of Bernoullis
    if var_sum <= 0:
        p_tail = 1.0 if obs_anydup <= exp_sum else 0.0
    else:
        z = ((obs_anydup - 0.5) - exp_sum) / math.sqrt(var_sum)
        p_tail = float(norm_sf(z))

    return n_dates, obs_anydup, exp_sum, p_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--bucket", default="all", choices=["all","top3","starter","consolation"])
    ap.add_argument("--out", default="data/processed/eda_human_fingerprints.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["n4"] = df["num"].astype(int).map(fmt4)

    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket]

    n4 = df["n4"].to_numpy(dtype=object)
    n = len(n4)

    if n == 0:
        raise SystemExit("No rows after filtering. Check bucket.")

    obs = fingerprint_counts(n4)

    # Exact uniform probabilities on 0000..9999
    p = {}
    p["all_same"] = 10 / 10000
    p["palindrome_abba"] = 100 / 10000        # A,B free => 10*10
    p["aabb"] = 90 / 10000                    # choose A!=B: 10*9
    p["abab"] = 90 / 10000                    # choose A!=B: 10*9
    # three of a kind: choose digit for triple (10) * choose position of odd (4) * choose odd digit (9) => 360 / 10000
    p["three_of_kind"] = 360 / 10000
    # ascending/descending runs: start 0..6 => 7 each
    p["ascending_run"] = 7 / 10000
    p["descending_run"] = 7 / 10000
    p["year_19xx"] = 100 / 10000
    p["year_20xx"] = 100 / 10000

    mmdd_space = count_mmdd_space()
    ddmm_space = count_ddmm_space()
    p["mmdd"] = mmdd_space / 10000
    p["ddmm"] = ddmm_space / 10000

    rows = []
    for k, obs_k in obs.items():
        pk = p.get(k, None)
        if pk is None:
            continue
        exp = n * pk
        p_tail = p_ge(n, pk, obs_k)
        rows.append({
            "metric": k,
            "n": n,
            "obs": int(obs_k),
            "p0": pk,
            "exp": exp,
            "obs_rate": obs_k / n,
            "p_ge_obs": p_tail,
        })

    out = pd.DataFrame(rows).sort_values("p_ge_obs")
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    print(f"\nBucket={args.bucket} n={n}")
    print(out.to_string(index=False))

    # within-day last2 collision check (this is where “plate-number vibes” would show up if real)
    n_dates, obs_anydup, exp_anydup, p_tail = last2_collision_stats(df)
    print("\nWithin-day last2 collision (any duplicate last2 within a date):")
    print(f"  n_dates: {n_dates}")
    print(f"  obs_anydup_days: {obs_anydup}  rate={obs_anydup/n_dates:.4f}")
    print(f"  exp_anydup_days: {exp_anydup:.2f}  rate={exp_anydup/n_dates:.4f}")
    print(f"  p_ge_obs (normal approx): {p_tail:.6f}")


if __name__ == "__main__":
    main()
