# --- patched: add 'none' to --correction choices ---
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import (
    chisquare,
    chi2_contingency,
    binomtest,
    pearsonr,
    kstest,
    power_divergence,
)

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numbers-long", required=True, help="numbers_long.csv with date,bucket,num (+optional operator)")
    p.add_argument("--bucket", default="all", choices=["all", "top3", "starter", "consolation"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--correction", default="bh", choices=["none", "bh", "holm", "bonferroni"])
    p.add_argument("--out", default="data/processed/stat_tests_report.csv")
    p.add_argument("--max-lags", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mc", type=int, default=3000, help="Monte Carlo reps for permutation-based tests")
    return p.parse_args()


    # --- patched: support --correction none (exploratory) ---
    # --- end patched block ---

# -----------------------------
# Multiple testing correction
# -----------------------------
def p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def p_adjust_bonferroni(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    return np.clip(p * len(p), 0.0, 1.0)


def p_adjust_holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 0.0
    for i in range(n):
        val = (n - i) * ranked[i]
        prev = max(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def cramer_v(chi2: float, n: int, k: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    return float(math.sqrt(max(0.0, chi2 / (n * k))))


# -----------------------------
# Feature helpers
# -----------------------------
def digits_matrix(nums: np.ndarray) -> np.ndarray:
    # nums are strings "0000".."9999"
    a = np.fromiter((ord(s[0]) - 48 for s in nums), dtype=np.int16)
    b = np.fromiter((ord(s[1]) - 48 for s in nums), dtype=np.int16)
    c = np.fromiter((ord(s[2]) - 48 for s in nums), dtype=np.int16)
    d = np.fromiter((ord(s[3]) - 48 for s in nums), dtype=np.int16)
    return np.vstack([a, b, c, d]).T


def expected_by_enumeration() -> Dict[str, object]:
    nums = np.array([f"{i:04d}" for i in range(10000)], dtype=object)
    d = digits_matrix(nums)
    a, b, c, e = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    sorted_row = np.sort(d, axis=1)
    any_repeat = ~np.all(np.diff(sorted_row, axis=1) != 0, axis=1)
    all_same = (a == b) & (b == c) & (c == e)
    abba = (a == e) & (b == c)
    abab = (a == c) & (b == e) & (a != b)
    aabb = (a == b) & (c == e) & (a != c)

    # multiplicity categories
    def mult_cat(row: np.ndarray) -> str:
        _, cnt = np.unique(row, return_counts=True)
        cnt = sorted(cnt, reverse=True)
        if cnt == [4]:
            return "4"
        if cnt == [3, 1]:
            return "3+1"
        if cnt == [2, 2]:
            return "2+2"
        if cnt == [2, 1, 1]:
            return "2+1+1"
        return "1+1+1+1"

    cats = np.array([mult_cat(r) for r in d], dtype=object)
    levels = ["4", "3+1", "2+2", "2+1+1", "1+1+1+1"]
    cat_probs = {k: float(np.mean(cats == k)) for k in levels}

    sums = d.sum(axis=1)
    sum_pmf = np.bincount(sums, minlength=37) / 10000.0

    last2 = (d[:, 2] * 10 + d[:, 3]).astype(int)
    last2_pmf = np.bincount(last2, minlength=100) / 10000.0

    return {
        "p_any_repeat": float(any_repeat.mean()),
        "p_all_same": float(all_same.mean()),
        "p_abba": float(abba.mean()),
        "p_abab": float(abab.mean()),
        "p_aabb": float(aabb.mean()),
        "cat_probs": cat_probs,
        "sum_pmf": sum_pmf,
        "last2_pmf": last2_pmf,
    }


def shannon_entropy(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def runs_test_binary(x: np.ndarray) -> Tuple[float, float]:
    # Wald–Wolfowitz runs test (normal approximation)
    x = np.asarray(x, dtype=int)
    n1 = int(np.sum(x == 1))
    n0 = int(np.sum(x == 0))
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0
    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    mu = 1 + (2 * n1 * n0) / (n1 + n0)
    var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    z = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(z), float(p)


def perm_pvalue_ge(obs: float, null: np.ndarray) -> float:
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def perm_pvalue_le(obs: float, null: np.ndarray) -> float:
    return float((np.sum(null <= obs) + 1) / (len(null) + 1))


@dataclass
class Result:
    test: str
    n: int
    stat: float
    p: float
    effect: float
    details: str


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.numbers_long)
    if args.bucket != "all":
        df = df[df["bucket"] == args.bucket].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # find any dataframe-like object in locals that has p and p_adj
    for _name, _obj in list(locals().items()):
        try:
            cols = list(getattr(_obj, 'columns', []))
        except Exception:
            cols = []
        if cols and ('p' in cols) and ('p_adj' in cols):
            _obj['p_adj'] = _obj['p'].astype(float)
            for _flag in ['is_sig', 'significant']:
                if _flag in cols:
                    _obj[_flag] = _obj['p_adj'] <= float(getattr(args, 'alpha', 0.05))
# --- end patched block ---

        df = df[df["date"].notna()].sort_values("date")

    df["num"] = df["num"].astype(str).str.zfill(4)
    df = df[df["num"].str.fullmatch(r"\d{4}", na=False)].copy()

    if len(df) < 50:
        raise SystemExit(f"Not enough rows after filtering: n={len(df)}")

    nums = df["num"].to_numpy(dtype=object)
    d = digits_matrix(nums)
    n = len(nums)
    exp = expected_by_enumeration()

    results: List[Result] = []

    # -----------------------------
    # A) Digit uniformity
    # -----------------------------
    all_digits = d.reshape(-1)
    obs = np.bincount(all_digits, minlength=10)
    exp_counts = np.ones(10) * (len(all_digits) / 10.0)

    chi2, p = chisquare(obs, f_exp=exp_counts)
    results.append(Result("A1_chi2_digits_overall", int(len(all_digits)), float(chi2), float(p), cramer_v(float(chi2), int(len(all_digits)), 9), "pooled digits"))

    for pos in range(4):
        obs_pos = np.bincount(d[:, pos], minlength=10)
        exp_pos = np.ones(10) * (n / 10.0)
        chi2, p = chisquare(obs_pos, f_exp=exp_pos)
        results.append(Result(f"A2_chi2_pos{pos}", n, float(chi2), float(p), cramer_v(float(chi2), n, 9), f"position={pos}"))

    g, p = power_divergence(obs, f_exp=exp_counts, lambda_="log-likelihood")
    results.append(Result("A3_gtest_digits_overall", int(len(all_digits)), float(g), float(p), cramer_v(float(g), int(len(all_digits)), 9), "G-test pooled digits"))

    for dig in range(10):
        k = int(np.sum(all_digits == dig))
        bt = binomtest(k, n=int(len(all_digits)), p=0.10, alternative="two-sided")
        results.append(Result(f"A4_binom_digit_{dig}", int(len(all_digits)), float(k), float(bt.pvalue), (k / len(all_digits)) - 0.10, "pooled digits"))

    # Entropy permutation (flag low entropy)
    H_obs = shannon_entropy(obs)
    null_H = []
    for _ in range(args.mc):
        sim = rng.integers(0, 10, size=len(all_digits))
        sim_obs = np.bincount(sim, minlength=10)
        null_H.append(shannon_entropy(sim_obs))
    null_H = np.array(null_H)
    p_ent = perm_pvalue_le(H_obs, null_H)
    results.append(Result("A6_entropy_perm_low", int(len(all_digits)), float(H_obs), float(p_ent), float(H_obs), "perm: lower-than-expected entropy"))

    # Jensen–Shannon distance permutation (flag high JS)
    p_hat = obs / obs.sum()
    q = np.ones(10) / 10.0
    m = 0.5 * (p_hat + q)
    js_obs = 0.5 * (np.sum(p_hat * np.log2(p_hat / m)) + np.sum(q * np.log2(q / m)))

    js_null = []
    for _ in range(args.mc):
        sim = rng.integers(0, 10, size=len(all_digits))
        sim_obs = np.bincount(sim, minlength=10)
        p_sim = sim_obs / sim_obs.sum()
        m2 = 0.5 * (p_sim + q)
        js = 0.5 * (np.sum(p_sim * np.log2(p_sim / m2)) + np.sum(q * np.log2(q / m2)))
        js_null.append(js)
    js_null = np.array(js_null)
    p_js = perm_pvalue_ge(js_obs, js_null)
    results.append(Result("A7_js_distance_perm_high", int(len(all_digits)), float(js_obs), float(p_js), float(js_obs), "perm: higher-than-expected JS"))

    # -----------------------------
    # B) Structure / repeats / patterns
    # -----------------------------
    sorted_row = np.sort(d, axis=1)
    any_repeat = ~np.all(np.diff(sorted_row, axis=1) != 0, axis=1)
    k_rep = int(any_repeat.sum())
    bt = binomtest(k_rep, n=n, p=float(exp["p_any_repeat"]), alternative="two-sided")
    results.append(Result("B1_repeat_rate_binom", n, float(k_rep), float(bt.pvalue), (k_rep / n) - float(exp["p_any_repeat"]), f"p0={exp['p_any_repeat']:.6f}"))

    a, b, c, e = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    all_same = (a == b) & (b == c) & (c == e)
    abba = (a == e) & (b == c)
    abab = (a == c) & (b == e) & (a != b)
    aabb = (a == b) & (c == e) & (a != c)

    for name, mask, p0 in [
        ("B4_all_same_binom", all_same, float(exp["p_all_same"])),
        ("B10_abab_binom", abab, float(exp["p_abab"])),
        ("B11_abba_palindrome_binom", abba, float(exp["p_abba"])),
        ("B10_aabb_binom", aabb, float(exp["p_aabb"])),
    ]:
        k = int(mask.sum())
        bt = binomtest(k, n=n, p=p0, alternative="two-sided")
        results.append(Result(name, n, float(k), float(bt.pvalue), (k / n) - p0, f"p0={p0:.6f}"))

    k0 = int(np.sum(a == 0))
    bt = binomtest(k0, n=n, p=0.10, alternative="two-sided")
    results.append(Result("B12_leading_zero_binom", n, float(k0), float(bt.pvalue), (k0 / n) - 0.10, "p0=0.10"))

    # multiplicity categories distribution
    def mult_cat(row: np.ndarray) -> str:
        _, cnt = np.unique(row, return_counts=True)
        cnt = sorted(cnt, reverse=True)
        if cnt == [4]:
            return "4"
        if cnt == [3, 1]:
            return "3+1"
        if cnt == [2, 2]:
            return "2+2"
        if cnt == [2, 1, 1]:
            return "2+1+1"
        return "1+1+1+1"

    cats = np.array([mult_cat(r) for r in d], dtype=object)
    levels = ["4", "3+1", "2+2", "2+1+1", "1+1+1+1"]
    obs_cat = np.array([np.sum(cats == lv) for lv in levels], dtype=float)
    exp_probs = exp["cat_probs"]
    exp_cat = np.array([float(exp_probs[lv]) * n for lv in levels], dtype=float)
    chi2, p = chisquare(obs_cat, f_exp=exp_cat)
    results.append(Result("B2_multiplicity_cat_chi2", n, float(chi2), float(p), cramer_v(float(chi2), n, len(levels) - 1), "cats 4 / 3+1 / 2+2 / 2+1+1 / distinct"))

    # -----------------------------
    # C) Independence across time (derived series)
    # -----------------------------
    last_digit = d[:, 3].astype(int)
    s = (last_digit >= 5).astype(int)
    z, p_runs = runs_test_binary(s)
    results.append(Result("C1_runs_lastdigit_ge5", n, float(z), float(p_runs), float(z), "runs test on lastdigit>=5"))

    if n >= 3:
        r, p_corr = pearsonr(last_digit[:-1], last_digit[1:])
        results.append(Result("C2_serial_corr_lastdigit_lag1", n - 1, float(r), float(p_corr), float(r), "pearson(last_t,last_t+1)"))

        lb = acorr_ljungbox(last_digit - np.mean(last_digit), lags=min(args.max_lags, n - 2), return_df=True)
        p_lb = float(lb["lb_pvalue"].min())
        stat_lb = float(lb.loc[lb["lb_pvalue"].idxmin(), "lb_stat"])
        results.append(Result("C3_ljung_box_lastdigit", n, stat_lb, p_lb, stat_lb, f"min p over lags<= {min(args.max_lags, n-2)}"))

        dw = float(durbin_watson(last_digit - np.mean(last_digit)))
        results.append(Result("C4_durbin_watson_lastdigit", n, dw, 1.0, dw, "diagnostic only"))

    # Markov transition chi2 (0-4 vs 5-9)
    if n >= 100:
        state = (last_digit >= 5).astype(int)
        tab = np.zeros((2, 2), dtype=float)
        for i, j in zip(state[:-1], state[1:]):
            tab[int(i), int(j)] += 1
        chi2, p, _, _ = chi2_contingency(tab)
        results.append(Result("C6_markov_chi2_lastdigit_bucket", n - 1, float(chi2), float(p), cramer_v(float(chi2), int(tab.sum()), 1), "states:0-4 vs 5-9"))
    else:
        results.append(Result("C6_markov_chi2_lastdigit_bucket", n - 1, float("nan"), 1.0, float("nan"), "skipped: need n>=100"))

    # -----------------------------
    # D) Pair dependencies (within number)
    # -----------------------------
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        tab = np.zeros((10, 10), dtype=float)
        for di, dj in zip(d[:, i], d[:, j]):
            tab[int(di), int(dj)] += 1
        chi2, p, _, _ = chi2_contingency(tab)
        results.append(Result(f"D1_chi2_indep_pos{i}_pos{j}", n, float(chi2), float(p), cramer_v(float(chi2), n, 9), "10x10 contingency"))

    # 2-gram within-number test
    pairs = np.concatenate([d[:, 0] * 10 + d[:, 1], d[:, 1] * 10 + d[:, 2], d[:, 2] * 10 + d[:, 3]]).astype(int)
    obs2 = np.bincount(pairs, minlength=100).astype(float)
    exp2 = np.ones(100) * (len(pairs) / 100.0)
    chi2, p = chisquare(obs2, f_exp=exp2)
    results.append(Result("D4_ngram_2gram_chi2", int(len(pairs)), float(chi2), float(p), cramer_v(float(chi2), int(len(pairs)), 99), "pairs: (0-1),(1-2),(2-3)"))

    # -----------------------------
    # E) Derived feature distributions
    # -----------------------------
    sums = d.sum(axis=1).astype(int)
    obs_sum = np.bincount(sums, minlength=37).astype(float)
    exp_sum = np.asarray(exp["sum_pmf"], dtype=float) * n
    chi2, p = chisquare(obs_sum, f_exp=exp_sum)
    results.append(Result("E1_sumdigits_chi2", n, float(chi2), float(p), cramer_v(float(chi2), n, 36), "sum 0..36"))

    last2 = (d[:, 2] * 10 + d[:, 3]).astype(int)
    obs_last2 = np.bincount(last2, minlength=100).astype(float)
    exp_last2 = np.asarray(exp["last2_pmf"], dtype=float) * n
    chi2, p = chisquare(obs_last2, f_exp=exp_last2)
    results.append(Result("E2_last2digits_chi2", n, float(chi2), float(p), cramer_v(float(chi2), n, 99), "00..99"))

    val = np.array([int(x) for x in nums], dtype=float) / 10000.0
    D, p = kstest(val, "uniform")
    results.append(Result("E3_ks_number_uniform01", n, float(D), float(p), float(D), "int(num)/10000 uniform"))

    # -----------------------------
    # F) Simple regime shift check (split-half)
    # -----------------------------
    mid = n // 2
    d1 = d[:mid].reshape(-1)
    d2 = d[mid:].reshape(-1)
    tab = np.vstack([np.bincount(d1, minlength=10), np.bincount(d2, minlength=10)]).astype(float)
    chi2, p, _, _ = chi2_contingency(tab)
    results.append(Result("F1_split_half_digit_dist_chi2", int(tab.sum()), float(chi2), float(p), cramer_v(float(chi2), int(tab.sum()), 1), "digits: first half vs second half"))

    rep1 = int((~np.all(np.diff(np.sort(d[:mid], axis=1), axis=1) != 0, axis=1)).sum())
    rep2 = int((~np.all(np.diff(np.sort(d[mid:], axis=1), axis=1) != 0, axis=1)).sum())
    tab2 = np.array([[rep1, mid - rep1], [rep2, (n - mid) - rep2]], dtype=float)
    chi2, p, _, _ = chi2_contingency(tab2)
    results.append(Result("F2_split_half_repeat_rate_chi2", int(tab2.sum()), float(chi2), float(p), cramer_v(float(chi2), int(tab2.sum()), 1), "repeat yes/no: first vs second half"))

    # -----------------------------
    # G) Compare groups (operator, weekday/weekend)
    # -----------------------------
    if "operator" in df.columns:
        ops = df["operator"].astype(str).fillna("NA")
        top_ops = ops.value_counts().head(2).index.tolist()
        if len(top_ops) == 2:
            aop, bop = top_ops
            da = digits_matrix(df.loc[ops == aop, "num"].to_numpy(dtype=object)).reshape(-1)
            db = digits_matrix(df.loc[ops == bop, "num"].to_numpy(dtype=object)).reshape(-1)
            tab = np.vstack([np.bincount(da, minlength=10), np.bincount(db, minlength=10)]).astype(float)
            chi2, p, _, _ = chi2_contingency(tab)
            results.append(Result("G1_operator_digit_dist_chi2", int(tab.sum()), float(chi2), float(p), cramer_v(float(chi2), int(tab.sum()), 1), f"{aop} vs {bop} (top2)"))
        else:
            results.append(Result("G1_operator_digit_dist_chi2", n * 4, float("nan"), 1.0, float("nan"), "skipped: need 2 operators"))
    else:
        results.append(Result("G1_operator_digit_dist_chi2", n * 4, float("nan"), 1.0, float("nan"), "skipped: no operator"))

    if "date" in df.columns and df["date"].notna().all():
        wd = pd.to_datetime(df["date"]).dt.dayofweek.to_numpy()
        wk = digits_matrix(df.loc[wd <= 4, "num"].to_numpy(dtype=object)).reshape(-1)
        we = digits_matrix(df.loc[wd >= 5, "num"].to_numpy(dtype=object)).reshape(-1)
        if len(wk) > 0 and len(we) > 0:
            tab = np.vstack([np.bincount(wk, minlength=10), np.bincount(we, minlength=10)]).astype(float)
            chi2, p, _, _ = chi2_contingency(tab)
            results.append(Result("G2_weekday_vs_weekend_digit_chi2", int(tab.sum()), float(chi2), float(p), cramer_v(float(chi2), int(tab.sum()), 1), "Mon-Fri vs Sat-Sun"))
        else:
            results.append(Result("G2_weekday_vs_weekend_digit_chi2", n * 4, float("nan"), 1.0, float("nan"), "skipped: need both weekday & weekend"))
    else:
        results.append(Result("G2_weekday_vs_weekend_digit_chi2", n * 4, float("nan"), 1.0, float("nan"), "skipped: date missing/unparseable"))

    # -----------------------------
    # Multiple testing correction + export
    # -----------------------------
    out = pd.DataFrame([r.__dict__ for r in results])
    out["p"] = out["p"].astype(float)
    if args.correction == "none":
        out["p_adj"] = out["p"].to_numpy(dtype=float)
    elif args.correction == "bh":
        out["p_adj"] = p_adjust_bh(out["p"].to_numpy(dtype=float))
    elif args.correction == "holm":
        out["p_adj"] = p_adjust_holm(out["p"].to_numpy(dtype=float))
    else:
        out["p_adj"] = p_adjust_bonferroni(out["p"].to_numpy(dtype=float))

    out["flag"] = out["p_adj"] <= args.alpha
    out = out.sort_values(["flag", "p_adj", "p"], ascending=[False, True, True])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nWrote report: {args.out}")
    print(f"Tests: {len(out)} | Significant (adj p <= {args.alpha}): {int(out['flag'].sum())}")
    print("\nTop 12 (by adjusted p):")
    print(out.head(12)[["test", "p", "p_adj", "effect", "details"]].to_string(index=False))


if __name__ == "__main__":
    main()
