#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def fmt4(x: int) -> str:
    return f"{int(x):04d}"

def is_mmdd(s: str) -> bool:
    mm = int(s[:2]); dd = int(s[2:])
    return 1 <= mm <= 12 and 1 <= dd <= 31

def is_ddmm(s: str) -> bool:
    dd = int(s[:2]); mm = int(s[2:])
    return 1 <= mm <= 12 and 1 <= dd <= 31

def year_19xx(s: str) -> bool:
    return s.startswith("19")

def year_20xx(s: str) -> bool:
    return s.startswith("20")

def three_of_kind(s: str) -> bool:
    vals = [s.count(ch) for ch in set(s)]
    return 3 in vals

def all_same(s: str) -> bool:
    return len(set(s)) == 1

def abab(s: str) -> bool:
    return s[0] == s[2] and s[1] == s[3] and s[0] != s[1]

def aabb(s: str) -> bool:
    return s[0] == s[1] and s[2] == s[3] and s[0] != s[2]

def abba(s: str) -> bool:
    return s[0] == s[3] and s[1] == s[2] and s[0] != s[1]

def asc_run(s: str) -> bool:
    a = list(map(int, s))
    return a[0]+1==a[1] and a[1]+1==a[2] and a[2]+1==a[3]

def desc_run(s: str) -> bool:
    a = list(map(int, s))
    return a[0]-1==a[1] and a[1]-1==a[2] and a[2]-1==a[3]

def build_priors(df_all: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    cnt = df_all["n4"].value_counts().rename("draw_count").to_frame()
    all_nums = pd.Index([f"{i:04d}" for i in range(10000)], name="n4")
    cnt = cnt.reindex(all_nums).fillna(0.0)
    cnt["draw_count"] = cnt["draw_count"].astype(int)
    tot = int(cnt["draw_count"].sum())
    cnt["draw_freq_sm"] = (cnt["draw_count"] + alpha) / (tot + alpha * 10000)
    return cnt.reset_index()

def featurize_all() -> pd.DataFrame:
    rows = []
    for i in range(10000):
        s = f"{i:04d}"
        rows.append({
            "n4": s,
            "lead0": int(s[0] == "0"),
            "mmdd": int(is_mmdd(s)),
            "ddmm": int(is_ddmm(s)),
            "year_19xx": int(year_19xx(s)),
            "year_20xx": int(year_20xx(s)),
            "three_kind": int(three_of_kind(s)),
            "all_same": int(all_same(s)),
            "abab": int(abab(s)),
            "aabb": int(aabb(s)),
            "abba": int(abba(s)),
            "asc_run": int(asc_run(s)),
            "desc_run": int(desc_run(s)),
        })
    return pd.DataFrame(rows)

def diversify(df: pd.DataFrame, top: int, min_last2: int, min_pos0: int) -> pd.DataFrame:
    picks = []
    used_last2 = set()
    used_pos0 = set()
    for _, r in df.iterrows():
        if len(picks) >= top:
            break
        last2 = r["n4"][-2:]
        pos0 = r["n4"][0]
        if min_last2 > 0 and last2 in used_last2:
            continue
        if min_pos0 > 0 and pos0 in used_pos0 and len(used_pos0) >= min_pos0:
            continue
        picks.append(r)
        used_last2.add(last2)
        used_pos0.add(pos0)
    return pd.DataFrame(picks)

def pick_today(feats_pri: pd.DataFrame, top: int, w_pattern: float, w_drawfreq: float, w_lead0: float,
              min_last2: int, min_pos0: int) -> pd.DataFrame:
    df = feats_pri.copy()

    pattern = (
        df["mmdd"] + df["ddmm"] +
        0.7*df["year_19xx"] + 0.7*df["year_20xx"] +
        0.6*df["three_kind"] + 0.8*df["all_same"] +
        0.6*df["abab"] + 0.6*df["aabb"] + 0.6*df["abba"] +
        0.6*df["asc_run"] + 0.6*df["desc_run"]
    )

    # popularity penalty (higher freq => higher log => worse)
    freq_pen = np.log(df["draw_freq_sm"].astype(float) + 1e-12)

    df["score"] = w_pattern*pattern + w_lead0*df["lead0"] + w_drawfreq*freq_pen
    ranked = df.sort_values(["score", "n4"]).reset_index(drop=True)
    picks = diversify(ranked, top=top, min_last2=min_last2, min_pos0=min_pos0)
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--w-pattern", type=float, default=1.0)
    ap.add_argument("--w-drawfreq", type=float, default=1.0)
    ap.add_argument("--w-lead0", type=float, default=0.25)
    ap.add_argument("--min-last2", type=int, default=1)
    ap.add_argument("--min-pos0", type=int, default=3)
    ap.add_argument("--random-trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/backtest_anticrowd.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.numbers_long)
    df["bucket"] = df["bucket"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["n4"] = df["num"].astype(int).map(fmt4)

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    # priors from all observed results
    pri = build_priors(df, alpha=1.0)
    feats = featurize_all()
    feats_pri = feats.merge(pri, on="n4", how="left")
    feats_pri["draw_freq_sm"] = feats_pri["draw_freq_sm"].fillna(feats_pri["draw_freq_sm"].median())

    # target: top3 winners each day
    top3 = df[df["bucket"] == "top3"].groupby("date")["n4"].apply(set).sort_index()
    dates = top3.index.to_list()

    rows = []
    for dt in dates:
        picks = pick_today(
            feats_pri,
            top=args.top,
            w_pattern=args.w_pattern,
            w_drawfreq=args.w_drawfreq,
            w_lead0=args.w_lead0,
            min_last2=args.min_last2,
            min_pos0=args.min_pos0,
        )
        pickset = set(picks["n4"].tolist())
        hits = top3.loc[dt] & pickset
        rows.append({
            "date": str(dt.date()),
            "hits_count": len(hits),
            "any_hit": int(len(hits) > 0),
            "hits": ",".join(sorted(hits)),
            "mean_pattern": float((picks["mmdd"] + picks["ddmm"]).mean()),
            "mean_drawfreq_sm": float(picks["draw_freq_sm"].mean()),
            "picks": ",".join(picks["n4"].tolist()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

    any_hit_rate = float(out["any_hit"].mean()) if len(out) else 0.0
    avg_hits = float(out["hits_count"].mean()) if len(out) else 0.0
    print("\nObserved:")
    print("  days_tested:", len(out))
    print("  any_hit_rate:", any_hit_rate)
    print("  avg_hits_per_day:", avg_hits)

    # random baseline (uniform) with same diversity rules
    rng = np.random.default_rng(args.seed)
    ge_any = 0
    ge_avg = 0
    for _ in range(args.random_trials):
        any_hits = []
        hit_counts = []
        for dt in dates:
            cand = rng.integers(0, 10000, size=5000)
            cand_df = pd.DataFrame({"n4": [f"{x:04d}" for x in cand]}).drop_duplicates("n4")
            cand_df["pos0"] = cand_df["n4"].str[0]
            cand_df["last2"] = cand_df["n4"].str[-2:]

            picks = []
            used_last2 = set()
            used_pos0 = set()
            for _, r in cand_df.iterrows():
                if len(picks) >= args.top:
                    break
                if args.min_last2 > 0 and r["last2"] in used_last2:
                    continue
                if args.min_pos0 > 0 and r["pos0"] in used_pos0 and len(used_pos0) >= args.min_pos0:
                    continue
                picks.append(r["n4"])
                used_last2.add(r["last2"])
                used_pos0.add(r["pos0"])

            pickset = set(picks)
            hits = top3.loc[dt] & pickset
            any_hits.append(int(len(hits) > 0))
            hit_counts.append(len(hits))

        any_r = float(np.mean(any_hits))
        avg_r = float(np.mean(hit_counts))
        if any_r >= any_hit_rate:
            ge_any += 1
        if avg_r >= avg_hits:
            ge_avg += 1

    p_any = (ge_any + 1) / (args.random_trials + 1)
    p_avg = (ge_avg + 1) / (args.random_trials + 1)
    print("\nP-values vs random (>= observed):")
    print("  p_any_hit_rate:", p_any)
    print("  p_avg_hits_per_day:", p_avg)
    print("  trials:", args.random_trials)

    print("\nLast 10 rows:")
    print(out.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
