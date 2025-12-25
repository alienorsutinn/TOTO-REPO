#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def is_mmdd(s):
    mm=int(s[:2]); dd=int(s[2:])
    return 1 <= mm <= 12 and 1 <= dd <= 31

def is_ddmm(s):
    dd=int(s[:2]); mm=int(s[2:])
    return 1 <= dd <= 31 and 1 <= mm <= 12

def year_19xx(s): return s.startswith("19")
def year_20xx(s): return s.startswith("20")

def three_of_kind(s):
    from collections import Counter
    return max(Counter(s).values()) >= 3

def all_same(s): return len(set(s)) == 1
def abab(s): return s[0]==s[2] and s[1]==s[3] and s[0]!=s[1]
def aabb(s): return s[0]==s[1] and s[2]==s[3] and s[0]!=s[2]
def abba(s): return s[0]==s[3] and s[1]==s[2] and s[0]!=s[1]
def asc_run(s): return int(s[0])+1==int(s[1]) and int(s[1])+1==int(s[2]) and int(s[2])+1==int(s[3])
def desc_run(s): return int(s[0])-1==int(s[1]) and int(s[1])-1==int(s[2]) and int(s[2])-1==int(s[3])

def lead_zero_count(s):
    # number of leading zeros (0..3)
    c=0
    for ch in s:
        if ch=='0': c+=1
        else: break
    return c

def build_feature_df():
    nums = np.array([f"{i:04d}" for i in range(10000)], dtype=object)
    rows=[]
    for s in nums:
        rows.append({
            "n4": s,
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
            "lead0": int(lead_zero_count(s)),
            "last2": s[-2:],
            "pos0": int(s[0]),
        })
    return pd.DataFrame(rows)

def diversify(df, top, min_last2, min_pos0):
    out=[]
    used_last2={}
    used_pos0={}
    for _,r in df.iterrows():
        l2=r["last2"]; p0=r["pos0"]
        if used_last2.get(l2,0) >= min_last2: 
            continue
        if used_pos0.get(p0,0) >= min_pos0:
            continue
        out.append(r)
        used_last2[l2]=used_last2.get(l2,0)+1
        used_pos0[p0]=used_pos0.get(p0,0)+1
        if len(out) >= top:
            break
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--priors", default="data/processed/crowd_popularity_priors.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-last2", type=int, default=1, help="max per last2")
    ap.add_argument("--min-pos0", type=int, default=3, help="max per first digit")
    ap.add_argument("--w-pattern", type=float, default=1.0)
    ap.add_argument("--w-lead0", type=float, default=0.35)
    ap.add_argument("--w-drawfreq", type=float, default=1.5)
    ap.add_argument("--out", default="data/processed/picks_anticrowd.csv")
    args=ap.parse_args()

    feats = build_feature_df()
    pri = pd.read_csv(args.priors)
    # ensure join key is 4-digit string
    pri["n4"] = pri["n4"].astype(int).map(lambda x: f"{x:04d}")
    df = feats.merge(pri, on="n4", how="left")
    df["draw_freq_sm"] = df["draw_freq_sm"].fillna(df["draw_freq_sm"].median())

    # pattern score: weighted sum of "human-ish" features
    pattern = (
        df["mmdd"] + df["ddmm"] +
        0.7*df["year_19xx"] + 0.7*df["year_20xx"] +
        0.6*df["three_kind"] + 0.8*df["all_same"] +
        0.6*df["abab"] + 0.6*df["aabb"] + 0.6*df["abba"] +
        0.6*df["asc_run"] + 0.6*df["desc_run"]
    )

    # lead0 penalty: keep but discourage 000x spam
    lead0_pen = df["lead0"]

    # history frequency penalty: higher freq => more penalised
    # use log to compress
    freq_pen = np.log(df["draw_freq_sm"].astype(float) + 1e-12)

    # combined score (lower is better)
    df["score"] = (
        args.w_pattern*pattern +
        args.w_lead0*lead0_pen +
        args.w_drawfreq*freq_pen
    )

    # rank then diversify
    ranked = df.sort_values(["score","n4"]).reset_index(drop=True)
    picks = diversify(ranked, args.top, args.min_last2, args.min_pos0)

    picks[["n4","score","draw_count","draw_freq_sm","lead0"]].to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(picks[["n4","score","draw_count","lead0"]].head(args.top).to_string(index=False))

if __name__=="__main__":
    main()
