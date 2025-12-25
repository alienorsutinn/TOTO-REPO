from __future__ import annotations
import pandas as pd

def longest_losing_streak(hit: pd.Series) -> int:
    longest = 0
    cur = 0
    for h in hit.tolist():
        if int(h) == 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)

def summarize(per_draw: pd.DataFrame) -> dict:
    stake = float(per_draw["stake_rm"].sum())
    payout = float(per_draw["payout_rm"].sum())
    profit = float(per_draw["profit_rm"].sum())
    roi = payout / stake if stake else 0.0
    hit_rate = float((per_draw["hit"] > 0).mean())
    return {
        "draws": int(len(per_draw)),
        "stake_total_rm": stake,
        "payout_total_rm": payout,
        "profit_total_rm": profit,
        "roi": float(roi),
        "hit_rate_any_prize": hit_rate,
        "longest_losing_streak": longest_losing_streak(per_draw["hit"]),
    }
