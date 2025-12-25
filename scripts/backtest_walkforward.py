from __future__ import annotations

import argparse
import random
from datetime import date

import pandas as pd

from dmc4d.backtest.walkforward import (
    WalkForwardConfig,
    run_walkforward_backtest,
    summarize_backtest,
)
from dmc4d.scoring.model import ScoreConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numbers-long", default="data/processed/numbers_long.csv")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--train-min-days", type=int, default=60)
    p.add_argument("--train-window-days", type=int, default=365, help="Use only last N days of history for scoring (0 = all)")
    p.add_argument("--recent-draws-k", type=int, default=5, help="Penalize numbers seen in the last K draws")
    p.add_argument("--recent-draw-penalty", type=float, default=1.0, help="Penalty strength for recent-draw numbers")
    p.add_argument("--out", default="data/processed/backtest_walkforward.csv")
    p.add_argument("--random-trials", type=int, default=200, help="Monte Carlo trials for random baseline")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def random_baseline(bt: pd.DataFrame, top_n: int, trials: int, seed: int) -> dict:
    """
    Compare vs random picks in the same setting:
      - each day has n_winners winners in a 10,000 universe
      - sample top_n without replacement
    """
    if trials <= 0:
        return {}

    rng = random.Random(seed)
    n_days = len(bt)
    if n_days == 0:
        return {}

    winners_per_day = bt["n_winners"].astype(int).tolist()

    any_hits = []
    avg_hits = []

    for _ in range(trials):
        trial_any = 0
        trial_hits = 0

        for w in winners_per_day:
            picks = set(rng.sample(range(10000), k=top_n))
            winners = set(rng.sample(range(10000), k=w))
            h = len(picks & winners)
            trial_hits += h
            if h > 0:
                trial_any += 1

        any_hits.append(trial_any / n_days)
        avg_hits.append(trial_hits / n_days)

    any_hits.sort()
    avg_hits.sort()

    def q(xs, quant):
        idx = int(round((len(xs) - 1) * quant))
        return float(xs[max(0, min(idx, len(xs) - 1))])

    return {
        "rand_any_hit_rate_mean": float(sum(any_hits) / len(any_hits)),
        "rand_any_hit_rate_p10": q(any_hits, 0.10),
        "rand_any_hit_rate_p50": q(any_hits, 0.50),
        "rand_any_hit_rate_p90": q(any_hits, 0.90),
        "rand_avg_hits_mean": float(sum(avg_hits) / len(avg_hits)),
        "rand_avg_hits_p10": q(avg_hits, 0.10),
        "rand_avg_hits_p50": q(avg_hits, 0.50),
        "rand_avg_hits_p90": q(avg_hits, 0.90),
    }



def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    long = pd.read_csv(args.numbers_long)

    wf_cfg = WalkForwardConfig(
        train_min_days=args.train_min_days,
        pick_top_n=args.top,
    )

    score_cfg = ScoreConfig(
        train_window_days=args.train_window_days,
        recent_draws_k=args.recent_draws_k,
        recent_draw_penalty=args.recent_draw_penalty,
    )

    bt = run_walkforward_backtest(
        long_df=long,
        start=start,
        end=end,
        score_cfg=score_cfg,
        wf_cfg=wf_cfg,
    )
    bt.to_csv(args.out, index=False)

    summary = summarize_backtest(bt)
    print("Wrote:", args.out)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    rb = random_baseline(bt, top_n=args.top, trials=args.random_trials, seed=args.seed)
    if rb:
        print("\nRandom baseline (Monte Carlo):")
        for k, v in rb.items():
            print(f"  {k}: {v}")


    print("\nLast 10 days:")
    print(bt.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
