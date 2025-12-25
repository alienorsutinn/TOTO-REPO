from __future__ import annotations

import argparse
import random
from datetime import date

import pandas as pd

from dmc4d.backtest.walkforward import WalkForwardConfig, run_walkforward_backtest, summarize_backtest
from dmc4d.scoring.digit_bias import DigitBiasConfig
from dmc4d.scoring.model import ScoreConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numbers-long", default="data/processed/numbers_long.csv")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--train-min-days", type=int, default=60)
    p.add_argument("--train-window-days", type=int, default=365, help="0 = all history")
    p.add_argument("--recent-draws-k", type=int, default=5)
    p.add_argument("--recent-draw-penalty", type=float, default=1.0)
    p.add_argument("--out", default="data/processed/backtest_walkforward.csv")
    p.add_argument("--random-trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--model", choices=["heuristic", "digit_bias"], default="heuristic")
    p.add_argument("--digit-alpha", type=float, default=1.0)
    p.add_argument("--digit-half-life-days", type=float, default=0.0)
    return p.parse_args()


def _parse_winners_to_int_set(winners_csv: str) -> set[int]:
    if not isinstance(winners_csv, str) or not winners_csv.strip():
        return set()
    out: set[int] = set()
    for x in winners_csv.split(","):
        x = x.strip()
        if x and x.isdigit():
            out.add(int(x))
    return out


def random_baseline(bt: pd.DataFrame, top_n: int, trials: int, seed: int) -> dict:
    if trials <= 0 or bt.empty:
        return {}
    if "winners" not in bt.columns:
        raise ValueError("Backtest output missing 'winners' column.")

    rng = random.Random(seed)
    n_days = len(bt)
    winners_sets = [_parse_winners_to_int_set(x) for x in bt["winners"].tolist()]

    any_hits: list[float] = []
    avg_hits: list[float] = []
    p_ge2: list[float] = []
    p_ge3: list[float] = []

    for _ in range(trials):
        trial_any = 0
        trial_hits = 0
        trial_ge2 = 0
        trial_ge3 = 0

        for wset in winners_sets:
            picks = set(rng.sample(range(10000), k=top_n))
            h = len(picks & wset)
            trial_hits += h
            if h > 0:
                trial_any += 1
            if h >= 2:
                trial_ge2 += 1
            if h >= 3:
                trial_ge3 += 1

        any_hits.append(trial_any / n_days)
        avg_hits.append(trial_hits / n_days)
        p_ge2.append(trial_ge2 / n_days)
        p_ge3.append(trial_ge3 / n_days)

    any_hits.sort()
    avg_hits.sort()
    p_ge2.sort()
    p_ge3.sort()

    def q(xs: list[float], quant: float) -> float:
        idx = int(round((len(xs) - 1) * quant))
        return float(xs[max(0, min(idx, len(xs) - 1))])

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs))

    return {
        "rand_any_hit_rate_mean": mean(any_hits),
        "rand_any_hit_rate_p10": q(any_hits, 0.10),
        "rand_any_hit_rate_p50": q(any_hits, 0.50),
        "rand_any_hit_rate_p90": q(any_hits, 0.90),
        "rand_avg_hits_mean": mean(avg_hits),
        "rand_avg_hits_p10": q(avg_hits, 0.10),
        "rand_avg_hits_p50": q(avg_hits, 0.50),
        "rand_avg_hits_p90": q(avg_hits, 0.90),
        "rand_p_hits_ge2_mean": mean(p_ge2),
        "rand_p_hits_ge2_p50": q(p_ge2, 0.50),
        "rand_p_hits_ge2_p90": q(p_ge2, 0.90),
        "rand_p_hits_ge3_mean": mean(p_ge3),
        "rand_p_hits_ge3_p50": q(p_ge3, 0.50),
        "rand_p_hits_ge3_p90": q(p_ge3, 0.90),
    }


def pick_diversity_report(bt: pd.DataFrame) -> dict:
    if bt.empty or "picks" not in bt.columns:
        return {}

    all_picks: list[str] = []
    per_day_unique: list[int] = []
    for s in bt["picks"].astype(str).tolist():
        picks = [x.strip() for x in s.split(",") if x.strip()]
        per_day_unique.append(len(set(picks)))
        all_picks.extend(picks)

    uniq_total = len(set(all_picks))
    total = len(all_picks)
    return {
        "picks_total": total,
        "picks_unique_total": uniq_total,
        "picks_unique_share": (uniq_total / float(total)) if total else 0.0,
        "picks_unique_per_day_min": int(min(per_day_unique)) if per_day_unique else 0,
        "picks_unique_per_day_mean": float(sum(per_day_unique) / len(per_day_unique)) if per_day_unique else 0.0,
    }



    def p_values_vs_random(bt: pd.DataFrame, top_n: int, trials: int, seed: int) -> dict:
        """Compute p-values: fraction of random trials >= observed metrics."""
        if trials <= 0 or bt.empty:
            return {}
        rng = random.Random(seed)
        n_days = len(bt)
        winners_sets = [_parse_winners_to_int_set(x) for x in bt["winners"].tolist()]

        obs_any = float(bt["any_hit"].mean())
        obs_avg_hits = float(bt["hits_count"].mean())
        obs_ge2 = float((bt["hits_count"] >= 2).mean())

        ge_any = 0
        ge_avg = 0
        ge_ge2 = 0

        for _ in range(trials):
            trial_any = 0
            trial_hits = 0
            trial_ge2 = 0
            for wset in winners_sets:
                picks = set(rng.sample(range(10000), k=top_n))
                h = len(picks & wset)
                trial_hits += h
                if h > 0:
                    trial_any += 1
                if h >= 2:
                    trial_ge2 += 1

            any_rate = trial_any / n_days
            avg_hits = trial_hits / n_days
            ge2_rate = trial_ge2 / n_days

            if any_rate >= obs_any:
                ge_any += 1
            if avg_hits >= obs_avg_hits:
                ge_avg += 1
            if ge2_rate >= obs_ge2:
                ge_ge2 += 1

        return {
            "p_any_hit_rate": (ge_any / trials),
            "p_avg_hits_per_day": (ge_avg / trials),
            "p_p_hits_ge2": (ge_ge2 / trials),
            "trials": trials,
        }



def p_values_vs_random_global(bt: pd.DataFrame, top_n: int, trials: int, seed: int) -> dict:
    """Compute p-values: fraction of random trials >= observed metrics."""
    if trials <= 0 or bt.empty:
        return {}

    rng = random.Random(seed)
    n_days = len(bt)
    winners_sets = [_parse_winners_to_int_set(x) for x in bt["winners"].tolist()]

    obs_any = float(bt["any_hit"].mean())
    obs_avg_hits = float(bt["hits_count"].mean())
    obs_ge2 = float((bt["hits_count"] >= 2).mean())

    ge_any = 0
    ge_avg = 0
    ge_ge2 = 0

    for _ in range(trials):
        trial_any = 0
        trial_hits = 0
        trial_ge2 = 0

        for wset in winners_sets:
            picks = set(rng.sample(range(10000), k=top_n))
            h = len(picks & wset)
            trial_hits += h
            if h > 0:
                trial_any += 1
            if h >= 2:
                trial_ge2 += 1

        any_rate = trial_any / n_days
        avg_hits = trial_hits / n_days
        ge2_rate = trial_ge2 / n_days

        if any_rate >= obs_any:
            ge_any += 1
        if avg_hits >= obs_avg_hits:
            ge_avg += 1
        if ge2_rate >= obs_ge2:
            ge_ge2 += 1

    return {
        "p_any_hit_rate": (ge_any / trials),
        "p_avg_hits_per_day": (ge_avg / trials),
        "p_p_hits_ge2": (ge_ge2 / trials),
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    long = pd.read_csv(args.numbers_long)

    wf_cfg = WalkForwardConfig(train_min_days=args.train_min_days, pick_top_n=args.top)

    score_cfg = ScoreConfig(
        train_window_days=args.train_window_days,
        recent_draws_k=args.recent_draws_k,
        recent_draw_penalty=args.recent_draw_penalty,
    )

    digit_cfg = DigitBiasConfig(
        train_window_days=args.train_window_days,
        alpha=args.digit_alpha,
        half_life_days=args.digit_half_life_days,
    )

    bt = run_walkforward_backtest(
        long_df=long,
        start=start,
        end=end,
        score_cfg=score_cfg,
        wf_cfg=wf_cfg,
        model=args.model,
        digit_cfg=digit_cfg,
    )
    bt.to_csv(args.out, index=False)

    summary = summarize_backtest(bt)
    print("Wrote:", args.out)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    rb = random_baseline(bt, top_n=args.top, trials=args.random_trials, seed=args.seed)
    if rb:
        pv = p_values_vs_random_global(bt, top_n=args.top, trials=args.random_trials, seed=args.seed + 1337)
        if pv:
            print('\nP-values vs random (>= observed):')
            for k, v in pv.items():
                print(f'  {k}: {v}')
        print("\nRandom baseline (Monte Carlo, actual winners):")
        for k, v in rb.items():
            print(f"  {k}: {v}")

    div = pick_diversity_report(bt)
    if div:
        print("\nPick diversity:")
        for k, v in div.items():
            print(f"  {k}: {v}")

    print("\nLast 10 days:")
    cols = ["date", "n_picks", "winners", "n_winners", "hits_count", "any_hit", "hit_rate", "picks", "hits"]
    cols = [c for c in cols if c in bt.columns]
    print(bt[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
