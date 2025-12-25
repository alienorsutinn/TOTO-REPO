from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date

import pandas as pd

from dmc4d.scoring.model import ScoreConfig, score_numbers_long


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numbers-long", default="data/processed/numbers_long.csv")
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--train-window-days", type=int, default=365)
    p.add_argument("--cooldown-days", type=int, default=10)

    # diversification controls
    p.add_argument("--max-per-prefix2", type=int, default=2)
    p.add_argument("--max-per-suffix2", type=int, default=2)

    p.add_argument("--out", default="data/processed/top_picks.csv")
    return p.parse_args()


def diversified_top_picks(
    scored: pd.DataFrame,
    top_n: int,
    max_per_prefix2: int = 2,
    max_per_suffix2: int = 2,
) -> pd.DataFrame:
    if scored.empty:
        return scored

    prefix_ct = defaultdict(int)
    suffix_ct = defaultdict(int)

    chosen_rows = []
    for _, r in scored.iterrows():
        n = str(r["num"]).zfill(4)
        p2 = n[:2]
        s2 = n[-2:]

        if prefix_ct[p2] >= max_per_prefix2:
            continue
        if suffix_ct[s2] >= max_per_suffix2:
            continue

        chosen_rows.append(r)
        prefix_ct[p2] += 1
        suffix_ct[s2] += 1

        if len(chosen_rows) >= top_n:
            break

    if not chosen_rows:
        return scored.head(top_n).copy()

    out = pd.DataFrame(chosen_rows).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    as_of = date.fromisoformat(args.as_of)

    long = pd.read_csv(args.numbers_long)

    cfg = ScoreConfig(
        train_window_days=args.train_window_days,
        cooldown_days=args.cooldown_days,
    )
    scored = score_numbers_long(long, as_of=as_of, cfg=cfg)

    picks = diversified_top_picks(
        scored,
        top_n=args.top,
        max_per_prefix2=args.max_per_prefix2,
        max_per_suffix2=args.max_per_suffix2,
    )

    picks.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    print(picks.to_string(index=False))


if __name__ == "__main__":
    main()
