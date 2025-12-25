from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from dmc4d.datasets.results_store import read_results
from dmc4d.datasets.schemas import validate_draw_lists
from dmc4d.pricing.payouts import BIG_PAYOUT, SMALL_PAYOUT
from dmc4d.strategies.base import Strategy
from dmc4d.types import Bet
from dmc4d.utils.strings import z4


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    budget_rm: int


def _payout_for_bet(bet: Bet, top3: list[str], starter: set[str], consolation: set[str]) -> int:
    n = z4(bet.number)
    if bet.bet_type == "BIG":
        if n == top3[0]:
            return BIG_PAYOUT["1st"]
        if n == top3[1]:
            return BIG_PAYOUT["2nd"]
        if n == top3[2]:
            return BIG_PAYOUT["3rd"]
        if n in starter:
            return BIG_PAYOUT["starter"]
        if n in consolation:
            return BIG_PAYOUT["consolation"]
        return 0
    if n == top3[0]:
        return SMALL_PAYOUT["1st"]
    if n == top3[1]:
        return SMALL_PAYOUT["2nd"]
    if n == top3[2]:
        return SMALL_PAYOUT["3rd"]
    return 0


def run_backtest(results_csv: Path, strategy: Strategy, cfg: BacktestConfig) -> pd.DataFrame:
    df = read_results(results_csv)
    df = df[(df["date"] >= cfg.start) & (df["date"] <= cfg.end)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No results in date range")

    rows = []
    for _, row in df.iterrows():
        top3 = row["top3"]
        starter = set(row["starter"])
        consolation = set(row["consolation"])
        validate_draw_lists(top3, row["starter"], row["consolation"])

        bets = strategy.generate_bets(cfg.budget_rm)
        seen = set()
        for b in bets:
            if b.stake_rm != 1:
                raise ValueError("Stake must be 1")
            key = (z4(b.number), b.bet_type)
            if key in seen:
                raise ValueError(f"Duplicate bet: {key}")
            seen.add(key)

        stake = sum(b.stake_rm for b in bets)
        payout = sum(_payout_for_bet(b, top3, starter, consolation) for b in bets)
        rows.append(
            {
                "date": row["date"],
                "draw_no": row["draw_no"],
                "operator": row["operator"],
                "stake_rm": stake,
                "payout_rm": payout,
                "profit_rm": payout - stake,
                "hit": int(payout > 0),
            }
        )
    return pd.DataFrame(rows)
