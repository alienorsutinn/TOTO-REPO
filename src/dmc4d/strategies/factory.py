from __future__ import annotations
from dmc4d.config import CFG
from dmc4d.strategies.low_crowd import LowCrowd
from dmc4d.strategies.random_pick import RandomPick
from dmc4d.strategies.base import Strategy


def create_strategy(name: str, seed: int = 0) -> Strategy:
    n = name.strip().lower()
    if n == "random":
        return RandomPick(seed=seed, bet_type="BIG")
    if n == "low_crowd":
        return LowCrowd(crowd_scores_csv=str(CFG.crowd_scores_csv), bet_type="BIG", seed=seed)
    raise ValueError(f"Unknown strategy: {name}")
