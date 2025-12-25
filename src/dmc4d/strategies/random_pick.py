from __future__ import annotations
import random
from dmc4d.strategies.base import Strategy
from dmc4d.types import Bet

class RandomPick(Strategy):
    name = "random"

    def __init__(self, seed: int | None = None, bet_type: str = "BIG") -> None:
        self.rng = random.Random(seed)
        self.bet_type = bet_type

    def generate_bets(self, budget_rm: int) -> list[Bet]:
        n = max(0, int(budget_rm))
        nums = self.rng.sample(range(10000), k=min(n, 10000))
        return [Bet(number=f"{x:04d}", bet_type=self.bet_type, stake_rm=1) for x in nums]
