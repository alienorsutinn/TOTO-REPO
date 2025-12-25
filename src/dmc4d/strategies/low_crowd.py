from __future__ import annotations
import pandas as pd
from dmc4d.strategies.base import Strategy
from dmc4d.types import Bet

class LowCrowd(Strategy):
    name = "low_crowd"

    def __init__(self, crowd_scores_csv: str, bet_type: str = "BIG", pool_size: int = 1500, seed: int = 0):
        self.crowd_scores_csv = crowd_scores_csv
        self.bet_type = bet_type
        self.pool_size = pool_size
        self.seed = seed

    def generate_bets(self, budget_rm: int) -> list[Bet]:
        n = max(0, int(budget_rm))
        df = pd.read_csv(self.crowd_scores_csv)
        pool = df.head(self.pool_size).sample(n=min(n, self.pool_size), random_state=self.seed)
        return [Bet(number=str(x).zfill(4), bet_type=self.bet_type, stake_rm=1) for x in pool["number"].tolist()]
