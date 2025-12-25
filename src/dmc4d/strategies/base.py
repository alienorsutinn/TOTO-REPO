from __future__ import annotations
from abc import ABC, abstractmethod
from dmc4d.types import Bet


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_bets(self, budget_rm: int) -> list[Bet]:
        raise NotImplementedError
