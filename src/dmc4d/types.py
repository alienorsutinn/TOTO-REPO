from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

BetType = Literal["BIG", "SMALL"]


@dataclass(frozen=True)
class Bet:
    number: str
    bet_type: BetType
    stake_rm: int
