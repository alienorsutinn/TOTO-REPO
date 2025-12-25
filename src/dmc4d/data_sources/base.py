from __future__ import annotations
from abc import ABC, abstractmethod


class ResultsSource(ABC):
    @abstractmethod
    def load(self) -> list[dict]:
        raise NotImplementedError


def parse_list_field(s: str, expected: int) -> list[str]:
    xs = [x.strip().zfill(4) for x in str(s).split(",") if x.strip()]
    if len(xs) != expected:
        raise ValueError(f"Expected {expected} nums, got {len(xs)}: {xs}")
    return xs
