from __future__ import annotations
from dmc4d.utils.strings import z4

def validate_draw_lists(top3: list[str], starter: list[str], consolation: list[str]) -> None:
    if len(top3) != 3:
        raise ValueError(f"top3 must be len=3, got {len(top3)}")
    if len(starter) != 10:
        raise ValueError(f"starter must be len=10, got {len(starter)}")
    if len(consolation) != 10:
        raise ValueError(f"consolation must be len=10, got {len(consolation)}")
    for group in (top3, starter, consolation):
        for x in group:
            _ = z4(x)
