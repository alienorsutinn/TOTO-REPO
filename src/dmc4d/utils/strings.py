from __future__ import annotations
def z4(x: str | int) -> str:
    s = str(x).strip()
    if not s.isdigit():
        raise ValueError(f"Not numeric: {x}")
    return s.zfill(4)
