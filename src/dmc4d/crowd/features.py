from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from dmc4d.utils.strings import z4

def _safe_md(mm: int, dd: int) -> str | None:
    if 1 <= mm <= 12 and 1 <= dd <= 31:
        return f"{mm:02d}-{dd:02d}"
    return None

def _yy_to_year(yy: int) -> int:
    return 2000 + yy if 0 <= yy <= 25 else 1900 + yy

@dataclass(frozen=True)
class PatternSignals:
    repeats: float
    all_same: float
    mirror: float
    lucky_cluster: float
    unlucky_4: float

def pattern_signals(num: str) -> PatternSignals:
    s = z4(num)
    cts = Counter(s)
    maxrep = max(cts.values())
    repeats = float(max(0, maxrep - 1))
    all_same = 1.0 if len(cts) == 1 else 0.0
    mirror = 1.0 if (s[0] == s[3] and s[1] == s[2]) else 0.0
    lucky_count = sum(ch in "68" for ch in s)
    lucky_cluster = float(max(0, lucky_count - 1))
    unlucky_4 = float(s.count("4"))
    return PatternSignals(repeats, all_same, mirror, lucky_cluster, unlucky_4)

@dataclass(frozen=True)
class BirthdaySignals:
    ddmm_md: str | None
    mmdd_md: str | None
    dmyy_md: str | None
    mdyy_md: str | None
    year: int

def birthday_signals(num: str) -> BirthdaySignals:
    s = z4(num)
    dd = int(s[:2]); mm = int(s[2:])
    ddmm_md = _safe_md(mm=mm, dd=dd)
    mm2 = int(s[:2]); dd2 = int(s[2:])
    mmdd_md = _safe_md(mm=mm2, dd=dd2)

    a = int(s[0]); b = int(s[1]); yy = int(s[2:])
    dmyy_md = _safe_md(mm=b, dd=a)
    mdyy_md = _safe_md(mm=a, dd=b)
    year = _yy_to_year(yy)
    return BirthdaySignals(ddmm_md, mmdd_md, dmyy_md, mdyy_md, year)
