from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from dmc4d.crowd.features import birthday_signals, pattern_signals
from dmc4d.crowd.weights.default import DEFAULT_WEIGHTS
from dmc4d.utils.strings import z4


@dataclass(frozen=True)
class BirthMaps:
    p_monthday: dict[str, float]
    p_year: dict[int, float]


def load_birth_maps(births_dir: Path) -> BirthMaps:
    md = pd.read_csv(births_dir / "births_by_monthday.csv")
    yy = pd.read_csv(births_dir / "births_by_year.csv")
    return BirthMaps(
        p_monthday=dict(zip(md["monthday"], md["p"])),
        p_year=dict(zip(yy["year"], yy["p"])),
    )


def crowd_score(num: str, maps: BirthMaps, weights: dict[str, float] | None = None) -> float:
    w = DEFAULT_WEIGHTS if weights is None else weights
    n = z4(num)
    b = birthday_signals(n)
    p_ddmm = maps.p_monthday.get(b.ddmm_md, 0.0) if b.ddmm_md else 0.0
    p_mmdd = maps.p_monthday.get(b.mmdd_md, 0.0) if b.mmdd_md else 0.0
    p_dmyy = (
        (maps.p_monthday.get(b.dmyy_md, 0.0) * maps.p_year.get(b.year, 0.0)) if b.dmyy_md else 0.0
    )
    p_mdyy = (
        (maps.p_monthday.get(b.mdyy_md, 0.0) * maps.p_year.get(b.year, 0.0)) if b.mdyy_md else 0.0
    )
    p_yy = maps.p_year.get(b.year, 0.0)

    p = pattern_signals(n)

    score = 0.0
    score += w["ddmm"] * p_ddmm
    score += w["mmdd"] * p_mmdd
    score += w["dmyy"] * p_dmyy
    score += w["mdyy"] * p_mdyy
    score += w["yy_suffix"] * p_yy
    score += w["repeats"] * p.repeats
    score += w["all_same"] * p.all_same
    score += w["mirror"] * p.mirror
    score += w["lucky_cluster"] * p.lucky_cluster
    score += w["unlucky_4"] * p.unlucky_4
    return float(score)


def score_all_numbers(births_dir: Path, out_path: Path) -> Path:
    maps = load_birth_maps(births_dir)
    rows = [
        {"number": f"{i:04d}", "crowd_score": crowd_score(f"{i:04d}", maps)} for i in range(10000)
    ]
    df = pd.DataFrame(rows).sort_values("crowd_score", ascending=True).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def explain_number(num: str, births_dir: Path) -> dict:
    maps = load_birth_maps(births_dir)
    n = z4(num)
    return {
        "number": n,
        "birthday_signals": birthday_signals(n).__dict__,
        "pattern_signals": pattern_signals(n).__dict__,
        "crowd_score": crowd_score(n, maps),
    }
