from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class ScoreConfig:
    """
    Scoring config for ranking 4D numbers.

    Notes:
    - train_window_days controls how far back we look for frequencies.
    - cooldown_days penalizes very recent occurrences (days-based).
    - recent_draws_k penalizes if number appeared in the last K draws (draw-based).
    - bucket_weights allows weighting top3 vs starter vs consolation when counting frequency.

    The goal is not to "predict lottery" (impossible), but to create a stable,
    testable ranking rule and see if it beats random under walk-forward.
    """

    train_window_days: int = 365
    cooldown_days: int = 10

    # If a number appeared in the last K draws (by date), penalize it.
    recent_draws_k: int = 5
    recent_draw_penalty: float = 1.0  # subtract this if seen in recent K draws

    # Core weights
    w_freq: float = 1.0
    w_overdue: float = 1.0
    cooldown_penalty: float = 0.5  # multiply by cooldown_flag (0/1)

    # Frequency weighting by bucket
    bucket_weights: Dict[str, float] = None  # set default in __post_init__-like helper


def _default_bucket_weights() -> Dict[str, float]:
    # If your objective is "hit any winner", starter/consolation matter more.
    return {"top3": 0.5, "starter": 1.0, "consolation": 1.0}


def _ensure_cfg(cfg: ScoreConfig) -> Tuple[ScoreConfig, Dict[str, float]]:
    bw = cfg.bucket_weights or _default_bucket_weights()
    return cfg, bw


def _prep_long(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["num"] = df["num"].astype(str).str.strip().str.zfill(4)
    if "bucket" not in df.columns:
        raise ValueError("numbers_long must contain 'bucket' column")
    return df


def _recent_draw_set(long_df: pd.DataFrame, as_of: date, k: int) -> set[str]:
    """
    Returns set of nums that appeared in the most recent k draw dates strictly before as_of.
    """
    if k <= 0:
        return set()

    df = long_df[long_df["date"] < as_of]
    if df.empty:
        return set()

    draw_dates = sorted(df["date"].unique())
    if not draw_dates:
        return set()

    recent_dates = set(draw_dates[-k:])
    recent_nums = set(df[df["date"].isin(recent_dates)]["num"].astype(str).tolist())
    return recent_nums


def score_numbers_long(
    long_df: pd.DataFrame, as_of: date, cfg: ScoreConfig = ScoreConfig()
) -> pd.DataFrame:
    """
    Produce a scored table: one row per number, with features + final score.
    """
    cfg, bucket_w = _ensure_cfg(cfg)
    df = _prep_long(long_df)

    # restrict to history strictly before as_of
    hist = df[df["date"] < as_of].copy()
    if hist.empty:
        return pd.DataFrame(
            columns=[
                "num",
                "score",
                "freq_norm",
                "freq_bucket_weighted",
                "last_seen",
                "days_since_seen",
                "overdue_boost",
                "cooldown_penalty_term",
                "recent_draw_penalty_term",
            ]
        )

    # training window
    min_train_date = as_of - timedelta(days=int(cfg.train_window_days))
    train = hist[hist["date"] >= min_train_date].copy()

    # --- Frequency (bucket-weighted) ---
    # count occurrences by num and bucket
    train["bucket_w"] = train["bucket"].map(bucket_w).fillna(1.0).astype(float)
    # bucket-weighted "frequency mass"
    freq_mass = train.groupby("num")["bucket_w"].sum()

    # normalize to [0,1] by dividing by max
    if len(freq_mass) > 0:
        freq_norm = (freq_mass / float(freq_mass.max())).fillna(0.0)
    else:
        freq_norm = pd.Series(dtype=float)

    # --- Last seen / overdue ---
    last_seen = hist.groupby("num")["date"].max()
    days_since = (pd.Series({n: (as_of - d).days for n, d in last_seen.items()})).astype(float)

    # overdue boost: scale to [0,1] using train_window_days (cap)
    overdue_boost = (days_since / float(max(cfg.train_window_days, 1))).clip(0, 1)

    # --- Cooldown penalty (days-based) ---
    cooldown_flag = (days_since <= float(cfg.cooldown_days)).astype(float)
    cooldown_penalty_term = cfg.cooldown_penalty * cooldown_flag

    # --- Recent draw penalty (draw-based) ---
    recent_nums = _recent_draw_set(hist, as_of=as_of, k=int(cfg.recent_draws_k))
    recent_draw_penalty_term = pd.Series(
        {n: (cfg.recent_draw_penalty if n in recent_nums else 0.0) for n in last_seen.index},
        dtype=float,
    )

    # Build output frame aligned on all known nums
    out = pd.DataFrame({"num": sorted(last_seen.index)})
    out["freq_norm"] = out["num"].map(freq_norm).fillna(0.0)
    out["last_seen"] = out["num"].map(last_seen)
    out["days_since_seen"] = out["num"].map(days_since).fillna(9999.0)
    out["overdue_boost"] = out["num"].map(overdue_boost).fillna(0.0)
    out["cooldown_penalty_term"] = out["num"].map(cooldown_penalty_term).fillna(0.0)
    out["recent_draw_penalty_term"] = out["num"].map(recent_draw_penalty_term).fillna(0.0)

    # Final score
    out["score"] = (
        cfg.w_freq * out["freq_norm"]
        + cfg.w_overdue * out["overdue_boost"]
        - out["cooldown_penalty_term"]
        - out["recent_draw_penalty_term"]
    )

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def top_picks(scored: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if scored.empty:
        return scored
    return scored.head(int(top_n)).copy()


def score_number(
    long_df: pd.DataFrame, as_of: date, num: str, cfg: ScoreConfig = ScoreConfig()
) -> pd.Series:
    scored = score_numbers_long(long_df, as_of=as_of, cfg=cfg)
    num = str(num).strip().zfill(4)
    hit = scored[scored["num"] == num]
    if hit.empty:
        return pd.Series(dtype=object)
    return hit.iloc[0]


def explain_score(scored_row: pd.Series, cfg: ScoreConfig = ScoreConfig()) -> dict:
    """
    Small explain helper for a single scored row (from score_numbers_long()).
    """
    cfg, _ = _ensure_cfg(cfg)
    freq = float(scored_row.get("freq_norm", 0.0))
    overdue = float(scored_row.get("overdue_boost", 0.0))
    cooldown = float(scored_row.get("cooldown_penalty_term", 0.0))
    recent = float(scored_row.get("recent_draw_penalty_term", 0.0))

    return {
        "term_freq": cfg.w_freq * freq,
        "term_overdue": cfg.w_overdue * overdue,
        "term_cooldown_penalty": -cooldown,
        "term_recent_draw_penalty": -recent,
        "score": float(scored_row.get("score", 0.0)),
    }
