from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DigitBiasConfig:
    """Factorized digit-bias model.

    Estimate P(digit | position) from historical winning numbers.
    Score each 4D number by log P(d1|pos1)+...+log P(d4|pos4).

    half_life_days applies exponential recency weighting (0 disables).
    alpha is Dirichlet smoothing to avoid zeros.
    """
    train_window_days: int = 365
    alpha: float = 1.0
    half_life_days: float = 0.0
    bucket_weights: Dict[str, float] | None = None


def _default_bucket_weights() -> Dict[str, float]:
    return {"top3": 0.5, "starter": 1.0, "consolation": 1.0}


def _prep_long(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    if "bucket" not in df.columns:
        raise ValueError("numbers_long must contain 'bucket' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["bucket"] = df["bucket"].astype(str).str.strip().str.lower()
    df["num"] = df["num"].astype(str).str.strip().str.zfill(4)
    df = df[df["num"].str.fullmatch(r"\d{4}", na=False)].copy()
    df["date"] = df["date"].dt.date
    return df


def fit_digit_tables(
    long_df: pd.DataFrame,
    as_of: date,
    cfg: DigitBiasConfig = DigitBiasConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return probs (4,10) and weighted counts (4,10)."""
    df = _prep_long(long_df)
    hist = df[df["date"] < as_of].copy()
    if hist.empty:
        probs = np.full((4, 10), 0.1, dtype=float)
        counts = np.zeros((4, 10), dtype=float)
        return probs, counts

    if cfg.train_window_days > 0:
        min_train = (pd.Timestamp(as_of) - pd.Timedelta(days=int(cfg.train_window_days))).date()
        hist = hist[hist["date"] >= min_train].copy()

    bw = cfg.bucket_weights or _default_bucket_weights()
    hist["bucket_w"] = hist["bucket"].map(bw).fillna(1.0).astype(float)

    if cfg.half_life_days and cfg.half_life_days > 0:
        ages = (pd.Timestamp(as_of) - pd.to_datetime(hist["date"]).astype("datetime64[ns]")).dt.days
        hist["rec_w"] = np.power(0.5, ages.astype(float) / float(cfg.half_life_days))
    else:
        hist["rec_w"] = 1.0

    hist["w"] = hist["bucket_w"] * hist["rec_w"]

    digs = hist["num"].str.extract(r"(?P<d0>\d)(?P<d1>\d)(?P<d2>\d)(?P<d3>\d)")
    for c in digs.columns:
        digs[c] = digs[c].astype(int)
    hist = pd.concat([hist.reset_index(drop=True), digs.reset_index(drop=True)], axis=1)

    counts = np.zeros((4, 10), dtype=float)
    for pos, col in enumerate(["d0", "d1", "d2", "d3"]):
        grp = hist.groupby(col)["w"].sum()
        for digit, v in grp.items():
            counts[pos, int(digit)] = float(v)

    alpha = float(cfg.alpha)
    probs = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + 10.0 * alpha)
    return probs, counts


def score_numbers_digit_bias_long(
    long_df: pd.DataFrame,
    as_of: date,
    cfg: DigitBiasConfig = DigitBiasConfig(),
) -> pd.DataFrame:
    """Score all 0000-9999 numbers under the digit-bias model."""
    probs, _ = fit_digit_tables(long_df, as_of=as_of, cfg=cfg)

    nums = np.arange(10000, dtype=int)
    d0 = nums // 1000
    d1 = (nums // 100) % 10
    d2 = (nums // 10) % 10
    d3 = nums % 10

    score = (
        np.log(probs[0, d0])
        + np.log(probs[1, d1])
        + np.log(probs[2, d2])
        + np.log(probs[3, d3])
    )

    out = pd.DataFrame({"num": [f"{n:04d}" for n in nums.tolist()], "score": score.astype(float)})
    return out.sort_values("score", ascending=False).reset_index(drop=True)
