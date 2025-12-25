from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from dmc4d.scoring.model import ScoreConfig, score_numbers_long, top_picks


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Walk-forward backtest configuration.

    - train_min_days: require at least this many days of history before scoring.
    - pick_top_n: how many numbers to pick each day.
    - buckets: which buckets define a "hit universe" for a given draw date.
    """

    train_min_days: int = 60
    pick_top_n: int = 20
    buckets: tuple[str, ...] = ("top3", "starter", "consolation")


def _ensure_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def _normalize_long(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    needed = {"date", "bucket", "num"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"numbers_long is missing columns: {sorted(missing)}")

    df["date"] = _ensure_datetime(df["date"])
    df = df[df["date"].notna()].copy()

    df["bucket"] = df["bucket"].astype(str).str.strip().str.lower()
    df["num"] = df["num"].astype(str).str.strip().str.zfill(4)

    # Keep only clean 4-digit numbers
    df = df[df["num"].str.fullmatch(r"\d{4}", na=False)].copy()
    return df


def build_draw_universe(long_df: pd.DataFrame, buckets: Iterable[str]) -> pd.DataFrame:
    """
    Builds a per-date set of winning numbers for the chosen buckets.
    Returns a DataFrame: date, winners_set (python set)
    """
    df = _normalize_long(long_df)
    buckets = {b.strip().lower() for b in buckets}

    df = df[df["bucket"].isin(buckets)].copy()

    # group by date -> set of numbers
    grp = df.groupby("date")["num"].apply(lambda s: set(s.tolist())).reset_index()
    grp = grp.rename(columns={"num": "winners_set"})
    grp = grp.sort_values("date").reset_index(drop=True)
    return grp


def run_walkforward_backtest(
    long_df: pd.DataFrame,
    start: date,
    end: date,
    score_cfg: ScoreConfig = ScoreConfig(),
    wf_cfg: WalkForwardConfig = WalkForwardConfig(),
) -> pd.DataFrame:
    """
    Walk-forward:
      For each draw date D in [start, end]:
        - train on data <= D-1 (i.e. strictly before the day)
        - score numbers
        - pick top N
        - compare against winners on D
    """
    df = _normalize_long(long_df)

    # Build winners per day
    winners = build_draw_universe(df, wf_cfg.buckets)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    winners = winners[(winners["date"] >= start_ts) & (winners["date"] <= end_ts)].copy()
    winners = winners.sort_values("date").reset_index(drop=True)

    if winners.empty:
        raise ValueError("No draw dates in the requested backtest window.")

    # Determine earliest date with enough history
    min_date = df["date"].min()
    rows = []

    for _, row in winners.iterrows():
        d: pd.Timestamp = row["date"]
        win_set: set[str] = row["winners_set"]

        # Require enough prior history
        days_hist = (d - min_date).days
        if days_hist < wf_cfg.train_min_days:
            continue

        # Train data strictly before d
        train = df[df["date"] < d].copy()
        if train.empty:
            continue

        scored = score_numbers_long(train, as_of=d.date(), cfg=score_cfg)
        picks_df = top_picks(scored, top_n=wf_cfg.pick_top_n)
        picks = set(picks_df["num"].astype(str).tolist())

        hits = picks & win_set
        rows.append(
            {
                "date": d.date(),
                "n_picks": wf_cfg.pick_top_n,
                "n_winners": len(win_set),
                "hits_count": len(hits),
                "any_hit": int(len(hits) > 0),
                "hit_rate": (len(hits) / float(wf_cfg.pick_top_n)) if wf_cfg.pick_top_n else 0.0,
                "picks": ",".join(sorted(picks)),
                "hits": ",".join(sorted(hits)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Backtest produced no rows. Increase date range or reduce train_min_days.")

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def summarize_backtest(bt: pd.DataFrame) -> dict:
    """
    Summary metrics:
      - days_tested
      - any_hit_rate (share of days with >=1 hit)
      - avg_hits_per_day
      - avg_hit_rate_per_pick
      - p50/p90 hits_count
    """
    if bt.empty:
        return {}

    days = len(bt)
    any_hit_rate = float(bt["any_hit"].mean())
    avg_hits = float(bt["hits_count"].mean())
    avg_hit_rate_pick = float(bt["hit_rate"].mean())

    p50 = float(bt["hits_count"].quantile(0.50))
    p90 = float(bt["hits_count"].quantile(0.90))

    return {
        "days_tested": days,
        "any_hit_rate": any_hit_rate,
        "avg_hits_per_day": avg_hits,
        "avg_hit_rate_per_pick": avg_hit_rate_pick,
        "hits_count_p50": p50,
        "hits_count_p90": p90,
    }
