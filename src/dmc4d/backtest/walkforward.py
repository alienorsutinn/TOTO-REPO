from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from dmc4d.scoring.digit_bias import DigitBiasConfig, score_numbers_digit_bias_long
from dmc4d.scoring.model import ScoreConfig, score_numbers_long, top_picks


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward backtest configuration."""
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
    df = df[df["num"].str.fullmatch(r"\d{4}", na=False)].copy()
    return df


def build_draw_universe(long_df: pd.DataFrame, buckets: Iterable[str]) -> pd.DataFrame:
    """Per-date set of winners for chosen buckets."""
    df = _normalize_long(long_df)
    buckets = {b.strip().lower() for b in buckets}
    df = df[df["bucket"].isin(buckets)].copy()

    grp = df.groupby("date")["num"].apply(lambda s: set(s.tolist())).reset_index()
    return grp.rename(columns={"num": "winners_set"}).sort_values("date").reset_index(drop=True)


def run_walkforward_backtest(
    long_df: pd.DataFrame,
    start: date,
    end: date,
    score_cfg: ScoreConfig = ScoreConfig(),
    wf_cfg: WalkForwardConfig = WalkForwardConfig(),
    model: str = "heuristic",
    digit_cfg: DigitBiasConfig | None = None,
) -> pd.DataFrame:
    df = _normalize_long(long_df)
    winners = build_draw_universe(df, wf_cfg.buckets)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    winners = winners[(winners["date"] >= start_ts) & (winners["date"] <= end_ts)].copy()
    winners = winners.sort_values("date").reset_index(drop=True)
    if winners.empty:
        raise ValueError("No draw dates in the requested backtest window.")

    min_date = df["date"].min()
    rows = []

    for _, row in winners.iterrows():
        d: pd.Timestamp = row["date"]
        win_set: set[str] = row["winners_set"]

        days_hist = (d - min_date).days
        if days_hist < wf_cfg.train_min_days:
            continue

        train = df[df["date"] < d].copy()
        if train.empty:
            continue

        if model == "digit_bias":
            scored = score_numbers_digit_bias_long(
                train,
                as_of=d.date(),
                cfg=digit_cfg or DigitBiasConfig(train_window_days=score_cfg.train_window_days),
            )
        else:
            scored = score_numbers_long(train, as_of=d.date(), cfg=score_cfg)

        picks_df = top_picks(scored, top_n=wf_cfg.pick_top_n)
        picks = set(picks_df["num"].astype(str).tolist())
        hits = picks & win_set

        rows.append(
            {
                "date": d.date(),
                "n_picks": wf_cfg.pick_top_n,
                "n_winners": len(win_set),
                "winners": ",".join(sorted(win_set)),
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
    return out.sort_values("date").reset_index(drop=True)


def summarize_backtest(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}

    days = len(bt)
    any_hit_rate = float(bt["any_hit"].mean())
    avg_hits = float(bt["hits_count"].mean())
    avg_hit_rate_pick = float(bt["hit_rate"].mean())

    p50 = float(bt["hits_count"].quantile(0.50))
    p90 = float(bt["hits_count"].quantile(0.90))

    p_ge2 = float((bt["hits_count"] >= 2).mean())
    p_ge3 = float((bt["hits_count"] >= 3).mean())

    return {
        "days_tested": days,
        "any_hit_rate": any_hit_rate,
        "avg_hits_per_day": avg_hits,
        "avg_hit_rate_per_pick": avg_hit_rate_pick,
        "hits_count_p50": p50,
        "hits_count_p90": p90,
        "p_hits_ge2": p_ge2,
        "p_hits_ge3": p_ge3,
    }
