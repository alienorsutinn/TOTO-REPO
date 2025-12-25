from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COLS = ["date", "draw_no", "operator", "top3", "starter", "consolation"]


def read_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLS)

    df = pd.read_csv(path)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""

    return df[COLS].copy()


def upsert_results(rows: list[dict], out_path: str | Path) -> None:
    """
    Append/merge result rows into a CSV at out_path.

    - If rows empty, still write a header-only CSV (so downstream reads don't crash).
    - Deduplicate on (date, draw_no, operator).
    - Sort by date.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        df_empty = pd.DataFrame(columns=COLS)
        df_empty.to_csv(out_path, index=False)
        logger.info("Saved results -> %s (rows=%s)", out_path, 0)
        return

    df_new = pd.DataFrame(rows).copy()

    # Normalize list columns to comma-separated strings
    for col in ["top3", "starter", "consolation"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].apply(
                lambda x: ",".join(x) if isinstance(x, list) else (x if x is not None else "")
            )

    for c in COLS:
        if c not in df_new.columns:
            df_new[c] = ""
    df_new = df_new[COLS]

    if out_path.exists():
        df_old = read_results(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["date", "draw_no", "operator"], keep="last")
    else:
        df_all = df_new

    df_all = df_all.sort_values(["date", "draw_no"]).reset_index(drop=True)
    df_all.to_csv(out_path, index=False)
    logger.info("Saved results -> %s (rows=%s)", out_path, len(df_all))
