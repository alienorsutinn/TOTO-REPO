from __future__ import annotations
from pathlib import Path
import pandas as pd
from dmc4d.datasets.schemas import validate_draw_lists
from dmc4d.logging import get_logger

log = get_logger(__name__)

def upsert_results(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)

    # normalize
    for col in ["top3", "starter", "consolation"]:
        if col not in df_new.columns:
            raise ValueError(f"Missing column: {col}")

    for _, r in df_new.iterrows():
        validate_draw_lists(r["top3"], r["starter"], r["consolation"])

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        # store lists as JSON strings in CSV; we keep them as python lists in memory
        # if user edits: they should use CLI to regenerate.
        pass

    # store as CSV with json-ish representation for simplicity
    df_out = df_new.copy()
    df_out["top3"] = df_out["top3"].apply(lambda x: ",".join(x))
    df_out["starter"] = df_out["starter"].apply(lambda x: ",".join(x))
    df_out["consolation"] = df_out["consolation"].apply(lambda x: ",".join(x))
    df_out.to_csv(out_path, index=False)
    log.info("Saved results -> %s (rows=%d)", out_path, len(df_out))
    return out_path

def read_results(out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(out_path)
    df["top3"] = df["top3"].apply(lambda s: [x.zfill(4) for x in str(s).split(",")])
    df["starter"] = df["starter"].apply(lambda s: [x.zfill(4) for x in str(s).split(",")])
    df["consolation"] = df["consolation"].apply(lambda s: [x.zfill(4) for x in str(s).split(",")])
    return df
