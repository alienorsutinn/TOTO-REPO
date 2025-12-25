from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from dmc4d.backtest.metrics import summarize


def write_reports(per_draw: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(per_draw)
    per_draw_path = out_dir / "per_draw.csv"
    summary_path = out_dir / "summary.json"
    per_draw.to_csv(per_draw_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"per_draw": per_draw_path, "summary": summary_path, "summary_obj": summary}
