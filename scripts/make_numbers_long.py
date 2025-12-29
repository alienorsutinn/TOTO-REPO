#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results.csv (wide format)")
    p.add_argument("--out", required=True, help="Path to write numbers_long.csv")
    return p.parse_args()

def _split_nums(s: str) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip().zfill(4) for x in s.split(",") if x.strip()]

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results)
    req = {"date", "draw_no", "operator", "top3", "starter", "consolation"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise SystemExit(f"results.csv missing required columns: {missing}")

    rows = []
    for _, r in df.iterrows():
        date = str(r["date"]).strip()
        draw_no = str(r["draw_no"]).strip()
        operator = str(r["operator"]).strip()

        for n in _split_nums(r["top3"]):
            rows.append({"date": date, "draw_no": draw_no, "operator": operator, "bucket": "top3", "num": n})
        for n in _split_nums(r["starter"]):
            rows.append({"date": date, "draw_no": draw_no, "operator": operator, "bucket": "starter", "num": n})
        for n in _split_nums(r["consolation"]):
            rows.append({"date": date, "draw_no": draw_no, "operator": operator, "bucket": "consolation", "num": n})

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out[out["date"].notna()].copy()
    out["num"] = out["num"].astype(str).str.zfill(4)
    out = out[out["num"].str.fullmatch(r"\d{4}", na=False)].copy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(out)}")

if __name__ == "__main__":
    main()
