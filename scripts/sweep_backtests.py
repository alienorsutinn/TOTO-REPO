#!/usr/bin/env python3
"""
Grid-sweep backtest settings and save results.

It calls scripts/backtest_walkforward.py as a subprocess so it works with your current code
(without importing internals).

Outputs:
- data/processed/sweep_results.csv

Usage example:
python scripts/sweep_backtests.py \
  --numbers-long data/processed/numbers_long.csv \
  --start 2022-01-01 --end 2025-12-24 \
  --model heuristic \
  --random-trials 2000
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


SUMMARY_KV_RE = re.compile(r"^\s+([a-zA-Z0-9_]+):\s+([-+eE0-9.]+)\s*$")


def _parse_metrics(stdout: str) -> Dict[str, Any]:
    """
    Extract key metrics from your script output.
    Works with blocks like:
      Summary:
        any_hit_rate: 0.0515
    and
      P-values vs random (>= observed):
        p_any_hit_rate: 0.2414
    """
    out: Dict[str, Any] = {}
    mode = None
    for line in stdout.splitlines():
        if line.strip() == "Summary:":
            mode = "summary"
            continue
        if line.strip().startswith("P-values vs random"):
            mode = "pvals"
            continue
        if line.strip().startswith("Random baseline"):
            mode = "rand"
            continue
        if line.strip().startswith("Pick diversity"):
            mode = "div"
            continue

        m = SUMMARY_KV_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        try:
            vv: Any = float(v)
        except ValueError:
            vv = v

        if mode == "summary":
            out[k] = vv
        elif mode == "pvals":
            out[k] = vv
        elif mode == "rand":
            out[f"rand_{k}"] = vv
        elif mode == "div":
            out[f"div_{k}"] = vv

    return out


def _run_one(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    stdout = p.stdout
    stderr = p.stderr
    if p.returncode != 0:
        return {
            "ok": False,
            "returncode": p.returncode,
            "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
            "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        }

    metrics = _parse_metrics(stdout)
    metrics["ok"] = True
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", default="heuristic")
    ap.add_argument("--random-trials", type=int, default=2000)

    # grids (tweak freely)
    ap.add_argument("--tops", default="20,50,100")
    ap.add_argument("--train-windows", default="30,90,180,365,730")
    ap.add_argument("--recent-ks", default="0,3,5,10")
    ap.add_argument("--penalties", default="0.0,0.5,1.0,2.0")

    ap.add_argument("--out", default="data/processed/sweep_results.csv")
    args = ap.parse_args()

    tops = [int(x) for x in args.tops.split(",") if x.strip()]
    windows = [int(x) for x in args.train_windows.split(",") if x.strip()]
    ks = [int(x) for x in args.recent_ks.split(",") if x.strip()]
    penalties = [float(x) for x in args.penalties.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []

    grid = list(itertools.product(tops, windows, ks, penalties))
    print(f"Running {len(grid)} configs...")

    for i, (top, win, k, pen) in enumerate(grid, start=1):
        cmd = [
            "python",
            "scripts/backtest_walkforward.py",
            "--numbers-long",
            args.numbers_long,
            "--start",
            args.start,
            "--end",
            args.end,
            "--top",
            str(top),
            "--train-window-days",
            str(win),
            "--model",
            args.model,
            "--random-trials",
            str(args.random_trials),
        ]
        if k > 0 and pen > 0:
            cmd += ["--recent-draws-k", str(k), "--recent-draw-penalty", str(pen)]
        else:
            # explicitly turn off if your script defaults to something
            cmd += ["--recent-draws-k", "0", "--recent-draw-penalty", "0.0"]

        metrics = _run_one(cmd)
        metrics.update(
            {
                "cfg_top": top,
                "cfg_train_window_days": win,
                "cfg_recent_k": k,
                "cfg_recent_penalty": pen,
                "cfg_model": args.model,
            }
        )

        rows.append(metrics)

        if metrics.get("ok"):
            ah = metrics.get("any_hit_rate", None)
            pv = metrics.get("p_any_hit_rate", None)
            print(f"[{i:03d}/{len(grid)}] top={top} win={win} k={k} pen={pen} any_hit_rate={ah} p_any={pv}")
        else:
            print(f"[{i:03d}/{len(grid)}] FAILED top={top} win={win} k={k} pen={pen} rc={metrics.get('returncode')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    # quick ranking: lowest p_any_hit_rate then higher any_hit_rate
    if "p_any_hit_rate" in df.columns:
        best = df[df["ok"] == True].sort_values(["p_any_hit_rate", "any_hit_rate"], ascending=[True, False]).head(25)
        print("\nTop 25 by (p_any_hit_rate asc, any_hit_rate desc):")
        cols = ["cfg_top", "cfg_train_window_days", "cfg_recent_k", "cfg_recent_penalty", "any_hit_rate", "p_any_hit_rate", "p_p_hits_ge2"]
        cols = [c for c in cols if c in best.columns]
        print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
