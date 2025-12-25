from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run_one(
    numbers_long: str,
    start: str,
    end: str,
    train_window_days: int,
    top: int,
    alpha: float,
    random_trials: int,
) -> dict:
    cmd = [
        sys.executable,
        "scripts/backtest_top3_pos0_model.py",
        "--numbers-long", numbers_long,
        "--start", start,
        "--end", end,
        "--train-window-days", str(train_window_days),
        "--top", str(top),
        "--alpha", str(alpha),
        "--random-trials", str(random_trials),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.strip()
    err = proc.stderr.strip()

    if proc.returncode != 0:
        return {
            "train_window_days": train_window_days,
            "top": top,
            "alpha": alpha,
            "ok": False,
            "error": err[-800:] if err else "unknown error",
        }

    # Parse summary + p-values from stdout
    # We keep it simple + robust: scan for "Summary:" and "P-values vs random"
    lines = out.splitlines()

    def _get_float(prefix: str) -> float | None:
        for ln in lines:
            if ln.strip().startswith(prefix):
                # e.g. "  any_hit_rate: 0.0515625"
                try:
                    return float(ln.split(":", 1)[1].strip())
                except Exception:
                    return None
        return None

    res = {
        "train_window_days": train_window_days,
        "top": top,
        "alpha": alpha,
        "ok": True,
        "any_hit_rate": _get_float("any_hit_rate"),
        "avg_hits_per_day": _get_float("avg_hits_per_day"),
        "p_any_hit_rate": _get_float("p_any_hit_rate"),
        "p_avg_hits_per_day": _get_float("p_avg_hits_per_day"),
        "p_p_hits_ge2": _get_float("p_p_hits_ge2"),
    }
    # Also store last bit of stdout so you can debug quickly if needed
    res["stdout_tail"] = "\n".join(lines[-25:])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers-long", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)

    ap.add_argument("--tops", default="10,20,50,100")
    ap.add_argument("--windows", default="30,60,90,120,180,365")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--random-trials", type=int, default=3000)

    ap.add_argument("--out", default="data/processed/grid_top3_pos0.csv")
    args = ap.parse_args()

    tops = [int(x.strip()) for x in args.tops.split(",") if x.strip()]
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    rows = []
    total = len(tops) * len(windows)
    i = 0

    for w in windows:
        for top in tops:
            i += 1
            print(f"[{i}/{total}] window={w} top={top} alpha={args.alpha}")
            r = run_one(
                numbers_long=args.numbers_long,
                start=args.start,
                end=args.end,
                train_window_days=w,
                top=top,
                alpha=args.alpha,
                random_trials=args.random_trials,
            )
            rows.append(r)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    ok = df[df["ok"] == True].copy()
    if ok.empty:
        print("\nNo successful runs. Check errors:")
        print(df[["train_window_days", "top", "error"]].to_string(index=False))
        return

    # Rank by lowest p_any_hit_rate then highest any_hit_rate
    ok["p_any_hit_rate"] = pd.to_numeric(ok["p_any_hit_rate"], errors="coerce")
    ok["any_hit_rate"] = pd.to_numeric(ok["any_hit_rate"], errors="coerce")
    ok_sorted = ok.sort_values(["p_any_hit_rate", "any_hit_rate"], ascending=[True, False])

    print("\nTop 10 configs (lowest p_any_hit_rate):")
    cols = ["train_window_days", "top", "any_hit_rate", "avg_hits_per_day", "p_any_hit_rate", "p_avg_hits_per_day", "p_p_hits_ge2"]
    print(ok_sorted[cols].head(10).to_string(index=False))

    # Show best config stdout tail for quick sanity
    best = ok_sorted.iloc[0].to_dict()
    print("\nBest config stdout tail:")
    print(best.get("stdout_tail", ""))


if __name__ == "__main__":
    main()
