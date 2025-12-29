import itertools, subprocess, re, csv, hashlib
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path

BASE = [
  "python","scripts/backtest_multifeature.py",
  "--numbers-long","data/processed/numbers_long_deduped.csv",
  "--train-window-days","365",
  "--train-bucket","top3","--target-buckets","top3",
]

TUNE = ("2022-01-01","2023-12-31")
VAL  = ("2024-01-01","2024-12-31")

TOPS = [50, 100, 200]
USE_PAIRS = [0, 1]
ALPHAS = [0.5, 1.0, 2.0]
DIV_PREFIX = [0, 1, 2]
MAX_PER = [1, 2, 3]

TRIALS = "0"  # skip MC for sweep

def grab(out: str, key: str):
    m = re.search(rf"{re.escape(key)}:\s*([0-9\.eE\-]+)", out)
    return float(m.group(1)) if m else None

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        return None
    return {
        "any": grab(out, "any_hit_rate"),
        "avg": grab(out, "avg_hits_per_day"),
        "days": grab(out, "days_tested"),
    }

def one_config(cfg):
    top, use_pairs, alpha, dp, mpb = cfg
    cmd_core = BASE + ["--alpha", str(alpha), "--top", str(top), "--random-trials", TRIALS]

    if use_pairs:
        cmd_core += ["--use-pairs", "--w-digits", "1.0", "--w-pairs", "1.0"]

    if dp > 0:
        cmd_core += ["--diversity-prefix", str(dp), "--max-per-bucket", str(mpb)]

    h = hashlib.md5((" ".join(cmd_core)).encode()).hexdigest()[:10]
    out_tune = f"/tmp/bt_tune_{h}.csv"
    out_val  = f"/tmp/bt_val_{h}.csv"

    st_tune = run_cmd(cmd_core + ["--start", TUNE[0], "--end", TUNE[1], "--out", out_tune])
    if st_tune is None:
        return None

    st_val = run_cmd(cmd_core + ["--start", VAL[0], "--end", VAL[1], "--out", out_val])
    if st_val is None:
        return None

    return {
        "top": top,
        "pairs": use_pairs,
        "alpha": alpha,
        "div_prefix": dp,
        "max_per": mpb if dp > 0 else 0,
        "any_tune": st_tune["any"],
        "any_val": st_val["any"],
        "avg_val": st_val["avg"],
        "cmd": " ".join(cmd_core),
        "out_val": out_val,
    }

def main():
    cfgs = list(itertools.product(TOPS, USE_PAIRS, ALPHAS, DIV_PREFIX, MAX_PER))
    procs = max(1, cpu_count() - 1)
    print(f"Running {len(cfgs)} configs (tune+val) on {procs} procs...")

    # spawn is fine now because this is a real file, but we can be explicit:
    ctx = get_context("spawn")
    with ctx.Pool(processes=procs) as pool:
        rows = [r for r in pool.imap_unordered(one_config, cfgs, chunksize=1) if r is not None]

    rows.sort(key=lambda r: (-(r["any_val"] or -1), -(r["any_tune"] or -1)))

    print("\nTOP 20 by 2024 validation any_hit_rate:")
    for i, r in enumerate(rows[:20], 1):
        print(f"{i:02d} val_any={r['any_val']:.6f}  tune_any={r['any_tune']:.6f}  :: {r['cmd']}  div={r['div_prefix']}/{r['max_per']}")

    outp = Path("data/processed/sweep_tune2022_23_val2024.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cmd"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote -> {outp}")

if __name__ == "__main__":
    main()
