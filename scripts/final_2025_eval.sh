#!/usr/bin/env bash
set -euo pipefail

NUMBERS="data/processed/numbers_long_deduped.csv"
OUTDIR="data/processed/final2025"
mkdir -p "$OUTDIR"

BASE=(
  python scripts/backtest_multifeature.py
  --numbers-long "$NUMBERS"
  --train-window-days 365
  --train-bucket top3
  --target-buckets top3
  --start 2025-01-01
  --end 2025-12-24
  --random-trials 20000
)

# Keep a small, distinct set (no point running alpha variants if they behave the same)
declare -a CONFIGS=(
  "A_noPairs_top200_dp2_mpb3|--alpha 1.0 --top 200 --diversity-prefix 2 --max-per-bucket 3"
  "B_noPairs_top200_dp2_mpb2|--alpha 1.0 --top 200 --diversity-prefix 2 --max-per-bucket 2"
  "C_noPairs_top200_dp2_mpb1|--alpha 1.0 --top 200 --diversity-prefix 2 --max-per-bucket 1"
  "D_pairs_top200_dp2_mpb2|--alpha 1.0 --top 200 --use-pairs --w-digits 1.0 --w-pairs 1.0 --diversity-prefix 2 --max-per-bucket 2"
  "E_pairs_top200_dp2_mpb3|--alpha 1.0 --top 200 --use-pairs --w-digits 1.0 --w-pairs 1.0 --diversity-prefix 2 --max-per-bucket 3"
  "F_noPairs_top200_dp1_mpb2|--alpha 1.0 --top 200 --diversity-prefix 1 --max-per-bucket 2"
)

SUMMARY="$OUTDIR/summary.csv"
echo "name,any_hit_rate,p_any_hit_rate,unique_last200,total_last200,top_prefixes" > "$SUMMARY"

for item in "${CONFIGS[@]}"; do
  name="${item%%|*}"
  args="${item#*|}"

  out_csv="$OUTDIR/${name}.csv"
  out_log="$OUTDIR/${name}.log"

  echo ""
  echo "=== RUN $name ==="
  echo "out: $out_csv"

  # run backtest
  "${BASE[@]}" --out "$out_csv" $args 2>&1 | tee "$out_log"

  # extract extra diagnostics from the produced CSV
  python - <<'PY' "$name" "$out_csv" "$SUMMARY"
import sys, re
import pandas as pd
from collections import Counter

name, csv_path, summary_path = sys.argv[1], sys.argv[2], sys.argv[3]

df = pd.read_csv(csv_path)

# any_hit_rate + p_any from log-style keys if present in the CSV metadata? (not there)
# So we parse the log file next to the csv.
log_path = csv_path.replace(".csv", ".log")
try:
    log = open(log_path, "r", encoding="utf-8").read()
except FileNotFoundError:
    log = ""

def grab(key):
    m = re.search(rf"{re.escape(key)}:\s*([0-9\.eE\-]+)", log)
    return float(m.group(1)) if m else None

any_hit = grab("any_hit_rate")
p_any   = grab("p_any_hit_rate")

tail = df.tail(200)
picks = ",".join(tail["picks"].astype(str)).split(",")
picks = [p for p in picks if p and p != "nan"]

unique_last200 = len(set(picks))
total_last200 = len(picks)

# prefix concentration (2-digit prefixes)
c2 = Counter([p[:2] for p in picks if len(p) >= 2])
top_prefixes = ";".join([f"{k}:{v}" for k,v in c2.most_common(5)])

line = f"{name},{any_hit},{p_any},{unique_last200},{total_last200},{top_prefixes}\n"
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(line)

print(f"[diag] unique picks last200 = {unique_last200} / {total_last200}")
print(f"[diag] top 2-digit prefixes: {c2.most_common(10)}")
PY

done

echo ""
echo "=== DONE ==="
echo "Summary -> $SUMMARY"
column -s, -t "$SUMMARY" | sed -n '1,20p'
