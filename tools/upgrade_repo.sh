#!/usr/bin/env bash
set -euo pipefail

echo "==> Repo audit (quick)"
echo "PWD: $(pwd)"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }
git status -sb
echo "Remote:"
git remote -v || true
echo "Last commit:"
git --no-pager log -1 --oneline || true

echo
echo "--- merge markers check (ignore .venv/.git) ---"
grep -R --exclude-dir=.venv --exclude-dir=.git "<<<<<<<" -n . || echo "No merge markers found ✅"

echo
echo "==> Ensure venv + install (editable + extras)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip

echo "==> Ensure pyproject has dev+stats under single [project.optional-dependencies]"
python - <<'PY'
from __future__ import annotations
import re
from pathlib import Path

p = Path("pyproject.toml")
txt = p.read_text(encoding="utf-8")

hdr = "[project.optional-dependencies]"
if hdr not in txt:
    txt += "\n\n" + hdr + "\n"

# insert dev/stats only if missing in the section
# (simple: if no 'dev =' anywhere, append to the section end)
lines = txt.splitlines(True)

def is_section_header(ln: str) -> bool:
    s = ln.strip()
    return s.startswith("[") and s.endswith("]")

# find section span
start = None
for i, ln in enumerate(lines):
    if ln.strip() == hdr:
        start = i
        break

if start is None:
    raise SystemExit("optional-dependencies header missing unexpectedly")

end = start + 1
while end < len(lines) and not is_section_header(lines[end]):
    end += 1

section = "".join(lines[start:end])
need_dev = "dev" not in section
need_stats = "stats" not in section

ins = []
if need_dev:
    ins.append('dev = ["pytest>=7.4", "ruff>=0.6"]\n')
if need_stats:
    ins.append('stats = ["scipy>=1.10", "statsmodels>=0.14"]\n')

if ins:
    lines[end:end] = ins
    p.write_text("".join(lines), encoding="utf-8")
    print("Added missing extras ✅")
else:
    print("Extras already present ✅")
PY

pip install -e '.[dev,stats]'

echo
echo "==> Patch: fix python -m dmc4d entrypoint"
mkdir -p src/dmc4d
cat > src/dmc4d/__main__.py <<'PY'
from dmc4d.cli.main import app

if __name__ == "__main__":
    app()
PY

echo "==> Patch: tests import path"
mkdir -p tests
cat > tests/conftest.py <<'PY'
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PY

echo "==> Add: scripts/make_numbers_long.py"
mkdir -p scripts
cat > scripts/make_numbers_long.py <<'PY'
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
PY
chmod +x scripts/make_numbers_long.py

echo "==> Add: scripts/stat_tests.py (battery + correction + report.csv)"
# (Keep your previously generated stat_tests.py if it already exists, otherwise create it)
if [ ! -f scripts/stat_tests.py ]; then
  echo "scripts/stat_tests.py not found; re-run your earlier command that created it or tell me and I'll generate a lighter one."
else
  chmod +x scripts/stat_tests.py
fi

echo
echo "==> Sanity checks"
python -m compileall -q src || true
pytest -q || echo "pytest failed (paste output if you want me to fix remaining tests)"

echo
echo "==> Done ✅"
echo "Next:"
echo "  python scripts/make_numbers_long.py --results data/processed/results.csv --out data/processed/numbers_long.csv"
echo "  python scripts/stat_tests.py --numbers-long data/processed/numbers_long.csv --bucket all --correction bh --alpha 0.05"
