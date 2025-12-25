from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v == "" else v

@dataclass(frozen=True)
class Config:
    results_csv: Path = Path(_env("DMC_RESULTS_CSV", "data/processed/results.csv"))
    births_dir: Path = Path(_env("DMC_BIRTHS_DIR", "data/processed"))
    crowd_scores_csv: Path = Path(_env("DMC_CROWD_SCORES_CSV", "data/processed/crowd_scores.csv"))

    start_date: str = _env("DMC_START_DATE", "2022-01-01")
    end_date: str = _env("DMC_END_DATE", "")

    report_dir: Path = Path(_env("DMC_REPORT_DIR", "reports"))
    log_level: str = _env("DMC_LOG_LEVEL", "INFO")

CFG = Config()
