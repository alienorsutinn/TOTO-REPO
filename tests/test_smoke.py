from __future__ import annotations
from pathlib import Path

from dmc4d.datasets.births_store import build_birth_maps
from dmc4d.crowd.crowd_model import score_all_numbers
from dmc4d.data_sources.local_csv import LocalCSVResults
from dmc4d.datasets.results_store import upsert_results


def test_smoke(tmp_path: Path):
    births_dir = tmp_path / "births"
    build_birth_maps(Path("tests/fixtures/malaysia_births_sample.csv"), births_dir)

    scores = tmp_path / "scores.csv"
    score_all_numbers(births_dir, scores)
    assert scores.exists()

    rows = LocalCSVResults(Path("tests/fixtures/results_sample.csv")).load()
    out = tmp_path / "results.csv"
    upsert_results(rows, out)
    assert out.exists()
