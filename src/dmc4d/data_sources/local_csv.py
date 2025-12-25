from __future__ import annotations
from pathlib import Path
import pandas as pd
from dmc4d.data_sources.base import ResultsSource, parse_list_field
from dmc4d.datasets.schemas import validate_draw_lists


class LocalCSVResults(ResultsSource):
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def load(self) -> list[dict]:
        df = pd.read_csv(self.csv_path)
        required = {"date", "draw_no", "operator", "top3", "starter", "consolation"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in results csv: {missing}")
        rows: list[dict] = []
        for _, r in df.iterrows():
            top3 = parse_list_field(r["top3"], 3)
            starter = parse_list_field(r["starter"], 10)
            consolation = parse_list_field(r["consolation"], 10)
            validate_draw_lists(top3, starter, consolation)
            rows.append(
                {
                    "date": str(r["date"]),
                    "draw_no": str(r["draw_no"]),
                    "operator": str(r["operator"]),
                    "top3": top3,
                    "starter": starter,
                    "consolation": consolation,
                }
            )
        return rows
