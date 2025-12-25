from datetime import date

import pandas as pd

from dmc4d.scoring.model import score_numbers_long, top_picks


def test_scoring_runs_on_small_sample():
    long = pd.DataFrame(
        [
            {"date": "2022-01-01", "bucket": "top3", "num": "1234"},
            {"date": "2022-01-02", "bucket": "starter", "num": "1234"},
            {"date": "2022-01-03", "bucket": "consolation", "num": "9999"},
        ]
    )
    
# as_of is exclusive: only dates strictly before as_of are used
    scored = score_numbers_long(long, as_of=date(2022, 1, 4))
    picks = top_picks(scored, top_n=2)
    assert len(picks) == 2
