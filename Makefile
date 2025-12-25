.PHONY: setup lint test demo

setup:
	@echo "python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e '.[dev]'"

lint:
	ruff check src tests

test:
	pytest -q

demo:
	python -m dmc4d build-births --input tests/fixtures/malaysia_births_sample.csv --out data/processed
	python -m dmc4d fetch-results --source local --input tests/fixtures/results_sample.csv --out data/processed
	python -m dmc4d score-numbers --births-dir data/processed --out data/processed/crowd_scores.csv
	python -m dmc4d backtest --results data/processed/results.csv --strategy low_crowd --budget 20 --start 2022-01-01 --end 2025-12-31
