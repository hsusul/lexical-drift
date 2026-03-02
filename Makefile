.PHONY: setup lint test demo demo-v2

setup:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

demo:
	PYTHONPATH=src python -m lexical_drift.cli generate-synth \
		--out data/raw/synth.csv --n-authors 50 --months 12
	PYTHONPATH=src python -m lexical_drift.cli train-baseline --config configs/train_baseline.yaml

demo-v2:
	PYTHONPATH=src python -m lexical_drift.cli run-hero-demo --fast
