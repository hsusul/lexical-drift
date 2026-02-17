# lexical-drift

`lexical-drift` is a lightweight research prototype for language drift detection over time.
It provides a CPU-only baseline workflow for synthetic data generation, model training, and
single-text inference.
This repository is for experimentation and benchmarking and is **not a medical diagnosis tool**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make demo
```

## What `make demo` does

- Generates a small synthetic longitudinal text dataset in `data/raw/synth.csv`.
- Trains a TF-IDF + logistic regression baseline from `configs/train_baseline.yaml`.
- Writes model artifacts to `artifacts/baseline.joblib` and `artifacts/metadata.json`.

## Deep Learning Baseline (Day 2)

```bash
pip install -e ".[dev,dl]"
lexdrift train-nn --config configs/train_nn.yaml
```

## CLI

```bash
lexdrift generate-synth --out data/raw/synth.csv --n-authors 50 --months 12
lexdrift train-baseline --config configs/train_baseline.yaml
lexdrift train-nn --config configs/train_nn.yaml
lexdrift predict --model artifacts/baseline.joblib --text "I keep using like filler words now"
```

## Repo Layout

- `configs/`: training configuration files.
- `data/raw/`, `data/processed/`: data directories.
- `src/lexical_drift/`: Python package source code.
- `src/lexical_drift/datasets/`: synthetic dataset generation.
- `src/lexical_drift/features/`: lexical feature utilities.
- `src/lexical_drift/models/`: baseline model construction and evaluation.
- `src/lexical_drift/training/`: training entrypoints.
- `src/lexical_drift/inference/`: prediction utilities.
- `tests/`: smoke and training tests.
- `.github/workflows/ci.yml`: CI workflow for lint and tests.

## Next Milestones

- Add temporal split evaluation and per-month performance tracking.
- Expand lexical and syntactic feature ablations.
- Add dataset cards and data quality checks.
- Add optional baseline comparison models.
