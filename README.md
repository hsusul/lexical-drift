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

## Synthetic Data Difficulty

`generate-synth` now supports configurable difficulty and drift/noise controls:
`--difficulty {easy,hard}` plus optional overrides
`--drift-strength`, `--noise-strength`, `--global-event-strength`, and
`--topic-shift-strength`.

Example hard-mode generation:

```bash
lexdrift generate-synth --out data/raw/synth_hard.csv --n-authors 80 --months 12 --difficulty hard
```

## Deep Learning Baseline

```bash
pip install -e ".[dev,torch]"
lexdrift train-nn --config configs/train_nn.yaml
```

## Temporal Transformer Baseline

Encodes monthly writing with a frozen transformer and trains a GRU over time.

```bash
pip install -e ".[dev,torch,nlp]"
lexdrift train-temporal --config configs/train_temporal.yaml
```

## Temporal Evaluation

Train on months `[0..train_months-1]`, then evaluate each later month using prefix sequences
up to that month. This workflow is for research experiments and is **not a medical diagnosis
tool**.

`EvalTemporalConfig.model_type` supports:
- `gru` (default): temporal GRU classifier over monthly embedding prefixes.
- `baseline_lr`: non-temporal logistic regression baseline on month embeddings.
- `attention`: temporal self-attention encoder with positional embeddings.
- `transformer`: temporal transformer encoder with optional month index embeddings.

Class-imbalance controls are available in eval/train configs:
- `loss_type: bce | focal`
- `pos_weight: <float|null>`
- `focal_gamma: <float>`

Each `eval-temporal` run writes these plot artifacts into that run's `output_dir`:
- `per_month_metrics.png`
- `threshold_over_time.png`
- `pred_rate_over_time.png`
- `embedding_drift_over_time.png`
- `drift_vs_accuracy_delta.png`

```bash
lexdrift eval-temporal --config configs/eval_temporal.yaml
lexdrift eval-temporal --config configs/eval_temporal_fixed.yaml
lexdrift eval-temporal --config configs/eval_temporal_transformer_time.yaml
lexdrift eval-temporal --config configs/eval_temporal_transformer_notime.yaml
```

## End-to-End Temporal Pipeline

Train and evaluate a temporal GRU where encoder representations are produced directly in the
training loop (no embedding cache):

```bash
lexdrift train-e2e --config configs/train_e2e_temporal.yaml
lexdrift train-e2e --config configs/train_e2e_temporal_focal.yaml
lexdrift eval-e2e --config configs/eval_e2e_temporal.yaml
```

## Contrastive Pretraining

Pretrain the encoder with adjacent-month positives (InfoNCE), then use the checkpoint in
`train-e2e` via `pretrained_encoder_path`:

```bash
lexdrift pretrain-contrastive --config configs/pretrain_contrastive.yaml
```

## Temporal Order Pretraining

Pretrain an encoder to predict whether adjacent monthly texts are in chronological order:

```bash
lexdrift pretrain-temporal-order --config configs/pretrain_temporal_order.yaml
```

## Multitask Temporal Training

Train a joint classifier + drift regressor and run drift-weight ablations:

```bash
lexdrift train-multitask --config configs/train_multitask.yaml
lexdrift ablation-drift-weight --lambdas 0,0.1,0.3,1.0 --seeds 1,2,3 --months 12
```

## Time-Embedding Ablation

Compare transformer temporal evaluation with and without explicit month embeddings:

```bash
lexdrift ablation-time-embeddings --config configs/eval_temporal_transformer_time.yaml \
  --seeds 1,2,3 --n-authors 50 --months 12 --difficulty hard
```

## Real Dataset Preparation

Normalize a local CSV/JSONL dataset to the expected schema, then run temporal evaluation:

```bash
lexdrift prepare-real --input data/raw/real_sample.csv --out data/processed/real.parquet
lexdrift eval-temporal-real --dataset prepared_local --path data/processed/real.parquet --config configs/real_eval.yaml
```

## Benchmark

Run repeated seed-based evaluation and compare baseline, NN, and temporal models:

```bash
lexdrift benchmark --seeds 1,2,3
```

Results are appended to `artifacts/benchmark_results.jsonl`.

## Eval Sweep

Run temporal evaluation across multiple synthetic seeds to get more stable metrics:

```bash
lexdrift eval-temporal-sweep --seeds 1,2,3 --n-authors 80 --months 12 --difficulty hard
```

This writes per-seed results to `artifacts/eval_temporal_sweep.jsonl` and prints aggregate
final-month and all-eval-month summaries.

## Eval Compare

Compare two eval configs on the same synthetic seeds:

```bash
lexdrift eval-temporal-compare \
  --config-a configs/eval_temporal_fixed.yaml \
  --config-b configs/eval_temporal_calib.yaml \
  --seeds 1,2,3 \
  --n-authors 50 \
  --months 12 \
  --difficulty hard
```

## Report Rendering

Generate a markdown report from compare results:

```bash
lexdrift render-report \
  --compare-summary artifacts/eval_temporal_compare_summary.json \
  --out docs/report.md
```

## How to Run Experiments and Generate a Report

```bash
# 1) baseline sweep
lexdrift eval-temporal-sweep --seeds 1,2,3 --n-authors 80 --months 12 --difficulty hard

# 2) compare two temporal configs (for example fixed vs calibrated thresholds)
lexdrift eval-temporal-compare \
  --config-a configs/eval_temporal_fixed.yaml \
  --config-b configs/eval_temporal_calib.yaml \
  --seeds 1,2,3 \
  --n-authors 50 \
  --months 12 \
  --difficulty hard

# 3) optional contrastive + multitask runs
lexdrift pretrain-contrastive --config configs/pretrain_contrastive.yaml
lexdrift train-multitask --config configs/train_multitask.yaml

# 4) render artifact-driven report
lexdrift render-report \
  --compare-summary artifacts/eval_temporal_compare_summary.json \
  --out docs/report.md
```

## Dashboard

Optional Streamlit dashboard for browsing sweep and compare outputs:

```bash
pip install -e ".[dev,viz]"
streamlit run apps/dashboard/app.py
```

## CLI

```bash
lexdrift generate-synth --out data/raw/synth.csv --n-authors 50 --months 12
lexdrift train-baseline --config configs/train_baseline.yaml
lexdrift train-nn --config configs/train_nn.yaml
lexdrift train-temporal --config configs/train_temporal.yaml
lexdrift eval-temporal --config configs/eval_temporal.yaml
lexdrift eval-temporal-sweep --seeds 1,2,3 --n-authors 80 --months 12
lexdrift eval-temporal-compare --config-a configs/eval_temporal_fixed.yaml --config-b configs/eval_temporal_calib.yaml --seeds 1,2,3 --n-authors 50 --months 12 --difficulty hard
lexdrift train-e2e --config configs/train_e2e_temporal.yaml
lexdrift eval-e2e --config configs/eval_e2e_temporal.yaml
lexdrift pretrain-contrastive --config configs/pretrain_contrastive.yaml
lexdrift pretrain-temporal-order --config configs/pretrain_temporal_order.yaml
lexdrift train-multitask --config configs/train_multitask.yaml
lexdrift ablation-drift-weight --lambdas 0,0.1,0.3,1.0 --seeds 1,2,3 --months 12
lexdrift ablation-time-embeddings --config configs/eval_temporal_transformer_time.yaml --seeds 1,2,3
lexdrift prepare-real --input data/raw/real_sample.csv --out data/processed/real.parquet
lexdrift eval-temporal-real --dataset prepared_local --path data/processed/real.parquet --config configs/real_eval.yaml
lexdrift benchmark --seeds 1,2,3
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

- Expand lexical and syntactic feature ablations.
- Add dataset cards and data quality checks.
- Add optional baseline comparison models.
