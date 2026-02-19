from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.features.encoder import encode_texts_to_embeddings
from lexical_drift.training.train_temporal import compute_cache_fingerprint
from lexical_drift.utils import ensure_dir

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def _import_torch_modules():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError('PyTorch is not installed. Install with: pip install -e ".[dl]"') from exc

    return torch, DataLoader, TensorDataset


def _cache_file_path(cache_dir: str | Path, model_name: str) -> Path:
    safe_model = model_name.replace("/", "__")
    return Path(cache_dir) / f"temporal_embeddings_{safe_model}.npz"


def _validate_equal_sequence_lengths(sequences_texts: list[list[str]]) -> int:
    lengths = sorted({len(sequence) for sequence in sequences_texts})
    if len(lengths) != 1:
        raise ValueError("All authors must have the same number of months for temporal evaluation")
    return lengths[0]


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    calibration_metric: str,
) -> float:
    thresholds = np.round(np.arange(0.05, 0.951, 0.01, dtype=np.float64), 2)
    if not np.any(np.isclose(thresholds, 0.5)):
        thresholds = np.sort(np.append(thresholds, 0.5))
    best_threshold = 0.5
    best_metric = -np.inf
    best_tnr = -np.inf

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metric_values = {
            "balanced_accuracy": 0.5 * (tpr + tnr),
            "youden_j": tpr - fpr,
        }
        metric_value = metric_values[calibration_metric]
        if metric_value > best_metric + 1e-12 or (
            abs(metric_value - best_metric) <= 1e-12 and tnr > best_tnr + 1e-12
        ):
            best_metric = metric_value
            best_tnr = tnr
            best_threshold = float(threshold)

    return best_threshold


def _load_or_encode_embeddings(
    config: EvalTemporalConfig,
    *,
    authors: list[str],
    sequences_texts: list[list[str]],
    months_matrix: np.ndarray,
    cache_fingerprint: str,
) -> tuple[np.ndarray, Path, bool]:
    cache_path = _cache_file_path(config.cache_dir, config.encoder_model)
    embeddings: np.ndarray | None = None
    used_cache = False

    if config.cache_embeddings and cache_path.exists():
        print(f"[eval-temporal] found cache file: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        cached_authors = cached["authors"].astype(str).tolist()
        cached_months = cached["months"]
        cached_embeddings = cached["embeddings"].astype(np.float32)
        cached_fingerprint = str(cached["fingerprint"].item()) if "fingerprint" in cached else ""

        if (
            cached_fingerprint == cache_fingerprint
            and cached_authors == authors
            and np.array_equal(cached_months, months_matrix)
        ):
            embeddings = cached_embeddings
            used_cache = True
            print("[eval-temporal] loaded cached embeddings")
        else:
            print("[eval-temporal] cache miss, recomputing embeddings")

    if embeddings is None:
        flat_texts = [text for sequence in sequences_texts for text in sequence]
        print(
            "[eval-temporal] encoding "
            f"{len(flat_texts)} monthly texts with model={config.encoder_model}"
        )
        flat_embeddings = encode_texts_to_embeddings(
            texts=flat_texts,
            model_name=config.encoder_model,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        n_authors = len(authors)
        n_months = int(months_matrix.shape[1])
        embedding_dim = int(flat_embeddings.shape[1])
        embeddings = flat_embeddings.reshape(n_authors, n_months, embedding_dim).astype(np.float32)

        if config.cache_embeddings:
            ensure_dir(config.cache_dir)
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                authors=np.asarray(authors, dtype=object),
                months=months_matrix,
                fingerprint=np.asarray(cache_fingerprint),
            )
            print(f"[eval-temporal] saved embeddings cache to {cache_path}")

    return embeddings, cache_path, used_cache


def run_eval_temporal(config: EvalTemporalConfig) -> dict[str, object]:
    torch, DataLoader, TensorDataset = _import_torch_modules()
    from lexical_drift.models.temporal_gru import build_temporal_gru

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    data_path = Path(config.input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    authors, sequences_texts, sequences_months, labels = build_author_sequences_with_months(frame)
    n_months = _validate_equal_sequence_lengths(sequences_texts)
    if config.train_months >= n_months:
        raise ValueError(
            f"train_months must be < total months ({n_months}), got {config.train_months}"
        )

    months_matrix = np.asarray(sequences_months, dtype=np.int64)
    cache_fingerprint = compute_cache_fingerprint(
        data_path,
        encoder_model=config.encoder_model,
        max_length=config.max_length,
    )
    embeddings, cache_path, used_cache = _load_or_encode_embeddings(
        config,
        authors=authors,
        sequences_texts=sequences_texts,
        months_matrix=months_matrix,
        cache_fingerprint=cache_fingerprint,
    )

    indices = np.arange(len(authors), dtype=np.int64)
    if config.test_size is None:
        train_idx = indices
        eval_idx = indices
    else:
        stratify = labels if len(np.unique(labels)) > 1 else None
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=stratify,
        )

    X_train = embeddings[train_idx, : config.train_months, :]
    y_train = labels[train_idx].astype(np.float32)
    y_eval = labels[eval_idx].astype(int)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )
    train_generator = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
    )

    input_dim = int(embeddings.shape[2])
    model = build_temporal_gru(
        input_dim=input_dim,
        hidden_dim=config.gru_hidden_dim,
        layers=config.gru_layers,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_x.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_count += batch_size
        avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"[eval-temporal] epoch={epoch + 1}/{config.epochs} train_loss={avg_loss:.4f}")

    model.eval()
    per_month: list[dict[str, float | int]] = []
    final_month_probs: np.ndarray | None = None
    final_month_preds: np.ndarray | None = None
    chosen_threshold = float(config.fixed_threshold)
    calibrate_first_mode = config.threshold_mode == "calibrate_first_eval"
    calibrate_each_mode = config.threshold_mode == "calibrate_each_month"
    with torch.no_grad():
        for month_index in range(config.train_months, n_months):
            X_eval = embeddings[eval_idx, : month_index + 1, :]
            logits = model(torch.from_numpy(X_eval))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            threshold_used = chosen_threshold
            if calibrate_each_mode:
                threshold_used = choose_threshold(y_eval, probs, config.calibration_metric)
                chosen_threshold = threshold_used
            elif calibrate_first_mode and month_index == config.train_months:
                threshold_used = choose_threshold(y_eval, probs, config.calibration_metric)
                chosen_threshold = threshold_used
            preds = (probs >= threshold_used).astype(int)
            tn = int(np.sum((preds == 0) & (y_eval == 0)))
            fp = int(np.sum((preds == 1) & (y_eval == 0)))
            fn = int(np.sum((preds == 0) & (y_eval == 1)))
            tp = int(np.sum((preds == 1) & (y_eval == 1)))
            month_metrics = _compute_binary_metrics(y_eval, preds)
            roc_auc: float | None = None
            pr_auc: float | None = None
            if np.unique(y_eval).size > 1:
                roc_auc = float(roc_auc_score(y_eval, probs))
                pr_auc = float(average_precision_score(y_eval, probs))
            per_month.append(
                {
                    "month_index": int(month_index),
                    "accuracy": month_metrics["accuracy"],
                    "f1": month_metrics["f1"],
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "true_pos_rate": float(np.mean(y_eval)),
                    "pred_pos_rate": float(np.mean(preds)),
                    "mean_pred_prob": float(np.mean(probs)),
                    "threshold_used": float(threshold_used),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
            final_month_probs = probs
            final_month_preds = preds

    if not per_month:
        raise ValueError(
            "No evaluation months available. Increase total months or reduce train_months."
        )
    if final_month_probs is None or final_month_preds is None:
        raise ValueError("Failed to compute final month predictions")

    final_month = per_month[-1]
    accuracy_values = np.asarray(
        [float(entry["accuracy"]) for entry in per_month], dtype=np.float64
    )
    f1_values = np.asarray([float(entry["f1"]) for entry in per_month], dtype=np.float64)
    per_month_summary = {
        "accuracy_min": float(accuracy_values.min()),
        "accuracy_mean": float(accuracy_values.mean()),
        "accuracy_max": float(accuracy_values.max()),
        "f1_min": float(f1_values.min()),
        "f1_mean": float(f1_values.mean()),
        "f1_max": float(f1_values.max()),
    }
    summary = {
        "accuracy_mean": float(accuracy_values.mean()),
        "accuracy_std": float(accuracy_values.std()),
        "f1_mean": float(f1_values.mean()),
        "f1_std": float(f1_values.std()),
    }
    tp = int(np.sum((final_month_preds == 1) & (y_eval == 1)))
    fp = int(np.sum((final_month_preds == 1) & (y_eval == 0)))
    tn = int(np.sum((final_month_preds == 0) & (y_eval == 0)))
    fn = int(np.sum((final_month_preds == 0) & (y_eval == 1)))
    final_month_confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    final_month_pred_counts = {
        "pred_0": int(np.sum(final_month_preds == 0)),
        "pred_1": int(np.sum(final_month_preds == 1)),
    }
    n_eval = max(int(len(y_eval)), 1)
    final_month_pred_rates = {
        "pred_0_rate": float(final_month_pred_counts["pred_0"] / n_eval),
        "pred_1_rate": float(final_month_pred_counts["pred_1"] / n_eval),
    }
    final_month_threshold = float(per_month[-1]["threshold_used"])
    final_month_probs_list = [float(value) for value in final_month_probs.tolist()]

    output_dir = ensure_dir(config.output_dir)
    model_path = output_dir / "eval_temporal_model.pt"
    metrics_path = output_dir / "eval_temporal_metrics.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "gru_hidden_dim": int(config.gru_hidden_dim),
            "gru_layers": int(config.gru_layers),
            "dropout": float(config.dropout),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
            "train_months": int(config.train_months),
        },
        model_path,
    )

    metrics_payload = {
        "model_type": "temporal_transformer_gru_eval",
        "input_path": str(data_path),
        "train_months": int(config.train_months),
        "months_total": int(n_months),
        "per_month": per_month,
        "final_month": final_month,
        "threshold_mode": config.threshold_mode,
        "fixed_threshold": float(config.fixed_threshold),
        "chosen_threshold": float(chosen_threshold),
        "calibration_metric": config.calibration_metric,
        "final_month_threshold": final_month_threshold,
        "final_month_probs": final_month_probs_list,
        "final_month_pred_counts": final_month_pred_counts,
        "final_month_pred_rates": final_month_pred_rates,
        "final_month_confusion": final_month_confusion,
        "per_month_summary": per_month_summary,
        "summary": summary,
        "config": asdict(config),
        "cache_path": str(cache_path) if config.cache_embeddings else None,
        "used_cache": bool(used_cache),
        "cache_fingerprint": cache_fingerprint,
        "n_authors_train": int(len(train_idx)),
        "n_authors_eval": int(len(eval_idx)),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "final_accuracy": float(final_month["accuracy"]),
        "final_f1": float(final_month["f1"]),
        "final_month_index": int(final_month["month_index"]),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "cache_path": str(cache_path) if config.cache_embeddings else "",
        "cache_fingerprint": cache_fingerprint,
        "used_cache": bool(used_cache),
        "threshold_mode": config.threshold_mode,
        "fixed_threshold": float(config.fixed_threshold),
        "chosen_threshold": float(chosen_threshold),
        "calibration_metric": config.calibration_metric,
        "final_month_threshold": final_month_threshold,
        "per_month": per_month,
        "per_month_summary": per_month_summary,
        "final_month_pred_counts": final_month_pred_counts,
        "final_month_pred_rates": final_month_pred_rates,
        "final_month_confusion": final_month_confusion,
        "final_month_probs": final_month_probs_list,
    }
