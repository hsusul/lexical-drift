from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.features.encoder import encode_texts_to_embeddings
from lexical_drift.losses.classification import build_binary_classification_loss
from lexical_drift.training.train_temporal import compute_cache_fingerprint
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, file_sha256, git_commit_hash

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


def _compute_confusion_metrics(
    *,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, float]:
    precision_den = tp + fp
    recall_den = tp + fn
    specificity_den = tn + fp

    precision = float(tp / precision_den) if precision_den > 0 else 0.0
    recall = float(tp / recall_den) if recall_den > 0 else 0.0
    specificity = float(tn / specificity_den) if specificity_den > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy),
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


def _import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for eval-temporal plots. Install with: pip install matplotlib"
        ) from exc

    return plt


def _series_from_per_month(
    per_month: list[dict[str, float | int | None]],
    key: str,
) -> np.ndarray:
    values: list[float] = []
    for entry in per_month:
        raw = entry.get(key)
        values.append(np.nan if raw is None else float(raw))
    return np.asarray(values, dtype=np.float64)


def _save_eval_plots(
    *,
    per_month: list[dict[str, float | int | None]],
    output_dir: Path,
) -> dict[str, str]:
    plt = _import_matplotlib_pyplot()
    months = np.asarray([int(entry["month_index"]) for entry in per_month], dtype=np.int64)

    per_month_metrics_path = output_dir / "per_month_metrics.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric_name in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "pr_auc"):
        values = _series_from_per_month(per_month, metric_name)
        if np.isnan(values).all():
            continue
        ax.plot(months, values, marker="o", label=metric_name)
    ax.set_xlabel("month_index")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-month evaluation metrics")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(per_month_metrics_path, dpi=150)
    plt.close(fig)

    threshold_path = output_dir / "threshold_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, _series_from_per_month(per_month, "threshold_used"), marker="o")
    ax.set_xlabel("month_index")
    ax.set_ylabel("threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Threshold over time")
    fig.tight_layout()
    fig.savefig(threshold_path, dpi=150)
    plt.close(fig)

    pred_rate_path = output_dir / "pred_rate_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        months,
        _series_from_per_month(per_month, "pred_pos_rate"),
        marker="o",
        label="pred_pos_rate",
    )
    ax.plot(
        months,
        _series_from_per_month(per_month, "true_pos_rate"),
        marker="o",
        label="true_pos_rate",
    )
    ax.set_xlabel("month_index")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Prediction rate over time")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(pred_rate_path, dpi=150)
    plt.close(fig)

    drift_path = output_dir / "embedding_drift_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric_name in ("cosine_drift", "l2_drift", "variance_shift"):
        values = _series_from_per_month(per_month, metric_name)
        if np.isnan(values).all():
            continue
        ax.plot(months, values, marker="o", label=metric_name)
    ax.set_xlabel("month_index")
    ax.set_ylabel("drift")
    ax.set_title("Embedding drift over time")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(drift_path, dpi=150)
    plt.close(fig)

    drift_vs_accuracy_delta_path = output_dir / "drift_vs_accuracy_delta.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    accuracy_delta = _series_from_per_month(per_month, "accuracy_delta_from_ref")
    for metric_name in ("cosine_drift", "l2_drift", "variance_shift"):
        drift_values = _series_from_per_month(per_month, metric_name)
        if np.isnan(drift_values).all() or np.isnan(accuracy_delta).all():
            continue
        ax.plot(
            accuracy_delta,
            drift_values,
            marker="o",
            linestyle="-",
            label=metric_name,
        )
    ax.set_xlabel("accuracy_delta_from_ref")
    ax.set_ylabel("drift")
    ax.set_title("Embedding drift vs accuracy delta")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(drift_vs_accuracy_delta_path, dpi=150)
    plt.close(fig)

    return {
        "per_month_metrics_path": str(per_month_metrics_path),
        "threshold_over_time_path": str(threshold_path),
        "pred_rate_over_time_path": str(pred_rate_path),
        "embedding_drift_over_time_path": str(drift_path),
        "drift_vs_accuracy_delta_path": str(drift_vs_accuracy_delta_path),
    }


def _save_attention_over_time_plot(
    *,
    attention_over_time: list[np.ndarray],
    month_indices: list[int],
    output_dir: Path,
) -> str | None:
    if not attention_over_time:
        return None

    plt = _import_matplotlib_pyplot()
    matrix = np.vstack(attention_over_time)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xlabel("source month_index")
    ax.set_ylabel("eval month_index")
    ax.set_yticks(np.arange(len(month_indices)))
    ax.set_yticklabels([str(value) for value in month_indices])
    ax.set_title("Average final-token attention over time")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    output_path = output_dir / "attention_over_time.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _compute_embedding_drift_metrics(
    *,
    reference_month_embeddings: np.ndarray,
    current_month_embeddings: np.ndarray,
) -> dict[str, float]:
    reference_mean = np.asarray(reference_month_embeddings.mean(axis=0), dtype=np.float64)
    current_mean = np.asarray(current_month_embeddings.mean(axis=0), dtype=np.float64)
    diff = reference_mean - current_mean

    ref_norm = float(np.linalg.norm(reference_mean))
    cur_norm = float(np.linalg.norm(current_mean))
    if ref_norm > 0.0 and cur_norm > 0.0:
        cosine_similarity = float(np.dot(reference_mean, current_mean) / (ref_norm * cur_norm))
    else:
        cosine_similarity = 1.0
    cosine_drift = float(1.0 - cosine_similarity)
    l2_drift = float(np.linalg.norm(diff))
    reference_variance = float(np.var(reference_month_embeddings))
    current_variance = float(np.var(current_month_embeddings))
    variance_shift = float(abs(current_variance - reference_variance))
    return {
        "cosine_drift": cosine_drift,
        "l2_drift": l2_drift,
        "variance_shift": variance_shift,
    }


def run_eval_temporal(config: EvalTemporalConfig) -> dict[str, object]:
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

    input_dim = int(embeddings.shape[2])
    y_eval = labels[eval_idx].astype(int)
    output_dir = ensure_dir(config.output_dir)

    if config.model_type in {"gru", "attention", "transformer"}:
        torch, DataLoader, TensorDataset = _import_torch_modules()
        if config.model_type == "gru":
            from lexical_drift.models.temporal_gru import build_temporal_gru
        elif config.model_type == "attention":
            from lexical_drift.models.temporal_attention import build_temporal_attention
        else:
            from lexical_drift.models.temporal_transformer import build_temporal_transformer

        torch.manual_seed(config.random_seed)
        X_train = embeddings[train_idx, : config.train_months, :]
        y_train = labels[train_idx].astype(np.float32)

        train_month_indices = months_matrix[train_idx, : config.train_months].astype(np.int64)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(train_month_indices),
            torch.from_numpy(y_train).unsqueeze(1),
        )
        train_generator = torch.Generator().manual_seed(config.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=train_generator,
        )

        if config.model_type == "gru":
            model = build_temporal_gru(
                input_dim=input_dim,
                hidden_dim=config.gru_hidden_dim,
                layers=config.gru_layers,
                dropout=config.dropout,
            )
        elif config.model_type == "attention":
            model = build_temporal_attention(
                input_dim=input_dim,
                hidden_dim=config.gru_hidden_dim,
                max_positions=n_months,
                layers=max(int(config.gru_layers), 1),
                heads=4,
                dropout=config.dropout,
            )
        else:
            model = build_temporal_transformer(
                input_dim=input_dim,
                hidden_dim=config.gru_hidden_dim,
                max_positions=n_months,
                layers=max(int(config.gru_layers), 1),
                heads=4,
                dropout=config.dropout,
                use_time_embeddings=config.use_time_embeddings,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = build_binary_classification_loss(
            loss_type=config.loss_type,
            pos_weight=config.pos_weight,
            focal_gamma=config.focal_gamma,
            device=torch.device("cpu"),
        )

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_count = 0
            for batch_x, batch_month_idx, batch_y in train_loader:
                optimizer.zero_grad()
                if config.model_type == "transformer":
                    logits = model(batch_x, month_indices=batch_month_idx)
                else:
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
        model_path = output_dir / "eval_temporal_model.pt"
        model_payload: dict[str, object] = {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "gru_hidden_dim": int(config.gru_hidden_dim),
            "gru_layers": int(config.gru_layers),
            "dropout": float(config.dropout),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
            "train_months": int(config.train_months),
            "model_type": config.model_type,
            "loss_type": config.loss_type,
            "pos_weight": config.pos_weight,
            "focal_gamma": float(config.focal_gamma),
        }
        if config.model_type in {"attention", "transformer"}:
            model_payload["attention_layers"] = int(max(config.gru_layers, 1))
            model_payload["attention_heads"] = 4
            model_payload["max_positions"] = int(n_months)
            model_payload["use_time_embeddings"] = bool(config.use_time_embeddings)
        torch.save(model_payload, model_path)

        def predict_probs(month_index: int) -> tuple[np.ndarray, np.ndarray | None]:
            with torch.no_grad():
                X_eval = embeddings[eval_idx, : month_index + 1, :]
                month_indices_eval = months_matrix[eval_idx, : month_index + 1].astype(np.int64)
                if config.model_type == "transformer":
                    logits, attention_layers = model(
                        torch.from_numpy(X_eval),
                        return_attention=True,
                        month_indices=torch.from_numpy(month_indices_eval),
                    )
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
                    if attention_layers:
                        # last layer attention: [batch, heads, tgt_len, src_len]
                        attn_last = attention_layers[-1][:, :, -1, :]
                        attn_summary = (
                            attn_last.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
                        )
                    else:
                        attn_summary = None
                    return probs, attn_summary
                logits = model(torch.from_numpy(X_eval))
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return probs.astype(np.float32), None

    elif config.model_type == "baseline_lr":
        X_train_flat = embeddings[train_idx, : config.train_months, :].reshape(-1, input_dim)
        y_train_flat = np.repeat(labels[train_idx].astype(int), config.train_months)
        baseline_model: Pipeline | None
        constant_prob: float | None
        if y_train_flat.size == 0:
            constant_prob = 0.0
            baseline_model = None
        elif np.unique(y_train_flat).size < 2:
            constant_prob = float(y_train_flat[0])
            baseline_model = None
            print(
                "[eval-temporal] baseline_lr detected single-class training data; "
                "using constant probabilities"
            )
        else:
            baseline_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            random_state=config.random_seed,
                            max_iter=1000,
                        ),
                    ),
                ]
            )
            baseline_model.fit(X_train_flat, y_train_flat)
            constant_prob = None

        model_path = output_dir / "eval_temporal_model.joblib"
        joblib.dump(
            {
                "model_type": config.model_type,
                "model": baseline_model,
                "constant_prob": constant_prob,
                "input_dim": input_dim,
                "encoder_model": config.encoder_model,
                "max_length": int(config.max_length),
                "train_months": int(config.train_months),
            },
            model_path,
        )

        def predict_probs(month_index: int) -> tuple[np.ndarray, np.ndarray | None]:
            X_eval = embeddings[eval_idx, month_index, :]
            if baseline_model is None:
                return np.full(X_eval.shape[0], float(constant_prob), dtype=np.float32), None
            probs = baseline_model.predict_proba(X_eval)[:, 1]
            return np.asarray(probs, dtype=np.float32), None

    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    per_month: list[dict[str, float | int | None]] = []
    final_month_probs: np.ndarray | None = None
    final_month_preds: np.ndarray | None = None
    attention_over_time: list[np.ndarray] = []
    attention_month_indices: list[int] = []
    chosen_threshold = float(config.fixed_threshold)
    calibrate_first_mode = config.threshold_mode == "calibrate_first_eval"
    calibrate_each_mode = config.threshold_mode == "calibrate_each_month"
    reference_month_embeddings = embeddings[eval_idx, config.train_months - 1, :]
    for month_index in range(config.train_months, n_months):
        probs, month_attention = predict_probs(month_index)
        if month_attention is not None:
            padded_attention = np.full(n_months, np.nan, dtype=np.float64)
            padded_attention[: month_index + 1] = month_attention.astype(np.float64)
            attention_over_time.append(padded_attention)
            attention_month_indices.append(int(month_index))
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
        confusion_metrics = _compute_confusion_metrics(tn=tn, fp=fp, fn=fn, tp=tp)
        current_month_embeddings = embeddings[eval_idx, month_index, :]
        drift_metrics = _compute_embedding_drift_metrics(
            reference_month_embeddings=reference_month_embeddings,
            current_month_embeddings=current_month_embeddings,
        )
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
                "precision": confusion_metrics["precision"],
                "recall": confusion_metrics["recall"],
                "specificity": confusion_metrics["specificity"],
                "balanced_accuracy": confusion_metrics["balanced_accuracy"],
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
                "cosine_drift": drift_metrics["cosine_drift"],
                "l2_drift": drift_metrics["l2_drift"],
                "variance_shift": drift_metrics["variance_shift"],
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

    # Reference is first evaluation month (month_index=train_months).
    reference_accuracy = float(per_month[0]["accuracy"])
    reference_f1 = float(per_month[0]["f1"])
    for entry in per_month:
        entry["accuracy_delta_from_ref"] = float(entry["accuracy"]) - reference_accuracy
        entry["f1_delta_from_ref"] = float(entry["f1"]) - reference_f1

    final_month = per_month[-1]
    accuracy_values = np.asarray(
        [float(entry["accuracy"]) for entry in per_month], dtype=np.float64
    )
    f1_values = np.asarray([float(entry["f1"]) for entry in per_month], dtype=np.float64)
    cosine_drift_values = np.asarray(
        [float(entry["cosine_drift"]) for entry in per_month], dtype=np.float64
    )
    l2_drift_values = np.asarray(
        [float(entry["l2_drift"]) for entry in per_month], dtype=np.float64
    )
    variance_shift_values = np.asarray(
        [float(entry["variance_shift"]) for entry in per_month], dtype=np.float64
    )
    accuracy_delta_values = np.asarray(
        [float(entry["accuracy_delta_from_ref"]) for entry in per_month], dtype=np.float64
    )
    f1_delta_values = np.asarray(
        [float(entry["f1_delta_from_ref"]) for entry in per_month], dtype=np.float64
    )
    per_month_summary = {
        "accuracy_min": float(accuracy_values.min()),
        "accuracy_mean": float(accuracy_values.mean()),
        "accuracy_max": float(accuracy_values.max()),
        "f1_min": float(f1_values.min()),
        "f1_mean": float(f1_values.mean()),
        "f1_max": float(f1_values.max()),
        "cosine_drift_min": float(cosine_drift_values.min()),
        "cosine_drift_mean": float(cosine_drift_values.mean()),
        "cosine_drift_max": float(cosine_drift_values.max()),
        "l2_drift_min": float(l2_drift_values.min()),
        "l2_drift_mean": float(l2_drift_values.mean()),
        "l2_drift_max": float(l2_drift_values.max()),
        "variance_shift_min": float(variance_shift_values.min()),
        "variance_shift_mean": float(variance_shift_values.mean()),
        "variance_shift_max": float(variance_shift_values.max()),
        "accuracy_delta_from_ref_min": float(accuracy_delta_values.min()),
        "accuracy_delta_from_ref_mean": float(accuracy_delta_values.mean()),
        "accuracy_delta_from_ref_max": float(accuracy_delta_values.max()),
        "f1_delta_from_ref_min": float(f1_delta_values.min()),
        "f1_delta_from_ref_mean": float(f1_delta_values.mean()),
        "f1_delta_from_ref_max": float(f1_delta_values.max()),
    }
    summary = {
        "accuracy_mean": float(accuracy_values.mean()),
        "accuracy_std": float(accuracy_values.std()),
        "f1_mean": float(f1_values.mean()),
        "f1_std": float(f1_values.std()),
        "cosine_drift_mean": float(cosine_drift_values.mean()),
        "cosine_drift_std": float(cosine_drift_values.std()),
        "l2_drift_mean": float(l2_drift_values.mean()),
        "l2_drift_std": float(l2_drift_values.std()),
        "variance_shift_mean": float(variance_shift_values.mean()),
        "variance_shift_std": float(variance_shift_values.std()),
        "accuracy_delta_from_ref_mean": float(accuracy_delta_values.mean()),
        "accuracy_delta_from_ref_std": float(accuracy_delta_values.std()),
        "f1_delta_from_ref_mean": float(f1_delta_values.mean()),
        "f1_delta_from_ref_std": float(f1_delta_values.std()),
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
    metrics_path = output_dir / "eval_temporal_metrics.json"
    per_month_csv_path = output_dir / "per_month_metrics.csv"
    run_metadata_path = output_dir / "run_metadata.json"
    pd.DataFrame(per_month).to_csv(per_month_csv_path, index=False)
    plot_paths = _save_eval_plots(per_month=per_month, output_dir=output_dir)
    attention_plot_path = _save_attention_over_time_plot(
        attention_over_time=attention_over_time,
        month_indices=attention_month_indices,
        output_dir=output_dir,
    )
    if attention_plot_path is not None:
        plot_paths["attention_over_time_path"] = attention_plot_path
    dataset_hash = file_sha256(data_path)
    config_hash = config_sha256(config)
    commit_hash = git_commit_hash()
    timestamp_iso = datetime.now(UTC).isoformat()

    metrics_payload = {
        "model_type": config.model_type,
        "use_time_embeddings": bool(config.use_time_embeddings),
        "loss_type": config.loss_type,
        "pos_weight": config.pos_weight,
        "focal_gamma": float(config.focal_gamma),
        "input_path": str(data_path),
        "git_commit_hash": commit_hash,
        "timestamp_iso": timestamp_iso,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
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
        "per_month_csv_path": str(per_month_csv_path),
        "plot_paths": plot_paths,
        "run_metadata_path": str(run_metadata_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    run_metadata = {
        "seed": int(config.random_seed),
        "model_type": config.model_type,
        "encoder_model": config.encoder_model,
        "max_length": int(config.max_length),
        "train_months": int(config.train_months),
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "cache_fingerprint": cache_fingerprint,
        "git_commit_hash": commit_hash,
        "timestamp_iso": timestamp_iso,
        "input_path": str(data_path),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "per_month_csv_path": str(per_month_csv_path),
        "model_path": str(model_path),
        "plot_paths": plot_paths,
    }
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    return {
        "model_type": config.model_type,
        "use_time_embeddings": bool(config.use_time_embeddings),
        "loss_type": config.loss_type,
        "pos_weight": config.pos_weight,
        "focal_gamma": float(config.focal_gamma),
        "final_accuracy": float(final_month["accuracy"]),
        "final_f1": float(final_month["f1"]),
        "final_month_index": int(final_month["month_index"]),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "cache_path": str(cache_path) if config.cache_embeddings else "",
        "cache_fingerprint": cache_fingerprint,
        "git_commit_hash": commit_hash,
        "timestamp_iso": timestamp_iso,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
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
        "plot_paths": plot_paths,
        "per_month_csv_path": str(per_month_csv_path),
        "run_metadata_path": str(run_metadata_path),
    }
