from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.datasets.e2e import build_sequence_batch
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.eval.eval_temporal import _save_eval_plots
from lexical_drift.losses.classification import build_binary_classification_loss
from lexical_drift.models.temporal_encoder import TemporalEncoder
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, file_sha256, git_commit_hash

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required for e2e temporal features. Install with: pip install -e ".[torch]"'
        ) from exc
    return torch


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)


def _validate_equal_sequence_lengths(sequences_texts: list[list[str]]) -> int:
    lengths = sorted({len(sequence) for sequence in sequences_texts})
    if len(lengths) != 1:
        raise ValueError("All authors must have the same number of months for temporal evaluation")
    return lengths[0]


def _prepare_dataset(
    path: Path,
) -> tuple[list[str], list[list[str]], list[list[int]], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    authors, sequences_texts, sequences_months, labels = build_author_sequences_with_months(frame)
    _validate_equal_sequence_lengths(sequences_texts)
    return authors, sequences_texts, sequences_months, labels.astype(np.int64)


def _batch_indices(
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    values = indices.astype(np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(values)
    batches: list[list[int]] = []
    for offset in range(0, int(values.shape[0]), batch_size):
        batch = values[offset : offset + batch_size]
        batches.append(batch.tolist())
    return batches


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return tn, fp, fn, tp


def _compute_confusion_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
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


def _build_time_embedding(
    *,
    torch,
    enabled: bool,
    max_positions: int,
    embedding_dim: int,
    device,
):
    if not enabled:
        return None
    return torch.nn.Embedding(max_positions, embedding_dim).to(device)


def _apply_time_embeddings(
    *,
    embeddings,
    time_embedding,
    month_indices=None,
):
    if time_embedding is None:
        return embeddings
    torch = _require_torch()
    if month_indices is None:
        sequence_length = int(embeddings.shape[1])
        positions = torch.arange(sequence_length, device=embeddings.device)
        positions = positions.unsqueeze(0).expand(int(embeddings.shape[0]), sequence_length)
    else:
        positions = month_indices.to(device=embeddings.device, dtype=torch.long)
    return embeddings + time_embedding(positions)


def _predict_probs_for_month(
    *,
    encoder: TemporalEncoder,
    head,
    device,
    author_ids: list[str],
    sequences_texts: list[list[str]],
    labels: np.ndarray,
    sequences_months: list[list[int]],
    eval_indices: np.ndarray,
    month_index: int,
    batch_size: int,
    time_embedding,
) -> np.ndarray:
    torch = _require_torch()
    probs_parts: list[np.ndarray] = []
    head.eval()
    encoder.eval()
    for batch in _batch_indices(
        eval_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        batch_data = build_sequence_batch(
            author_ids=author_ids,
            sequences_texts=sequences_texts,
            sequences_months=sequences_months,
            labels=labels,
            indices=batch,
            max_months=month_index + 1,
        )
        with torch.no_grad():
            embeddings = encoder.encode_sequences(batch_data.texts, device=device)
            embeddings = _apply_time_embeddings(
                embeddings=embeddings,
                time_embedding=time_embedding,
                month_indices=torch.from_numpy(batch_data.month_indices),
            )
            logits = head(embeddings)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy().astype(np.float32)
        probs_parts.append(probs)
    if not probs_parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(probs_parts, axis=0)


def _evaluate_e2e(
    *,
    encoder: TemporalEncoder,
    head,
    device,
    author_ids: list[str],
    sequences_texts: list[list[str]],
    labels: np.ndarray,
    sequences_months: list[list[int]],
    eval_indices: np.ndarray,
    train_months: int,
    batch_size: int,
    threshold: float,
    time_embedding,
) -> list[dict[str, float | int | None]]:
    y_eval = labels[eval_indices].astype(int)
    total_months = len(sequences_texts[0])
    per_month: list[dict[str, float | int | None]] = []

    for month_index in range(train_months, total_months):
        probs = _predict_probs_for_month(
            encoder=encoder,
            head=head,
            device=device,
            author_ids=author_ids,
            sequences_texts=sequences_texts,
            labels=labels,
            sequences_months=sequences_months,
            eval_indices=eval_indices,
            month_index=month_index,
            batch_size=batch_size,
            time_embedding=time_embedding,
        )
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = _confusion_counts(y_eval, preds)
        confusion = _compute_confusion_metrics(tn=tn, fp=fp, fn=fn, tp=tp)
        roc_auc: float | None = None
        pr_auc: float | None = None
        if np.unique(y_eval).size > 1:
            roc_auc = float(roc_auc_score(y_eval, probs))
            pr_auc = float(average_precision_score(y_eval, probs))
        per_month.append(
            {
                "month_index": int(month_index),
                "accuracy": float(accuracy_score(y_eval, preds)),
                "f1": float(f1_score(y_eval, preds, zero_division=0)),
                "precision": confusion["precision"],
                "recall": confusion["recall"],
                "specificity": confusion["specificity"],
                "balanced_accuracy": confusion["balanced_accuracy"],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "true_pos_rate": float(np.mean(y_eval)),
                "pred_pos_rate": float(np.mean(preds)),
                "mean_pred_prob": float(np.mean(probs)),
                "threshold_used": float(threshold),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    if not per_month:
        raise ValueError("No evaluation months available. Reduce train_months or increase months.")
    return per_month


def _save_eval_outputs(
    *,
    output_dir: Path,
    config_obj: object,
    input_path: Path,
    model_path: Path,
    per_month: list[dict[str, float | int | None]],
    seed: int,
    model_type: str,
) -> tuple[Path, Path, Path, dict[str, str]]:
    per_month_csv_path = output_dir / "per_month_metrics.csv"
    pd.DataFrame(per_month).to_csv(per_month_csv_path, index=False)
    plot_paths = _save_eval_plots(per_month=per_month, output_dir=output_dir)

    final_month = per_month[-1]
    metrics_payload = {
        "model_type": model_type,
        "input_path": str(input_path),
        "final_month": final_month,
        "per_month": per_month,
        "plot_paths": plot_paths,
        "per_month_csv_path": str(per_month_csv_path),
        "config": asdict(config_obj),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metadata_payload = {
        "seed": int(seed),
        "encoder_model": str(getattr(config_obj, "encoder_model", "")),
        "model_type": model_type,
        "config_hash": config_sha256(config_obj),
        "dataset_hash": file_sha256(input_path),
        "git_commit_hash": git_commit_hash(),
        "timestamp_iso": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "per_month_csv_path": str(per_month_csv_path),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return metrics_path, metadata_path, per_month_csv_path, plot_paths


def run_train_e2e(config: TrainE2EConfig) -> dict[str, object]:
    _set_seed(config.random_seed)
    torch = _require_torch()
    from lexical_drift.models.temporal_gru import build_temporal_gru

    device = torch.device("cpu")

    authors, sequences_texts, sequences_months, labels = _prepare_dataset(Path(config.input_path))
    total_months = len(sequences_texts[0])
    if config.train_months >= total_months:
        raise ValueError(f"train_months must be < total months ({total_months})")

    indices = np.arange(len(authors), dtype=np.int64)
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify,
    )

    encoder = TemporalEncoder(
        model_name=config.encoder_model,
        max_length=config.max_length,
        pooling=config.pooling,
        freeze=config.freeze_encoder,
    ).to(device)
    if config.pretrained_encoder_path:
        checkpoint = torch.load(config.pretrained_encoder_path, map_location=device)
        state_dict = checkpoint.get("encoder_state_dict")
        if isinstance(state_dict, dict):
            encoder.load_state_dict(state_dict, strict=False)

    head = build_temporal_gru(
        input_dim=encoder.output_dim,
        hidden_dim=config.gru_hidden_dim,
        layers=config.gru_layers,
        dropout=config.dropout,
    ).to(device)
    time_embedding = _build_time_embedding(
        torch=torch,
        enabled=config.use_time_embeddings,
        max_positions=total_months,
        embedding_dim=encoder.output_dim,
        device=device,
    )

    parameters = list(head.parameters()) + list(
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    )
    if time_embedding is not None:
        parameters.extend(time_embedding.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    criterion = build_binary_classification_loss(
        loss_type=config.loss_type,
        pos_weight=config.pos_weight,
        focal_gamma=config.focal_gamma,
        device=device,
    )

    for epoch in range(config.epochs):
        head.train()
        encoder.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch in _batch_indices(
            train_idx,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.random_seed + epoch,
        ):
            batch_data = build_sequence_batch(
                author_ids=authors,
                sequences_texts=sequences_texts,
                sequences_months=sequences_months,
                labels=labels,
                indices=batch,
                max_months=config.train_months,
            )
            labels_tensor = (
                torch.from_numpy(batch_data.labels)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(1)
            )
            embeddings = encoder.encode_sequences(batch_data.texts, device=device)
            embeddings = _apply_time_embeddings(
                embeddings=embeddings,
                time_embedding=time_embedding,
                month_indices=torch.from_numpy(batch_data.month_indices),
            )
            logits = head(embeddings)
            loss = criterion(logits, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(batch)
            epoch_loss += float(loss.item()) * batch_size
            epoch_count += batch_size

        avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"[train-e2e] epoch={epoch + 1}/{config.epochs} train_loss={avg_loss:.4f}")

    config_hash = config_sha256(config)[:8]
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config.output_dir) / f"train_e2e_{run_stamp}_{config_hash}")
    model_path = output_dir / "e2e_model.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "encoder_model": config.encoder_model,
            "pooling": config.pooling,
            "max_length": int(config.max_length),
            "gru_hidden_dim": int(config.gru_hidden_dim),
            "gru_layers": int(config.gru_layers),
            "dropout": float(config.dropout),
            "train_months": int(config.train_months),
            "model_type": "e2e_gru",
            "use_time_embeddings": bool(config.use_time_embeddings),
            "time_embedding_state_dict": (
                time_embedding.state_dict() if time_embedding is not None else None
            ),
            "loss_type": config.loss_type,
            "pos_weight": config.pos_weight,
            "focal_gamma": float(config.focal_gamma),
        },
        model_path,
    )

    per_month = _evaluate_e2e(
        encoder=encoder,
        head=head,
        device=device,
        author_ids=authors,
        sequences_texts=sequences_texts,
        labels=labels,
        sequences_months=sequences_months,
        eval_indices=eval_idx,
        train_months=config.train_months,
        batch_size=config.batch_size,
        threshold=0.5,
        time_embedding=time_embedding,
    )
    metrics_path, metadata_path, per_month_csv_path, plot_paths = _save_eval_outputs(
        output_dir=output_dir,
        config_obj=config,
        input_path=Path(config.input_path),
        model_path=model_path,
        per_month=per_month,
        seed=config.random_seed,
        model_type="e2e_gru",
    )
    final_month = per_month[-1]
    return {
        "model_type": "e2e_gru",
        "use_time_embeddings": bool(config.use_time_embeddings),
        "loss_type": config.loss_type,
        "pos_weight": config.pos_weight,
        "focal_gamma": float(config.focal_gamma),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "run_metadata_path": str(metadata_path),
        "per_month_csv_path": str(per_month_csv_path),
        "plot_paths": plot_paths,
        "final_month_index": int(final_month["month_index"]),
        "final_accuracy": float(final_month["accuracy"]),
        "final_f1": float(final_month["f1"]),
    }


def run_eval_e2e(config: EvalE2EConfig) -> dict[str, object]:
    _set_seed(config.random_seed)
    torch = _require_torch()
    from lexical_drift.models.temporal_gru import build_temporal_gru

    device = torch.device("cpu")

    authors, sequences_texts, sequences_months, labels = _prepare_dataset(Path(config.input_path))
    total_months = len(sequences_texts[0])
    if config.train_months >= total_months:
        raise ValueError(f"train_months must be < total months ({total_months})")

    indices = np.arange(len(authors), dtype=np.int64)
    stratify = labels if len(np.unique(labels)) > 1 else None
    _train_idx, eval_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify,
    )

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    encoder = TemporalEncoder(
        model_name=config.encoder_model,
        max_length=config.max_length,
        pooling=config.pooling,
        freeze=True,
    ).to(device)
    if config.pretrained_encoder_path:
        pretrained = torch.load(config.pretrained_encoder_path, map_location=device)
        pretrained_state = pretrained.get("encoder_state_dict")
        if isinstance(pretrained_state, dict):
            encoder.load_state_dict(pretrained_state, strict=False)
    encoder_state = checkpoint.get("encoder_state_dict")
    if isinstance(encoder_state, dict):
        encoder.load_state_dict(encoder_state, strict=False)

    head = build_temporal_gru(
        input_dim=encoder.output_dim,
        hidden_dim=int(checkpoint.get("gru_hidden_dim", 128)),
        layers=int(checkpoint.get("gru_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    head_state = checkpoint.get("head_state_dict")
    if isinstance(head_state, dict):
        head.load_state_dict(head_state, strict=False)
    time_embedding = None
    checkpoint_loss_type = str(checkpoint.get("loss_type", "bce"))
    checkpoint_pos_weight = checkpoint.get("pos_weight")
    checkpoint_focal_gamma = float(checkpoint.get("focal_gamma", 2.0))
    if bool(checkpoint.get("use_time_embeddings", False)):
        time_state = checkpoint.get("time_embedding_state_dict")
        if isinstance(time_state, dict) and "weight" in time_state:
            max_positions, embedding_dim = map(int, time_state["weight"].shape)
            time_embedding = _build_time_embedding(
                torch=torch,
                enabled=True,
                max_positions=max_positions,
                embedding_dim=embedding_dim,
                device=device,
            )
            if time_embedding is not None:
                time_embedding.load_state_dict(time_state)

    config_hash = config_sha256(config)[:8]
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config.output_dir) / f"eval_e2e_{run_stamp}_{config_hash}")
    per_month = _evaluate_e2e(
        encoder=encoder,
        head=head,
        device=device,
        author_ids=authors,
        sequences_texts=sequences_texts,
        labels=labels,
        sequences_months=sequences_months,
        eval_indices=eval_idx,
        train_months=config.train_months,
        batch_size=config.batch_size,
        threshold=config.threshold,
        time_embedding=time_embedding,
    )

    model_copy_path = output_dir / "e2e_model.pt"
    torch.save(checkpoint, model_copy_path)
    metrics_path, metadata_path, per_month_csv_path, plot_paths = _save_eval_outputs(
        output_dir=output_dir,
        config_obj=config,
        input_path=Path(config.input_path),
        model_path=model_copy_path,
        per_month=per_month,
        seed=config.random_seed,
        model_type="e2e_gru",
    )
    final_month = per_month[-1]
    return {
        "model_type": "e2e_gru",
        "use_time_embeddings": bool(time_embedding is not None),
        "loss_type": checkpoint_loss_type,
        "pos_weight": checkpoint_pos_weight,
        "focal_gamma": float(checkpoint_focal_gamma),
        "output_dir": str(output_dir),
        "model_path": str(model_copy_path),
        "metrics_path": str(metrics_path),
        "run_metadata_path": str(metadata_path),
        "per_month_csv_path": str(per_month_csv_path),
        "plot_paths": plot_paths,
        "final_month_index": int(final_month["month_index"]),
        "final_accuracy": float(final_month["accuracy"]),
        "final_f1": float(final_month["f1"]),
    }
