from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import PretrainTemporalOrderConfig
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.models.temporal_encoder import TemporalEncoder
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, file_sha256, git_commit_hash

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for temporal order pretraining. "
            'Install with: pip install -e ".[torch]"'
        ) from exc
    return torch


def _prepare_order_examples(
    *,
    data_path: Path,
    train_months: int,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")
    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    _authors, sequences_texts, _sequences_months, _labels = build_author_sequences_with_months(
        frame
    )
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    for sequence in sequences_texts:
        upper = min(max(len(sequence) - 1, 0), train_months)
        for month_index in range(upper):
            current = sequence[month_index]
            nxt = sequence[month_index + 1]
            pairs.append((current, nxt))
            labels.append(1)
            pairs.append((nxt, current))
            labels.append(0)
    if len(pairs) < 2:
        raise ValueError("Need at least two temporal order examples")
    return pairs, np.asarray(labels, dtype=np.float32)


def run_pretrain_temporal_order(config: PretrainTemporalOrderConfig) -> dict[str, object]:
    np.random.seed(config.random_seed)
    torch = _require_torch()
    torch.manual_seed(config.random_seed)
    device = torch.device("cpu")

    data_path = Path(config.input_path)
    pairs, labels = _prepare_order_examples(data_path=data_path, train_months=config.train_months)
    encoder = TemporalEncoder(
        model_name=config.encoder_model,
        max_length=config.max_length,
        pooling=config.pooling,
        freeze=config.freeze_encoder,
    ).to(device)

    classifier = torch.nn.Sequential(
        torch.nn.Linear(encoder.output_dim * 2, config.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden_dim, 1),
    ).to(device)

    parameters = list(classifier.parameters()) + list(
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(config.random_seed)
    epoch_losses: list[float] = []
    epoch_accuracies: list[float] = []
    for epoch in range(config.epochs):
        order = rng.permutation(len(pairs))
        batch_losses: list[float] = []
        all_preds: list[int] = []
        all_labels: list[int] = []
        for start in range(0, len(order), config.batch_size):
            batch_ids = order[start : start + config.batch_size]
            left_texts = [pairs[index][0] for index in batch_ids]
            right_texts = [pairs[index][1] for index in batch_ids]
            y_batch = labels[batch_ids]
            y_tensor = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32).unsqueeze(1)

            left_emb = encoder.encode_texts(left_texts, device=device)
            right_emb = encoder.encode_texts(right_texts, device=device)
            features = torch.cat([left_emb, right_emb], dim=1)
            logits = classifier(features)
            loss = criterion(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))
            preds = (torch.sigmoid(logits).detach().cpu().numpy() >= 0.5).astype(int).reshape(-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.astype(int).tolist())
        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        mean_acc = (
            float(np.mean(np.asarray(all_preds) == np.asarray(all_labels))) if all_preds else 0.0
        )
        epoch_losses.append(mean_loss)
        epoch_accuracies.append(mean_acc)
        print(
            "[pretrain-temporal-order] "
            f"epoch={epoch + 1}/{config.epochs} "
            f"loss={mean_loss:.4f} "
            f"acc={mean_acc:.4f}"
        )

    config_hash = config_sha256(config)[:8]
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config.output_dir) / f"temporal_order_{run_stamp}_{config_hash}")
    checkpoint_path = output_dir / "encoder_checkpoint.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "order_classifier_state_dict": classifier.state_dict(),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
            "pooling": config.pooling,
            "hidden_dim": int(config.hidden_dim),
            "model_type": "temporal_order_pretraining",
        },
        checkpoint_path,
    )

    metrics_payload = {
        "model_type": "temporal_order_pretraining",
        "input_path": str(data_path),
        "n_examples": int(len(pairs)),
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "final_accuracy": float(epoch_accuracies[-1]) if epoch_accuracies else 0.0,
        "config": asdict(config),
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path = output_dir / "pretrain_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metadata_payload = {
        "seed": int(config.random_seed),
        "encoder_model": config.encoder_model,
        "model_type": "temporal_order_pretraining",
        "config_hash": config_sha256(config),
        "dataset_hash": file_sha256(data_path),
        "git_commit_hash": git_commit_hash(),
        "timestamp_iso": datetime.now(UTC).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "run_metadata_path": str(metadata_path),
        "final_loss": float(metrics_payload["final_loss"]),
        "final_accuracy": float(metrics_payload["final_accuracy"]),
        "n_examples": int(len(pairs)),
    }
