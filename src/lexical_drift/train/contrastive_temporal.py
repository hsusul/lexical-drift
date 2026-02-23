from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import PretrainContrastiveConfig
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.losses.infonce import info_nce_loss
from lexical_drift.models.temporal_encoder import TemporalEncoder
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, file_sha256, git_commit_hash

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError('PyTorch is required. Install with: pip install -e ".[dl]"') from exc
    return torch


def _prepare_positive_pairs(
    *,
    data_path: Path,
    train_months: int,
) -> list[tuple[str, str]]:
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
    for sequence in sequences_texts:
        upper = min(max(len(sequence) - 1, 0), train_months)
        for month_index in range(upper):
            pairs.append((sequence[month_index], sequence[month_index + 1]))
    if len(pairs) < 2:
        raise ValueError("Need at least two positive pairs for contrastive pretraining")
    return pairs


def run_pretrain_contrastive(config: PretrainContrastiveConfig) -> dict[str, object]:
    np.random.seed(config.random_seed)
    torch = _import_torch()
    torch.manual_seed(config.random_seed)
    device = torch.device("cpu")

    data_path = Path(config.input_path)
    pairs = _prepare_positive_pairs(data_path=data_path, train_months=config.train_months)
    encoder = TemporalEncoder(
        model_name=config.encoder_model,
        max_length=config.max_length,
        pooling=config.pooling,
        freeze=config.freeze_encoder,
    ).to(device)
    projector = torch.nn.Sequential(
        torch.nn.Linear(encoder.output_dim, config.projection_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(config.projection_dim, config.projection_dim),
    ).to(device)
    parameters = list(projector.parameters()) + list(
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(parameters, lr=config.lr)

    rng = np.random.default_rng(config.random_seed)
    epoch_losses: list[float] = []
    for epoch in range(config.epochs):
        order = rng.permutation(len(pairs))
        batch_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_ids = order[start : start + config.batch_size]
            if len(batch_ids) < 2:
                continue
            anchors = [pairs[index][0] for index in batch_ids]
            positives = [pairs[index][1] for index in batch_ids]
            anchor_emb = encoder.encode_texts(anchors, device=device)
            positive_emb = encoder.encode_texts(positives, device=device)
            loss = info_nce_loss(
                projector(anchor_emb),
                projector(positive_emb),
                temperature=config.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        print(f"[pretrain-contrastive] epoch={epoch + 1}/{config.epochs} loss={mean_loss:.4f}")

    config_hash = config_sha256(config)[:8]
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config.output_dir) / f"contrastive_{run_stamp}_{config_hash}")
    checkpoint_path = output_dir / "encoder_checkpoint.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
            "pooling": config.pooling,
            "projection_dim": int(config.projection_dim),
            "model_type": "contrastive_temporal",
        },
        checkpoint_path,
    )

    metrics_payload = {
        "model_type": "contrastive_temporal",
        "input_path": str(data_path),
        "n_pairs": int(len(pairs)),
        "epoch_losses": epoch_losses,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "config": asdict(config),
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path = output_dir / "pretrain_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metadata_payload = {
        "seed": int(config.random_seed),
        "encoder_model": config.encoder_model,
        "model_type": "contrastive_temporal",
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
        "n_pairs": int(len(pairs)),
    }
