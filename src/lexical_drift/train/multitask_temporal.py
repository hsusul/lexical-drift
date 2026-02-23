from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lexical_drift.config import TrainMultiTaskConfig
from lexical_drift.datasets.e2e import build_sequence_batch
from lexical_drift.datasets.synthetic import save_synthetic_dataset
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
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for multitask temporal features. "
            'Install with: pip install -e ".[torch]"'
        ) from exc
    return torch, nn


def _import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for ablation plots. Install with: pip install matplotlib"
        ) from exc
    return plt


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
    lengths = sorted({len(sequence) for sequence in sequences_texts})
    if len(lengths) != 1:
        raise ValueError("All authors must have the same number of months for multitask training")
    return authors, sequences_texts, sequences_months, labels.astype(np.int64)


class MultiTaskTemporalHead:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        torch, nn = _require_torch()
        _ = torch
        gru_dropout = dropout if layers > 1 else 0.0
        self.model = nn.ModuleDict(
            {
                "gru": nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=layers,
                    batch_first=True,
                    dropout=gru_dropout,
                ),
                "dropout": nn.Dropout(dropout),
                "cls_head": nn.Linear(hidden_dim, 1),
                "drift_head": nn.Linear(hidden_dim, 1),
            }
        )

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)

    def __call__(self, x):
        _output, hidden = self.model["gru"](x)
        last_hidden = hidden[-1]
        last_hidden = self.model["dropout"](last_hidden)
        cls_logits = self.model["cls_head"](last_hidden)
        drift_pred = self.model["drift_head"](last_hidden)
        return cls_logits, drift_pred


def _drift_target_from_embeddings(embeddings, *, metric: str):
    torch, _nn = _require_torch()
    reference = embeddings[:, 0, :]
    current = embeddings[:, -1, :]
    if metric == "cosine":
        cosine = torch.nn.functional.cosine_similarity(reference, current, dim=1)
        return 1.0 - cosine
    return torch.norm(reference - current, dim=1, p=2)


def _compute_per_month_metrics(
    *,
    encoder: TemporalEncoder,
    head: MultiTaskTemporalHead,
    device,
    author_ids: list[str],
    sequences_texts: list[list[str]],
    sequences_months: list[list[int]],
    labels: np.ndarray,
    eval_indices: np.ndarray,
    train_months: int,
    batch_size: int,
    threshold: float,
    time_embedding,
) -> list[dict[str, float | int | None]]:
    torch, _nn = _require_torch()
    y_eval = labels[eval_indices].astype(int)
    total_months = len(sequences_texts[0])
    per_month: list[dict[str, float | int | None]] = []

    for month_index in range(train_months, total_months):
        probs_parts: list[np.ndarray] = []
        for offset in range(0, int(eval_indices.shape[0]), batch_size):
            batch = eval_indices[offset : offset + batch_size].tolist()
            batch_data = build_sequence_batch(
                author_ids=author_ids,
                sequences_texts=sequences_texts,
                sequences_months=sequences_months,
                labels=labels,
                indices=batch,
                max_months=month_index + 1,
            )
            with torch.no_grad():
                encoder.eval()
                head.eval()
                embeddings = encoder.encode_sequences(batch_data.texts, device=device)
                if time_embedding is not None:
                    positions = torch.from_numpy(batch_data.month_indices).to(
                        device=embeddings.device,
                        dtype=torch.long,
                    )
                    embeddings = embeddings + time_embedding(positions)
                logits, _drift_pred = head(embeddings)
                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy().astype(np.float32)
            probs_parts.append(probs)
        probs = np.concatenate(probs_parts, axis=0)
        preds = (probs >= threshold).astype(int)
        tn = int(np.sum((preds == 0) & (y_eval == 0)))
        fp = int(np.sum((preds == 1) & (y_eval == 0)))
        fn = int(np.sum((preds == 0) & (y_eval == 1)))
        tp = int(np.sum((preds == 1) & (y_eval == 1)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        balanced_accuracy = 0.5 * (recall + specificity)
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
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": float(balanced_accuracy),
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
    return per_month


def run_train_multitask(config: TrainMultiTaskConfig) -> dict[str, object]:
    np.random.seed(config.random_seed)
    torch, _nn = _require_torch()
    torch.manual_seed(config.random_seed)
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
    head = MultiTaskTemporalHead(
        input_dim=encoder.output_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    time_embedding = None
    if config.use_time_embeddings:
        time_embedding = torch.nn.Embedding(total_months, encoder.output_dim).to(device)

    parameters = list(head.parameters()) + list(
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    )
    if time_embedding is not None:
        parameters.extend(time_embedding.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    cls_loss_fn = build_binary_classification_loss(
        loss_type=config.loss_type,
        pos_weight=config.pos_weight,
        focal_gamma=config.focal_gamma,
        device=device,
    )
    drift_loss_fn = torch.nn.MSELoss()

    for epoch in range(config.epochs):
        order = train_idx.copy()
        np.random.default_rng(config.random_seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        for offset in range(0, int(order.shape[0]), config.batch_size):
            batch = order[offset : offset + config.batch_size].tolist()
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
            if time_embedding is not None:
                positions = torch.from_numpy(batch_data.month_indices).to(
                    device=embeddings.device,
                    dtype=torch.long,
                )
                embeddings = embeddings + time_embedding(positions)
            cls_logits, drift_pred = head(embeddings)
            drift_target = _drift_target_from_embeddings(
                embeddings.detach(),
                metric=config.drift_target_metric,
            ).unsqueeze(1)

            cls_loss = cls_loss_fn(cls_logits, labels_tensor)
            drift_loss = drift_loss_fn(drift_pred, drift_target)
            loss = cls_loss + config.drift_lambda * drift_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"[train-multitask] epoch={epoch + 1}/{config.epochs} loss={epoch_loss:.4f}")

    per_month = _compute_per_month_metrics(
        encoder=encoder,
        head=head,
        device=device,
        author_ids=authors,
        sequences_texts=sequences_texts,
        sequences_months=sequences_months,
        labels=labels,
        eval_indices=eval_idx,
        train_months=config.train_months,
        batch_size=config.batch_size,
        threshold=config.threshold,
        time_embedding=time_embedding,
    )
    if not per_month:
        raise ValueError("No evaluation months available for multitask training")

    config_hash = config_sha256(config)[:8]
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(config.output_dir) / f"multitask_{run_stamp}_{config_hash}")
    model_path = output_dir / "multitask_model.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
            "pooling": config.pooling,
            "hidden_dim": int(config.hidden_dim),
            "layers": int(config.layers),
            "dropout": float(config.dropout),
            "train_months": int(config.train_months),
            "model_type": "multitask_gru",
            "drift_lambda": float(config.drift_lambda),
            "drift_target_metric": config.drift_target_metric,
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

    per_month_csv_path = output_dir / "per_month_metrics.csv"
    pd.DataFrame(per_month).to_csv(per_month_csv_path, index=False)
    plot_paths = _save_eval_plots(per_month=per_month, output_dir=output_dir)
    final_month = per_month[-1]

    metrics_payload = {
        "model_type": "multitask_gru",
        "input_path": str(config.input_path),
        "final_month": final_month,
        "per_month": per_month,
        "plot_paths": plot_paths,
        "per_month_csv_path": str(per_month_csv_path),
        "config": asdict(config),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metadata_payload = {
        "seed": int(config.random_seed),
        "encoder_model": config.encoder_model,
        "model_type": "multitask_gru",
        "config_hash": config_sha256(config),
        "dataset_hash": file_sha256(Path(config.input_path)),
        "git_commit_hash": git_commit_hash(),
        "timestamp_iso": datetime.now(UTC).isoformat(),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "per_month_csv_path": str(per_month_csv_path),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return {
        "model_type": "multitask_gru",
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
        "final_balanced_accuracy": float(final_month["balanced_accuracy"]),
    }


def run_ablation_drift_weight(
    *,
    config_template: TrainMultiTaskConfig,
    lambdas: list[float],
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts",
) -> dict[str, object]:
    output_root = ensure_dir(Path(artifact_root) / "ablation_drift_weight")
    data_root = output_root / "data"
    rows: list[dict[str, object]] = []

    for drift_lambda in lambdas:
        final_acc: list[float] = []
        final_f1: list[float] = []
        final_bal_acc: list[float] = []
        for seed in seeds:
            data_path = data_root / f"synth_seed_{seed}.csv"
            save_synthetic_dataset(
                out_path=data_path,
                n_authors=n_authors,
                months=months,
                random_seed=int(seed),
                difficulty=difficulty,
            )
            run_config = replace(
                config_template,
                input_path=str(data_path),
                output_dir=str(
                    output_root / "runs" / f"lambda_{drift_lambda:.3f}" / f"seed_{seed}"
                ),
                random_seed=int(seed),
                drift_lambda=float(drift_lambda),
            )
            result = run_train_multitask(run_config)
            final_acc.append(float(result["final_accuracy"]))
            final_f1.append(float(result["final_f1"]))
            final_bal_acc.append(float(result["final_balanced_accuracy"]))
        rows.append(
            {
                "drift_lambda": float(drift_lambda),
                "accuracy_mean": float(np.mean(final_acc)),
                "f1_mean": float(np.mean(final_f1)),
                "balanced_accuracy_mean": float(np.mean(final_bal_acc)),
            }
        )

    summary_payload = {
        "lambdas": [float(value) for value in lambdas],
        "seeds": [int(seed) for seed in seeds],
        "n_authors": int(n_authors),
        "months": int(months),
        "difficulty": difficulty,
        "rows": rows,
    }
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    plot_path = output_root / "ablation_drift_weight.png"
    plt = _import_matplotlib_pyplot()
    x_values = [float(row["drift_lambda"]) for row in rows]
    acc_values = [float(row["accuracy_mean"]) for row in rows]
    f1_values = [float(row["f1_mean"]) for row in rows]
    bal_values = [float(row["balanced_accuracy_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, acc_values, marker="o", label="accuracy_mean")
    ax.plot(x_values, f1_values, marker="o", label="f1_mean")
    ax.plot(x_values, bal_values, marker="o", label="balanced_accuracy_mean")
    ax.set_xlabel("drift_lambda")
    ax.set_ylabel("metric")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Drift weight ablation")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "rows": rows,
    }
