from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from lexical_drift.config import TemporalTrainConfig
from lexical_drift.datasets.temporal import build_author_sequences_with_months
from lexical_drift.features.encoder import encode_texts_to_embeddings
from lexical_drift.utils import ensure_dir

SMALL_FILE_THRESHOLD_BYTES = 2 * 1024 * 1024
SAMPLED_HASH_BYTES = 64 * 1024


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


def compute_dataset_fingerprint(path: Path) -> str:
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}".encode())

    size = int(stat.st_size)
    if size <= SMALL_FILE_THRESHOLD_BYTES:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    # Large-file branch: hash only prefix/suffix samples (plus size/mtime above).
    sample_size = min(SAMPLED_HASH_BYTES, size)
    with path.open("rb") as f:
        # Prefix sample
        hasher.update(f.read(sample_size))
        f.seek(max(size - sample_size, 0))
        # Suffix sample
        hasher.update(f.read(sample_size))
    return hasher.hexdigest()


def compute_cache_fingerprint(path: Path, encoder_model: str, max_length: int) -> str:
    dataset_fingerprint = compute_dataset_fingerprint(path)
    hasher = hashlib.sha256()
    hasher.update(f"encoder_model={encoder_model}".encode())
    hasher.update(f"max_length={max_length}".encode())
    hasher.update(f"dataset={dataset_fingerprint}".encode())
    return hasher.hexdigest()


def _validate_equal_sequence_lengths(sequences_texts: list[list[str]]) -> int:
    lengths = sorted({len(sequence) for sequence in sequences_texts})
    if len(lengths) != 1:
        raise ValueError("All authors must have the same number of months for this baseline")
    return lengths[0]


def run_training_temporal(config: TemporalTrainConfig) -> dict[str, float | str]:
    torch, DataLoader, TensorDataset = _import_torch_modules()
    from lexical_drift.models.temporal_gru import build_temporal_gru

    data_path = Path(config.input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    dataset_fingerprint = compute_dataset_fingerprint(data_path)
    cache_fingerprint = compute_cache_fingerprint(
        data_path,
        encoder_model=config.encoder_model,
        max_length=config.max_length,
    )
    frame = pd.read_csv(data_path)
    authors, sequences_texts, sequences_months, labels = build_author_sequences_with_months(frame)

    if len(authors) < 2:
        raise ValueError("Need at least two authors for train/test split")

    n_months = _validate_equal_sequence_lengths(sequences_texts)
    months_matrix = np.asarray(sequences_months, dtype=np.int64)

    cache_path = _cache_file_path(config.cache_dir, config.encoder_model)
    embeddings = None
    used_cache = False

    if config.cache_embeddings and cache_path.exists():
        print(f"[train-temporal] found cache file: {cache_path}")
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
            print("[train-temporal] loaded cached embeddings")
            print(f"[train-temporal] fingerprint match: {cache_fingerprint}")
        elif cached_fingerprint != cache_fingerprint:
            print("[train-temporal] cache fingerprint mismatch, recomputing embeddings")
            print(
                f"[train-temporal] expected={cache_fingerprint[:12]} "
                f"cached={cached_fingerprint[:12]}"
            )
        else:
            print("[train-temporal] cache mismatch, recomputing embeddings")

    if embeddings is None:
        flat_texts = [text for sequence in sequences_texts for text in sequence]
        print(
            "[train-temporal] encoding "
            f"{len(flat_texts)} monthly texts with model={config.encoder_model}"
        )
        flat_embeddings = encode_texts_to_embeddings(
            texts=flat_texts,
            model_name=config.encoder_model,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        embedding_dim = int(flat_embeddings.shape[1])
        embeddings = flat_embeddings.reshape(
            len(authors),
            n_months,
            embedding_dim,
        ).astype(np.float32)

        if config.cache_embeddings:
            ensure_dir(config.cache_dir)
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                authors=np.asarray(authors, dtype=object),
                months=months_matrix,
                fingerprint=np.asarray(cache_fingerprint),
            )
            print(f"[train-temporal] saved embeddings cache to {cache_path}")

    indices = np.arange(len(authors), dtype=np.int64)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = labels[train_idx].astype(np.float32)
    y_test = labels[test_idx].astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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

        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        print(f"[train-temporal] epoch={epoch + 1}/{config.epochs} train_loss={avg_epoch_loss:.4f}")

    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_size = int(batch_x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            targets = batch_y.squeeze(1).cpu().numpy().astype(int)

            all_preds.append(preds)
            all_targets.append(targets)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / max(total_count, 1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_loss": float(avg_loss),
    }

    output_dir = ensure_dir(config.output_dir)
    model_path = output_dir / "temporal_gru.pt"
    metadata_path = output_dir / "temporal_metadata.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "gru_hidden_dim": int(config.gru_hidden_dim),
            "gru_layers": int(config.gru_layers),
            "dropout": float(config.dropout),
            "encoder_model": config.encoder_model,
            "max_length": int(config.max_length),
        },
        model_path,
    )

    metadata = {
        "model_type": "temporal_transformer_gru",
        "input_path": str(data_path),
        "model_path": str(model_path),
        "cache_path": str(cache_path) if config.cache_embeddings else None,
        "metrics": metrics,
        "config": asdict(config),
        "n_authors": int(len(authors)),
        "months": int(n_months),
        "dataset_fingerprint": dataset_fingerprint,
        "cache_fingerprint": cache_fingerprint,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "avg_loss": metrics["avg_loss"],
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "cache_path": str(cache_path) if config.cache_embeddings else "",
        "used_cache": used_cache,
    }
