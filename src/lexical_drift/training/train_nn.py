from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from lexical_drift.config import NNTrainConfig
from lexical_drift.models.nn_mlp import build_mlp
from lexical_drift.utils import ensure_dir

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def _import_torch_modules():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError('PyTorch is not installed. Install with: pip install -e ".[dl]"') from exc

    return torch, DataLoader, TensorDataset


def run_training_nn(config: NNTrainConfig) -> dict[str, float | str]:
    torch, DataLoader, TensorDataset = _import_torch_modules()

    data_path = Path(config.input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    X = frame["text"].astype(str).tolist()
    y = frame["drift_label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=config.max_features,
    )
    X_train_arr = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    X_test_arr = vectorizer.transform(X_test).toarray().astype(np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_arr),
        torch.from_numpy(y_train_arr).unsqueeze(1),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_arr),
        torch.from_numpy(y_test_arr).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = build_mlp(
        input_dim=int(X_train_arr.shape[1]),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

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
    model_path = output_dir / "nn_mlp.pt"
    vectorizer_path = output_dir / "nn_vectorizer.joblib"
    metadata_path = output_dir / "nn_metadata.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X_train_arr.shape[1]),
            "hidden_dim": int(config.hidden_dim),
            "dropout": float(config.dropout),
        },
        model_path,
    )
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "model_type": "tfidf_mlp",
        "input_path": str(data_path),
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "metrics": metrics,
        "config": asdict(config),
        "n_rows": int(len(frame)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "avg_loss": metrics["avg_loss"],
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "metadata_path": str(metadata_path),
    }
