from __future__ import annotations

import numpy as np


def encode_texts_to_embeddings(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode texts with a frozen transformer into a 2D embedding matrix [N, D]."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'Transformers dependencies are missing. Install with: pip install -e ".[dl,nlp]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs = model(**inputs)

            if getattr(outputs, "last_hidden_state", None) is not None:
                embedding = outputs.last_hidden_state[:, 0, :]
            elif getattr(outputs, "pooler_output", None) is not None:
                embedding = outputs.pooler_output
            else:
                raise ValueError("Model output has neither last_hidden_state nor pooler_output")

            batches.append(embedding.cpu().numpy().astype(np.float32))

    return np.concatenate(batches, axis=0)
