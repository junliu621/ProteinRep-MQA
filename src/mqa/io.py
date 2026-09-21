from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EMBEDDING_KEYS = ("embedding", "embeddings", "representation", "representations")


def normalize_embedding(array: np.ndarray | torch.Tensor, expected_dim: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected an [L, D] embedding, got {tuple(tensor.shape)}")
    if expected_dim is not None and tensor.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Embedding contains NaN or infinity")
    return tensor.contiguous()


def save_embedding(
    path: str | Path,
    embedding: np.ndarray | torch.Tensor,
    *,
    model_name: str,
    source: str,
    sequence: str | None = None,
    chain_ids: Iterable[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = normalize_embedding(embedding)
    payload: dict[str, Any] = {
        "embedding": tensor.cpu().numpy().astype(np.float32, copy=False),
        "model_name": np.asarray(model_name),
        "source": np.asarray(source),
    }
    if sequence is not None:
        payload["sequence"] = np.asarray(sequence)
    if chain_ids is not None:
        chain_ids = list(chain_ids)
        if len(chain_ids) != tensor.shape[0]:
            raise ValueError("chain_ids length must match the number of residues")
        payload["chain_ids"] = np.asarray(chain_ids)
    np.savez_compressed(path, **payload)


def load_embedding(path: str | Path, expected_dim: int | None = None) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = next((candidate for candidate in EMBEDDING_KEYS if candidate in data), None)
            if key is None:
                raise KeyError(f"{path} has no embedding key; found {list(data.keys())}")
            array = data[key].copy()
    elif path.suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            key = next((candidate for candidate in EMBEDDING_KEYS if candidate in value), None)
            if key is None:
                raise KeyError(f"{path} has no embedding key; found {list(value.keys())}")
            value = value[key]
        array = value
    else:
        raise ValueError(f"Unsupported embedding format: {path.suffix}")
    return normalize_embedding(array, expected_dim)


def load_local_lddt(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        local = payload.get("local_lddt", payload)
        values: list[float] = []
        valid: list[bool] = []
        for chain in sorted(local):
            chain_values = local[chain]
            for index in sorted(chain_values, key=lambda item: int(item)):
                value = chain_values[index]
                values.append(0.0 if value is None else float(value))
                valid.append(value is not None)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            key = "local_lddt" if "local_lddt" in payload else "lddt"
            values_array = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            mask_array = np.asarray(payload["mask"], dtype=bool).reshape(-1) if "mask" in payload else np.isfinite(values_array)
        values = np.nan_to_num(values_array, nan=0.0).tolist()
        valid = mask_array.tolist()
    else:
        raise ValueError(f"Unsupported label format: {path.suffix}")
    if not values or not any(valid):
        raise ValueError(f"No valid local lDDT labels in {path}")
    return torch.tensor(values, dtype=torch.float32), torch.tensor(valid, dtype=torch.bool)


def read_manifest(path: str | Path, required: Iterable[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(required) - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    base = path.parent
    path_fields = {"embedding", "label", "feature", "native"}
    for row in rows:
        for field in path_fields & row.keys():
            candidate = Path(row[field])
            if row[field] and not candidate.is_absolute():
                row[field] = str((base / candidate).resolve())
    return rows

