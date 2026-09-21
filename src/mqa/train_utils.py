from __future__ import annotations

import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def write_history(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: str | Path, *, allow_legacy_pickle: bool = False):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not allow_legacy_pickle:
            raise RuntimeError(
                "This legacy checkpoint contains objects outside PyTorch's weights-only format. "
                "Load it only if you trust its source, using --allow-legacy-pickle."
            ) from exc
        return torch.load(path, map_location="cpu", weights_only=False)
