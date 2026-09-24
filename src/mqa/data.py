from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_embedding, load_local_lddt, read_manifest
from .models.deepaccnet import calculate_lddt


ESTOGRAM_BINS = np.asarray(
    [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]
)


class HeadDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        feature_dim: int,
        *,
        one_decoy_per_target: bool = False,
        random_decoy: bool = False,
    ) -> None:
        rows = read_manifest(manifest, ("target", "decoy", "embedding", "label"))
        self.feature_dim = feature_dim
        self.random_decoy = random_decoy
        if one_decoy_per_target:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row["target"]].append(row)
            self.items: list[list[dict[str, str]]] = [grouped[key] for key in sorted(grouped)]
        else:
            self.items = [[row] for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        candidates = self.items[index]
        row = random.choice(candidates) if self.random_decoy else sorted(candidates, key=lambda x: x["decoy"])[0]
        embedding = load_embedding(row["embedding"], self.feature_dim)
        label, mask = load_local_lddt(row["label"])
        if embedding.shape[0] == label.shape[0] + 2:
            embedding = embedding[1:-1]
        if embedding.shape[0] != label.shape[0]:
            raise ValueError(
                f"Length mismatch for {row['target']}/{row['decoy']}: "
                f"embedding={embedding.shape[0]}, label={label.shape[0]}"
            )
        return {"embedding": embedding, "label": label, "mask": mask, "row": row}


def collate_residues(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "embedding": torch.cat([item["embedding"] for item in batch], dim=0),
        "label": torch.cat([item["label"] for item in batch], dim=0),
        "mask": torch.cat([item["mask"] for item in batch], dim=0),
        "lengths": [item["embedding"].shape[0] for item in batch],
        "rows": [item["row"] for item in batch],
    }


def _distance_transform(values: np.ndarray, cutoff: float = 4.0, scale: float = 3.0) -> np.ndarray:
    return np.arcsinh(np.maximum(values, cutoff) - cutoff) / scale


def load_deepaccnet_sample(
    feature_path: str | Path,
    embedding_path: str | Path,
    representation_dim: int,
    native_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    with np.load(feature_path, allow_pickle=False) as data:
        required = {
            "idx", "val", "phi", "psi", "omega6d", "theta6d", "phi6d",
            "tbt", "obt", "prop", "euler", "maps",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{feature_path} is missing DeepAccNet arrays: {sorted(missing)}")
        idx = data["idx"].copy()
        values = data["val"].copy()
        phi = data["phi"].copy()
        psi = data["psi"].copy()
        omega = data["omega6d"].copy()
        theta = data["theta6d"].copy()
        phi6d = data["phi6d"].copy()
        tbt_raw = data["tbt"].copy()
        obt = data["obt"].T.copy()
        prop = data["prop"].T.copy()
        euler = data["euler"].copy()
        maps = data["maps"].copy()

    length = tbt_raw.shape[-1]
    embedding = load_embedding(embedding_path, representation_dim)
    if embedding.shape[0] == length + 2:
        embedding = embedding[1:-1]
    if embedding.shape[0] != length:
        raise ValueError(f"Embedding length {embedding.shape[0]} != DeepAccNet length {length}")

    angles = np.stack((np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)), axis=-1)
    one_body = np.concatenate((angles, obt, prop), axis=-1)
    orientations = np.stack((omega, theta, phi6d), axis=-1)
    orientations = np.concatenate((np.sin(orientations), np.cos(orientations)), axis=-1)
    euler_features = np.concatenate((np.sin(euler), np.cos(euler)), axis=-1)
    separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :]) / 100.0 - 1.0
    tbt = tbt_raw.transpose(1, 2, 0)
    tbt[:, :, 0] = _distance_transform(tbt[:, :, 0])
    maps = _distance_transform(maps)
    two_body = np.concatenate((tbt, maps, euler_features, orientations, separation[:, :, None]), axis=-1)

    sample = {
        "voxel_indices": torch.from_numpy(idx).long(),
        "voxel_values": torch.from_numpy(values).float(),
        "one_body": torch.from_numpy(one_body).float(),
        "two_body": torch.from_numpy(two_body.transpose(2, 0, 1)).float(),
        "representation": embedding,
    }
    if native_path is not None:
        with np.load(native_path, allow_pickle=False) as native:
            native_distances = native["tbt"][0].astype(np.float32)
        if native_distances.shape != (length, length):
            raise ValueError(f"Native distance shape {native_distances.shape} != ({length}, {length})")
        estogram = np.digitize(tbt_raw[0] - native_distances, ESTOGRAM_BINS).astype(np.int64)
        mask = (native_distances < 15.0).astype(np.float32)
        estogram_one_hot = torch.nn.functional.one_hot(
            torch.from_numpy(estogram), num_classes=15
        ).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask)
        sample.update(
            {
                "estogram_target": torch.from_numpy(estogram),
                "mask_target": mask_tensor,
                "local_lddt_target": calculate_lddt(estogram_one_hot, mask_tensor),
            }
        )
    return sample


class FusionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        representation_dim: int,
        require_native: bool = True,
        *,
        one_decoy_per_target: bool = False,
        random_decoy: bool = False,
    ) -> None:
        required = ["target", "decoy", "feature", "embedding"]
        if require_native:
            required.append("native")
        rows = read_manifest(manifest, required)
        if one_decoy_per_target:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row["target"]].append(row)
            self.items = [grouped[key] for key in sorted(grouped)]
        else:
            self.items = [[row] for row in rows]
        self.representation_dim = representation_dim
        self.require_native = require_native
        self.random_decoy = random_decoy

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        candidates = self.items[index]
        row = random.choice(candidates) if self.random_decoy else sorted(candidates, key=lambda x: x["decoy"])[0]
        sample = load_deepaccnet_sample(
            row["feature"],
            row["embedding"],
            self.representation_dim,
            row.get("native") if self.require_native else None,
        )
        sample["row"] = row
        return sample
