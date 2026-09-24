from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import FusionDataset
from .models import create_fusion_model
from .train_utils import load_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepAccNet fusion inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature", help="Required only for a legacy checkpoint without feature metadata")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-legacy-pickle", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, allow_legacy_pickle=args.allow_legacy_pickle)
    feature = checkpoint.get("feature", args.feature)
    if not feature:
        raise ValueError("Specify --feature for a legacy checkpoint")
    model = create_fusion_model(
        feature,
        chunks=int(checkpoint.get("chunks", 5)),
        channels=int(checkpoint.get("channels", 128)),
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    dataset = FusionDataset(args.manifest, model.representation_dim, require_native=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    residue_path = output_dir / "residue_scores.csv"
    with residue_path.open("w", newline="") as residue_handle, torch.no_grad():
        residue_writer = csv.DictWriter(
            residue_handle,
            fieldnames=("target", "decoy", "residue_index", "local_lddt"),
        )
        residue_writer.writeheader()
        for sample in dataset:
            row = sample.pop("row")
            tensors = {key: value.to(device) for key, value in sample.items()}
            prediction = model(
                tensors["voxel_indices"],
                tensors["voxel_values"],
                tensors["one_body"],
                tensors["two_body"],
                tensors["representation"],
            )["local_lddt"].clamp(0.0, 1.0).cpu().numpy()
            destination = output_dir / row["target"] / f"{row['decoy']}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, local_lddt=prediction.astype(np.float32), global_lddt=prediction.mean())
            records.append(
                {
                    "target": row["target"],
                    "decoy": row["decoy"],
                    "n_residues": int(prediction.size),
                    "global_lddt": float(prediction.mean()),
                    "prediction_file": str(destination.relative_to(output_dir)),
                }
            )
            for residue_index, local_lddt in enumerate(prediction, start=1):
                residue_writer.writerow(
                    {
                        "target": row["target"],
                        "decoy": row["decoy"],
                        "residue_index": residue_index,
                        "local_lddt": float(local_lddt),
                    }
                )
    with (output_dir / "predictions.json").open("w") as handle:
        json.dump(records, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
