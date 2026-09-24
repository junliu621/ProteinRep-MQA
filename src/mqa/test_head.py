from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import HeadDataset, collate_residues
from .metrics import regression_metrics
from .models import ResidueQualityHead
from .registry import get_feature_spec
from .train_utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test a trained residue-level quality head.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature", help="Required for a legacy state-dict-only checkpoint")
    parser.add_argument("--head", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint and "feature" in checkpoint:
        state = checkpoint["model_state_dict"]
        feature = checkpoint["feature"]
        feature_dim = checkpoint["feature_dim"]
        hidden_dims = checkpoint["hidden_dims"]
        dropout = checkpoint["dropout"]
    elif "model_state_dict" in checkpoint or "state_dict" in checkpoint:
        if not args.feature:
            raise ValueError("Specify --feature for this legacy checkpoint")
        spec = get_feature_spec(args.feature)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        feature = spec.name
        feature_dim = int(checkpoint.get("input_dim", spec.dimension))
        hidden_dims = checkpoint.get("hidden_dims", () if args.head == "linear" else spec.head_hidden_dims)
        dropout = float(checkpoint.get("dropout", args.dropout))
    else:
        if not args.feature:
            raise ValueError("Specify --feature for a legacy state-dict-only checkpoint")
        spec = get_feature_spec(args.feature)
        state = checkpoint
        feature = spec.name
        feature_dim = spec.dimension
        hidden_dims = () if args.head == "linear" else spec.head_hidden_dims
        dropout = args.dropout
    model = ResidueQualityHead(
        feature_dim, hidden_dims, dropout
    )
    model.load_state_dict(state)
    model.to(device).eval()
    dataset = HeadDataset(args.manifest, feature_dim)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_residues)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    truth_all: list[float] = []
    prediction_all: list[float] = []
    records = []
    residue_path = output / "residue_scores.csv"
    with residue_path.open("w", newline="") as residue_handle, torch.no_grad():
        residue_writer = csv.DictWriter(
            residue_handle,
            fieldnames=(
                "target",
                "decoy",
                "residue_index",
                "predicted_local_lddt",
                "true_local_lddt",
                "valid_label",
            ),
        )
        residue_writer.writeheader()
        for batch in loader:
            prediction = model(batch["embedding"].to(device)).clamp(0.0, 1.0).cpu()
            label = batch["label"]
            mask = batch["mask"]
            row = batch["rows"][0]
            target_dir = output / row["target"]
            target_dir.mkdir(parents=True, exist_ok=True)
            result_path = target_dir / f"{row['decoy']}.npz"
            np.savez_compressed(result_path, local_lddt=prediction.numpy().astype(np.float32))
            metrics = regression_metrics(label[mask].tolist(), prediction[mask].tolist())
            valid_count = metrics.pop("n_residues")
            records.append(
                {
                    "target": row["target"],
                    "decoy": row["decoy"],
                    "n_residues": int(prediction.numel()),
                    "n_valid_labels": valid_count,
                    "predicted_global_lddt": float(prediction.mean()),
                    "true_global_lddt": float(label[mask].mean()),
                    **metrics,
                    "prediction_file": str(result_path.relative_to(output)),
                }
            )
            for residue_index, (predicted, observed, valid) in enumerate(
                zip(prediction.tolist(), label.tolist(), mask.tolist()), start=1
            ):
                residue_writer.writerow(
                    {
                        "target": row["target"],
                        "decoy": row["decoy"],
                        "residue_index": residue_index,
                        "predicted_local_lddt": predicted,
                        "true_local_lddt": observed if valid else "",
                        "valid_label": int(valid),
                    }
                )
            truth_all.extend(label[mask].tolist())
            prediction_all.extend(prediction[mask].tolist())
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "feature": feature,
        "overall": regression_metrics(truth_all, prediction_all),
        "samples": records,
    }
    with (output / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
