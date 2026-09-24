from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .data import FusionDataset
from .metrics import regression_metrics
from .models import create_fusion_model, deepaccnet_loss
from .registry import get_feature_spec
from .train_utils import load_checkpoint, resolve_device, seed_everything, write_history


def _to_device(sample: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}


def _epoch(model, dataset, device, optimizer=None, order=None):
    training = optimizer is not None
    model.train(training)
    indices = list(range(len(dataset))) if order is None else order
    losses = []
    truth: list[float] = []
    prediction: list[float] = []
    details = {"estogram_loss": [], "mask_loss": [], "lddt_loss": []}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index in indices:
            sample = _to_device(dataset[index], device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                sample["voxel_indices"],
                sample["voxel_values"],
                sample["one_body"],
                sample["two_body"],
                sample["representation"],
            )
            loss, components = deepaccnet_loss(
                output,
                sample["estogram_target"],
                sample["mask_target"],
                sample["local_lddt_target"],
            )
            if training:
                loss.backward()
                optimizer.step()
            losses.append(components["loss"])
            for name in details:
                details[name].append(components[name])
            truth.extend(sample["local_lddt_target"].detach().cpu().tolist())
            prediction.extend(output["local_lddt"].detach().cpu().clamp(0.0, 1.0).tolist())
    metrics = regression_metrics(truth, prediction)
    metrics.update({name: sum(values) / len(values) for name, values in details.items()})
    return sum(losses) / len(losses), metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the checkpoint-compatible DeepAccNet fusion model.")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--allow-legacy-pickle", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_feature_spec(args.feature)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = FusionDataset(
        args.train_manifest,
        spec.dimension,
        one_decoy_per_target=True,
        random_decoy=True,
    )
    valid_data = FusionDataset(
        args.valid_manifest,
        spec.dimension,
        one_decoy_per_target=True,
    )
    model = create_fusion_model(spec.name, chunks=args.chunks, channels=args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, allow_legacy_pickle=args.allow_legacy_pickle)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_valid_loss", best_loss))

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = args.learning_rate * args.decay ** (epoch - 1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        order = list(range(len(train_data)))
        random.shuffle(order)
        train_loss, _ = _epoch(model, train_data, device, optimizer, order)
        valid_loss, valid_metrics = _epoch(model, valid_data, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_rmse": valid_metrics["rmse"],
            "valid_pearson": valid_metrics["pearson"],
            "valid_spearman": valid_metrics["spearman"],
            "learning_rate": learning_rate,
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "epoch": epoch,
            "feature": spec.name,
            "feature_dim": spec.dimension,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_loss": min(best_loss, valid_loss),
            "chunks": args.chunks,
            "channels": args.channels,
            "valid_metrics": valid_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, output_dir / "best.pt")
        write_history(output_dir / "history.csv", history)
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
