from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import HeadDataset, collate_residues
from .metrics import regression_metrics
from .models import ResidueQualityHead
from .registry import get_feature_spec
from .train_utils import resolve_device, seed_everything, write_history


def _run_epoch(model, loader, device, optimizer=None) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    truths: list[float] = []
    predictions: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            embedding = batch["embedding"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(embedding)
            if not training:
                prediction = prediction.clamp(0.0, 1.0)
            loss = nn.functional.mse_loss(prediction[mask], label[mask])
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            truths.extend(label[mask].detach().cpu().tolist())
            predictions.extend(prediction[mask].detach().cpu().tolist())
    if not batches:
        raise RuntimeError("The data loader produced no batches")
    return total_loss / batches, regression_metrics(truths, predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a residue-level quality head on frozen representations.")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--all-train-decoys",
        action="store_true",
        help="Use every manifest row each epoch instead of sampling one decoy per target.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_feature_spec(args.feature)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_data = HeadDataset(
        args.train_manifest,
        spec.dimension,
        one_decoy_per_target=not args.all_train_decoys,
        random_decoy=not args.all_train_decoys,
    )
    valid_data = HeadDataset(args.valid_manifest, spec.dimension, one_decoy_per_target=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_residues,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_residues,
    )
    hidden_dims = () if args.head == "linear" else spec.head_hidden_dims
    model = ResidueQualityHead(spec.dimension, hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = _run_epoch(model, train_loader, device, optimizer)
        valid_loss, valid_metrics = _run_epoch(model, valid_loader, device)
        scheduler.step(valid_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
            "valid_pearson": valid_metrics["pearson"],
            "valid_spearman": valid_metrics["spearman"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature": spec.name,
            "feature_dim": spec.dimension,
            "checkpoint": spec.checkpoint,
            "head": args.head,
            "hidden_dims": list(hidden_dims),
            "dropout": args.dropout,
            "epoch": epoch,
            "valid_metrics": valid_metrics,
        }
        torch.save(checkpoint, output / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, output / "best.pt")
        write_history(output / "history.csv", history)
    with (output / "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
