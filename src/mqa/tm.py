"""Train or apply the standalone and DeepAccNet-based TM-score predictors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import load_deepaccnet_sample
from .io import load_embedding, read_manifest
from .metrics import regression_metrics
from .models import ResidueQualityHead, create_tm_fusion_model
from .registry import canonical_feature_name, get_feature_spec
from .train_utils import load_checkpoint, resolve_device, seed_everything, write_history


TM_HEAD_DIMS = (512, 256, 128)


def tm_label(row):
    value = float(row["tm_score"])
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid TM-score for {row['target']}/{row['decoy']}: {value}")
    return value


class TMDataset(Dataset):
    def __init__(self, manifest, framework, feature, *, sampling="all",
                 require_labels=False, trim_terminal_tokens=False, size_limit=None):
        required = ["target", "decoy"]
        if framework == "fusion":
            required.append("feature")
        if feature != "deepaccnet":
            required.append("embedding")
        if require_labels:
            required.append("tm_score")
        rows = read_manifest(manifest, required)
        seen = set()
        grouped = defaultdict(list)
        for row in rows:
            key = (row["target"], row["decoy"])
            if key in seen or any(not row.get(field) for field in required):
                raise ValueError(f"Duplicate or incomplete manifest row: {key}")
            seen.add(key)
            if row.get("tm_score"):
                tm_label(row)
            grouped[row["target"]].append(row)

        for candidates in grouped.values():
            candidates.sort(key=lambda row: row["decoy"])
        if size_limit is not None:
            retained = set()
            for target, candidates in grouped.items():
                row = candidates[0]
                if row.get("length"):
                    length = int(row["length"])
                else:
                    with np.load(row["feature"], allow_pickle=False) as archive:
                        length = int(archive["tbt"].shape[-1])
                if 0 < length < size_limit:
                    retained.add(target)
            grouped = {target: records for target, records in grouped.items() if target in retained}
        if not grouped:
            raise ValueError(f"No eligible samples in {manifest}")

        self.targets = set(grouped)
        self.framework = framework
        self.feature_dim = None if feature == "deepaccnet" else get_feature_spec(feature).dimension
        self.trim_terminal_tokens = trim_terminal_tokens
        self.sampling = sampling
        self.items = ([[row] for row in rows if row["target"] in grouped] if sampling == "all"
                      else [grouped[target] for target in sorted(grouped)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        candidates = self.items[index]
        row = random.choice(candidates) if self.sampling == "random" else candidates[0]
        if self.framework == "fusion":
            sample = load_deepaccnet_sample(
                row["feature"], row.get("embedding") if self.feature_dim else None,
                self.feature_dim, tm_features=True,
            )
        else:
            embedding = load_embedding(row["embedding"], self.feature_dim)
            if self.trim_terminal_tokens:
                embedding = embedding[1:-1]
            if len(embedding) == 0:
                raise ValueError(f"No residues in {row['embedding']}")
            sample = {"embedding": embedding}
        return row, sample


def _collate(batch):
    return batch


def _prediction(model, framework, batch, device):
    if framework == "fusion":
        _, sample = batch[0]
        inputs = {key: value.to(device) for key, value in sample.items()}
        return model(**inputs)["tm_score"].reshape(1)
    embeddings = [sample["embedding"] for _, sample in batch]
    values = model(torch.cat(embeddings).to(device))
    return torch.stack([part.mean() for part in values.split([len(x) for x in embeddings])])


def _metrics(truth, predictions):
    metrics = regression_metrics(truth, predictions)
    metrics["n_decoys"] = metrics.pop("n_residues")
    return metrics


def _run_epoch(model, framework, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    truth, predictions, losses = [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            targets = torch.tensor([tm_label(row) for row, _ in batch], device=device)
            prediction = _prediction(model, framework, batch, device)
            if not training:
                prediction = prediction.clamp(0, 1)
            loss = nn.functional.mse_loss(prediction, targets)
            if not math.isfinite(float(loss.detach())):
                raise ValueError("Non-finite TM-score loss")
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            truth.extend(targets.cpu().tolist())
            predictions.extend(prediction.detach().cpu().tolist())
    return sum(losses) / len(losses), _metrics(truth, predictions)


def _checkpoint_save(checkpoint, output):
    pending = output.with_suffix(".pending")
    torch.save(checkpoint, pending)
    pending.replace(output)


def train(argv=None):
    parser = argparse.ArgumentParser(description="Train a decoy-level TM-score predictor.")
    parser.add_argument("--framework", choices=("standalone", "fusion"), required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--size-limit", type=int, default=280, help="Fusion training length limit, exclusive")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all-train-decoys", action="store_true")
    parser.add_argument("--trim-terminal-tokens", action="store_true")
    args = parser.parse_args(argv)

    fusion = args.framework == "fusion"
    feature = canonical_feature_name(args.feature)
    if not fusion and feature == "deepaccnet":
        parser.error("The DeepAccNet baseline uses --framework fusion")
    if fusion and args.trim_terminal_tokens:
        parser.error("Fusion aligns terminal tokens to the structure length automatically")
    args.epochs = args.epochs if args.epochs is not None else (200 if fusion else 30)
    args.batch_size = args.batch_size if args.batch_size is not None else (1 if fusion else 8)
    args.learning_rate = args.learning_rate if args.learning_rate is not None else (5e-4 if fusion else 1e-4)
    if args.epochs < 1 or args.batch_size < 1 or not args.learning_rate > 0:
        parser.error("Epochs, batch size and learning rate must be positive")
    if fusion and args.batch_size != 1:
        parser.error("Fusion uses one structure per batch")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_options = {
        "require_labels": True,
        "trim_terminal_tokens": args.trim_terminal_tokens,
        "size_limit": args.size_limit if fusion else None,
    }
    training = TMDataset(args.train_manifest, args.framework, feature,
                         sampling="all" if args.all_train_decoys else "random", **dataset_options)
    validation = TMDataset(args.valid_manifest, args.framework, feature, sampling="first", **dataset_options)
    if training.targets & validation.targets:
        raise ValueError("Train/validation targets overlap")
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=_collate)
    train_loader = DataLoader(training, shuffle=True, **loader_options)
    valid_loader = DataLoader(validation, **loader_options)
    metadata = {"task": "tm_score", "framework": args.framework, "feature": feature}
    if fusion:
        metadata.update(chunks=5, channels=128)
        model = create_tm_fusion_model(feature, chunks=5, channels=128)
    else:
        feature_dim = get_feature_spec(feature).dimension
        model = ResidueQualityHead(feature_dim, TM_HEAD_DIMS, args.dropout)
        metadata.update(feature_dim=feature_dim, hidden_dims=list(TM_HEAD_DIMS), dropout=args.dropout,
                        trim_terminal_tokens=args.trim_terminal_tokens, pooling="mean_residue_predictions")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay) if fusion else
                 torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2))

    args.output.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = _run_epoch(model, args.framework, train_loader, device, optimizer)
        valid_loss, valid_metrics = _run_epoch(model, args.framework, valid_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{f"valid_{key}": val for key, val in valid_metrics.items()}}
        print(json.dumps(row), flush=True)
        history.append(row)
        checkpoint = {**metadata, "epoch": epoch, "model_state_dict": model.state_dict(),
                      "valid_metrics": valid_metrics}
        _checkpoint_save(checkpoint, args.output / "last.pt")
        if valid_loss < best:
            best = valid_loss
            _checkpoint_save(checkpoint, args.output / "best.pt")
        write_history(args.output / "history.csv", history)
        if fusion:
            scheduler.step()
        else:
            scheduler.step(valid_loss)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))


def predict(argv=None):
    parser = argparse.ArgumentParser(description="Predict one TM-score per decoy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", help="Feature name for legacy TM fusion checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-legacy-pickle", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint(args.checkpoint, allow_legacy_pickle=args.allow_legacy_pickle)
    if checkpoint.get("task") != "tm_score":
        raise ValueError("Expected a TM-score checkpoint")
    state = checkpoint["model_state_dict"]
    framework = checkpoint.get("framework", "fusion" if "tm_head.0.weight" in state else "standalone")
    if args.feature and checkpoint.get("feature") and canonical_feature_name(args.feature) != checkpoint["feature"]:
        parser.error("--feature does not match the checkpoint")
    feature = checkpoint.get("feature", args.feature)
    if not feature:
        parser.error("Set --feature for a legacy fusion checkpoint (use deepaccnet for the baseline)")
    feature = canonical_feature_name(feature)
    if framework == "fusion":
        model = create_tm_fusion_model(feature, chunks=int(checkpoint.get("chunks", 5)),
                                       channels=int(state["tm_head.0.weight"].shape[1]))
    else:
        model = ResidueQualityHead(
            int(checkpoint["feature_dim"]), checkpoint["hidden_dims"], float(checkpoint["dropout"])
        )
    device = resolve_device(args.device)
    model.load_state_dict(state)
    model.to(device).eval()
    dataset = TMDataset(args.manifest, framework, feature,
                        trim_terminal_tokens=bool(checkpoint.get("trim_terminal_tokens", False)))
    records, truth, scored_predictions = [], [], []
    with torch.inference_mode():
        for row, sample in dataset:
            score = float(_prediction(model, framework, [(row, sample)], device)[0])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite TM-score for {row['target']}/{row['decoy']}")
            score = max(0.0, min(1.0, score))
            reference = tm_label(row) if row.get("tm_score") else None
            if reference is not None:
                truth.append(reference)
                scored_predictions.append(score)
            records.append(dict(target=row["target"], decoy=row["decoy"],
                                pred_tm_score=score, tm_score=reference))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "decoy", "pred_tm_score", "tm_score"])
        writer.writeheader()
        writer.writerows(records)
    summary = dict(framework=framework, feature=feature, n_decoys=len(records))
    if truth:
        summary["overall"] = _metrics(truth, scored_predictions)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
