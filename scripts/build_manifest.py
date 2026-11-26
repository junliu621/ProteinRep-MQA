#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def lddt_rows(args):
    embedding_root = Path(args.embedding_root).resolve()
    rows = []
    for embedding in sorted(embedding_root.glob(f"*/*{args.embedding_suffix}")):
        target = embedding.parent.name
        decoy = embedding.name[: -len(args.embedding_suffix)]
        row = {"target": target, "decoy": decoy, "embedding": str(embedding)}
        if args.label_root:
            label_root = Path(args.label_root).resolve() / target
            label = next(
                (candidate for suffix in (".json", ".npz") if (candidate := label_root / f"{decoy}{suffix}").exists()),
                None,
            )
            if label is None:
                continue
            row["label"] = str(label)
        if args.feature_root:
            root = Path(args.feature_root).resolve()
            feature = root / target / f"{decoy}.features.npz"
            native = root / target / "native.features.npz"
            if not feature.exists() or not native.exists():
                continue
            row.update({"feature": str(feature), "native": str(native)})
        rows.append(row)
    return rows


def tm_rows(args):
    labels = {}
    if args.tm_labels:
        with Path(args.tm_labels).open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"target", "decoy", "tm_score"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"TM label table is missing columns: {sorted(missing)}")
            for row in reader:
                key = (row["target"], row["decoy"])
                score = float(row["tm_score"])
                if not all(key) or key in labels or not 0 <= score <= 1:
                    raise ValueError(f"Invalid or duplicate TM label for {key}")
                labels[key] = score
        if not labels:
            raise ValueError("TM label table is empty")

    embedding_root = Path(args.embedding_root).resolve() if args.embedding_root else None
    feature_root = Path(args.feature_root).resolve() if args.feature_root else None
    sample_root, suffix = (
        (embedding_root, args.embedding_suffix) if embedding_root else (feature_root, ".features.npz")
    )
    rows = []
    for sample in sorted(sample_root.glob(f"*/*{suffix}")):
        if not sample.is_file():
            continue
        key = (sample.parent.name, sample.name[:-len(suffix)])
        target, decoy = key
        if embedding_root is None and decoy == "native" and not args.tm_labels:
            continue
        row = {"target": target, "decoy": decoy}
        if embedding_root:
            row["embedding"] = str(sample)
        if feature_root:
            feature = feature_root / target / f"{decoy}.features.npz"
            if not feature.is_file():
                continue
            row["feature"] = str(feature)
        if args.tm_labels:
            if key not in labels:
                continue
            row["tm_score"] = labels[key]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an MQA manifest from matching target/decoy trees.")
    parser.add_argument("--task", choices=("lddt", "tm"), default="lddt")
    parser.add_argument("--embedding-root")
    parser.add_argument("--label-root", help="lDDT label files")
    parser.add_argument("--tm-labels", help="CSV with target, decoy, tm_score columns")
    parser.add_argument("--feature-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-suffix", default=".npz")
    args = parser.parse_args()
    if args.task == "lddt" and (not args.embedding_root or args.tm_labels):
        parser.error("lDDT requires --embedding-root and uses --label-root for labels")
    if args.task == "tm" and (not (args.embedding_root or args.feature_root) or args.label_root):
        parser.error("TM requires at least one feature root and uses --tm-labels for labels")
    if not args.embedding_suffix:
        parser.error("--embedding-suffix cannot be empty")
    rows = tm_rows(args) if args.task == "tm" else lddt_rows(args)
    if not rows:
        raise SystemExit("No matching samples were found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
