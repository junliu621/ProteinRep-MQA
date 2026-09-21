#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an MQA manifest from matching target/decoy trees.")
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--label-root")
    parser.add_argument("--feature-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-suffix", default=".npz")
    args = parser.parse_args()

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
