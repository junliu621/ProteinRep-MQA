from __future__ import annotations

import argparse
import importlib

from ..registry import FEATURE_SPECS, canonical_feature_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract standardized residue-level representations.")
    parser.add_argument("--feature", required=True, help=f"One of: {', '.join(sorted(FEATURE_SPECS))}")
    parser.add_argument("--input", required=True, help="FASTA file or structure file/directory")
    parser.add_argument("--output", required=True, help="Output directory for standardized NPZ files")
    parser.add_argument("--model-path", help="Checkpoint name or local checkpoint path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--chain", help="Optional structure chain identifier")
    parser.add_argument("--foldseek", help="Foldseek executable required by SaProt")
    parser.add_argument("--proteinmpnn-repo", help="Path to an upstream ProteinMPNN checkout")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.feature = canonical_feature_name(args.feature)
    if args.feature not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature: {args.feature}")
    backend = importlib.import_module(f"mqa.extractors.{args.feature}")
    backend.extract(args)


if __name__ == "__main__":
    main()

