from __future__ import annotations

from pathlib import Path

import torch


STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    sequence: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(sequence).upper()))
                identifier = line[1:].split()[0]
                sequence = []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence found before a FASTA header in {path}")
                sequence.append(line.replace(" ", ""))
    if identifier is not None:
        records.append((identifier, "".join(sequence).upper()))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def structure_files(path: str | Path) -> tuple[Path, list[Path]]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in STRUCTURE_SUFFIXES:
            raise ValueError(f"Unsupported structure file: {path}")
        return path.parent, [path]
    files = sorted(item for item in path.rglob("*") if item.suffix.lower() in STRUCTURE_SUFFIXES)
    if not files:
        raise ValueError(f"No PDB or mmCIF structures found under {path}")
    return path, files


def output_path(output_root: str | Path, input_root: Path, source: Path) -> Path:
    relative = source.relative_to(input_root).with_suffix(".npz")
    return Path(output_root) / relative

