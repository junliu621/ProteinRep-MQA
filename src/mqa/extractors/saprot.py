from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch

from ..io import save_embedding
from .common import output_path, resolve_device, structure_files


def _structure_sequences(foldseek: str, source: Path) -> list[tuple[str, str, str]]:
    with tempfile.TemporaryDirectory(prefix="mqa_foldseek_") as temporary:
        result_path = Path(temporary) / "result.tsv"
        command = [
            foldseek,
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(source),
            str(result_path),
        ]
        subprocess.run(command, check=True)
        records = []
        with result_path.open() as handle:
            for line in handle:
                description, sequence, structure_sequence = line.rstrip("\n").split("\t")[:3]
                chain = description.split()[0].split("_")[-1]
                records.append((chain, sequence, structure_sequence))
    if not records:
        raise RuntimeError(f"Foldseek returned no chains for {source}")
    return records


def extract(args) -> None:
    from transformers import AutoModel, AutoTokenizer

    if not args.foldseek:
        raise ValueError("SaProt extraction requires --foldseek /path/to/foldseek")
    device = resolve_device(args.device)
    model_name = args.model_path or "westlake-repl/SaProt_650M_AF2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    input_root, files = structure_files(args.input)
    for source in files:
        destination = output_path(args.output, input_root, source)
        if destination.exists() and not args.overwrite:
            continue
        embeddings = []
        chain_ids: list[str] = []
        sequences: list[str] = []
        for chain, sequence, structure_sequence in _structure_sequences(args.foldseek, source):
            if args.chain and chain != args.chain:
                continue
            if len(sequence) > 1022:
                raise ValueError(f"{source} chain {chain} has {len(sequence)} residues; SaProt supports at most 1022")
            combined = "".join(a + b.lower() for a, b in zip(sequence, structure_sequence))
            tokens = tokenizer(combined, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state[0, 1:-1]
            if hidden.shape[0] != len(sequence):
                raise RuntimeError(f"SaProt token alignment failed for {source} chain {chain}")
            embeddings.append(hidden.cpu())
            chain_ids.extend([chain] * len(sequence))
            sequences.append(sequence)
        if not embeddings:
            raise ValueError(f"Requested chain was not found in {source}")
        save_embedding(
            destination,
            torch.cat(embeddings),
            model_name=model_name,
            source=str(source.resolve()),
            sequence="".join(sequences),
            chain_ids=chain_ids,
        )
