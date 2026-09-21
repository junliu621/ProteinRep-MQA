from __future__ import annotations

from pathlib import Path

import torch

from ..io import save_embedding
from .common import read_fasta, resolve_device


def extract(args) -> None:
    import esm

    device = resolve_device(args.device)
    model_name = args.model_path or "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for identifier, sequence in read_fasta(args.input):
        destination = output / f"{identifier}.npz"
        if destination.exists() and not args.overwrite:
            continue
        if len(sequence) > 1022:
            raise ValueError(f"{identifier} has {len(sequence)} residues; ESM-2 supports at most 1022")
        _, _, tokens = converter([(identifier, sequence)])
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model(tokens, repr_layers=[model.num_layers])
        embedding = result["representations"][model.num_layers][0, 1 : len(sequence) + 1]
        save_embedding(destination, embedding, model_name=model_name, source=str(Path(args.input).resolve()), sequence=sequence)

