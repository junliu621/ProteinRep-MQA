from __future__ import annotations

from pathlib import Path

import torch

from ..io import save_embedding
from .common import output_path, resolve_device, structure_files


def extract(args) -> None:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils.structure.protein_complex import ProteinComplex

    device = resolve_device(args.device)
    model_name = args.model_path or "esm3-open"
    model = ESM3.from_pretrained(model_name).eval().to(device)
    input_root, files = structure_files(args.input)
    for source in files:
        destination = output_path(args.output, input_root, source)
        if destination.exists() and not args.overwrite:
            continue
        chain_ids = None
        if args.chain:
            protein = ESMProtein.from_pdb(path=str(source), chain_id=args.chain)
        else:
            complex_structure = ProteinComplex.from_pdb(path=str(source))
            protein = ESMProtein.from_protein_complex(complex_structure)
            chain_ids = [str(chain) for chain in complex_structure.chain_id]
        encoded = model.encode(protein)
        with torch.no_grad():
            result = model.logits(encoded, LogitsConfig(return_embeddings=True))
        if result.embeddings is None:
            raise RuntimeError("ESM-3 did not return embeddings")
        embedding = result.embeddings[0]
        sequence = protein.sequence or ""
        if sequence and embedding.shape[0] == len(sequence) + 2:
            embedding = embedding[1:-1]
        elif sequence and embedding.shape[0] != len(sequence):
            raise RuntimeError(
                f"ESM-3 length mismatch for {source}: embedding={embedding.shape[0]}, sequence={len(sequence)}"
            )
        save_embedding(
            destination,
            embedding,
            model_name=model_name,
            source=str(source.resolve()),
            sequence=sequence or None,
            chain_ids=chain_ids,
        )
