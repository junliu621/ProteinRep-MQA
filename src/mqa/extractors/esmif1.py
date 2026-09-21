from __future__ import annotations

from pathlib import Path

import torch

from ..io import save_embedding
from .common import output_path, resolve_device, structure_files


def extract(args) -> None:
    import esm

    device = resolve_device(args.device)
    model_name = "esm_if1_gvp4_t16_142M_UR50"
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
    input_root, files = structure_files(args.input)
    for source in files:
        destination = output_path(args.output, input_root, source)
        if destination.exists() and not args.overwrite:
            continue
        structure = esm.inverse_folding.util.load_structure(str(source), args.chain)
        coordinates, sequence = esm.inverse_folding.util.extract_coords_from_structure(structure)
        batch = [(coordinates, None, sequence)]
        coords, confidence, _, _, padding_mask = converter(batch, device=device)
        with torch.no_grad():
            encoder = model.encoder.forward(coords, padding_mask, confidence, return_all_hiddens=False)
        embedding = encoder["encoder_out"][0][1:-1, 0]
        if embedding.shape[0] != len(sequence):
            raise RuntimeError(f"ESM-IF1 length mismatch for {source}")
        save_embedding(destination, embedding, model_name=model_name, source=str(source.resolve()), sequence=sequence)

