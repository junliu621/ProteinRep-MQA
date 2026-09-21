from __future__ import annotations

import re
from pathlib import Path

import torch

from ..io import save_embedding
from .common import read_fasta, resolve_device


def extract(args) -> None:
    from transformers import T5EncoderModel, T5Tokenizer

    device = resolve_device(args.device)
    model_name = args.model_path or "Rostlab/ProstT5"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).eval().to(device)
    model = model.half() if device.type == "cuda" else model.float()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for identifier, sequence in read_fasta(args.input):
        destination = output / f"{identifier}.npz"
        if destination.exists() and not args.overwrite:
            continue
        sequence = re.sub(r"[UZOB]", "X", sequence)
        prepared = "<AA2fold> " + " ".join(sequence)
        tokens = tokenizer(prepared, add_special_tokens=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state[0]
        embedding = hidden[1 : len(sequence) + 1]
        if embedding.shape[0] != len(sequence):
            raise RuntimeError(f"ProstT5 token alignment failed for {identifier}: {tuple(hidden.shape)}")
        save_embedding(destination, embedding, model_name=model_name, source=str(Path(args.input).resolve()), sequence=sequence)

