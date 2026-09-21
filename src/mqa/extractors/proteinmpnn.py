from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

from ..io import save_embedding
from .common import output_path, resolve_device, structure_files


def extract(args) -> None:
    if not args.proteinmpnn_repo:
        raise ValueError("ProteinMPNN extraction requires --proteinmpnn-repo /path/to/ProteinMPNN")
    repository = Path(args.proteinmpnn_repo).resolve()
    sys.path.insert(0, str(repository))
    try:
        from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB, parse_PDB, tied_featurize
    finally:
        sys.path.pop(0)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.model_path) if args.model_path else repository / "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = 128
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_encoder(_module, _inputs, output):
        captured["embedding"] = output[0].detach()

    hook = model.encoder_layers[-1].register_forward_hook(capture_encoder)
    input_root, files = structure_files(args.input)
    try:
        for source in files:
            destination = output_path(args.output, input_root, source)
            if destination.exists() and not args.overwrite:
                continue
            chain_filter = [args.chain] if args.chain else None
            parsed = parse_PDB(str(source), input_chain_list=chain_filter, ca_only=False)
            dataset = StructureDatasetPDB(parsed, truncate=None, max_length=100000)
            if len(dataset) != 1:
                raise RuntimeError(f"Expected one ProteinMPNN structure for {source}, got {len(dataset)}")
            protein = dataset[0]
            chains = sorted(key[-1:] for key in protein if key.startswith("seq_chain"))
            if args.chain and args.chain not in chains:
                raise ValueError(f"Chain {args.chain} was not found in {source}")
            chain_dict = {protein["name"]: (chains, [])}
            batch = [copy.deepcopy(protein)]
            values = tied_featurize(
                batch,
                device,
                chain_dict,
                None,
                None,
                None,
                None,
                None,
                ca_only=False,
            )
            X, S, mask, lengths, chain_M, chain_encoding, _, _, _, _, chain_M_pos, _, residue_idx, *_ = values
            captured.clear()
            with torch.no_grad():
                model(
                    X,
                    S,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding,
                    torch.zeros_like(chain_M),
                )
            if "embedding" not in captured:
                # Locally modified ProteinMPNN versions may return the encoder directly.
                result = model(
                    X,
                    S,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding,
                    torch.zeros_like(chain_M),
                )
                if result.ndim != 3 or result.shape[-1] != hidden_dim:
                    raise RuntimeError("Could not capture the ProteinMPNN encoder representation")
                captured["embedding"] = result.detach()
            valid_length = int(lengths[0])
            embedding = captured["embedding"][0, :valid_length]
            sequence_parts = [protein[f"seq_chain_{chain}"] for chain in chains]
            sequence = "".join(sequence_parts)
            chain_ids = [chain for chain, part in zip(chains, sequence_parts) for _ in part]
            save_embedding(
                destination,
                embedding,
                model_name=checkpoint_path.stem,
                source=str(source.resolve()),
                sequence=sequence if len(sequence) == valid_length else None,
                chain_ids=chain_ids if len(chain_ids) == valid_length else None,
            )
    finally:
        hook.remove()
