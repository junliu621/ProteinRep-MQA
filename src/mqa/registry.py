from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    checkpoint: str
    dimension: int
    input_kind: str
    head_hidden_dims: tuple[int, ...]
    fusion_hidden_dims: tuple[int, ...]


FEATURE_SPECS = {
    "esm2": FeatureSpec(
        "esm2", "esm2_t33_650M_UR50D", 1280, "sequence", (512, 256, 128), (512, 256, 64)
    ),
    "esm3": FeatureSpec(
        "esm3", "esm3-open", 1536, "structure", (512, 256, 128), (512, 256, 64)
    ),
    "saprot": FeatureSpec(
        "saprot", "SaProt_650M_AF2", 1280, "structure", (512, 256, 128), (512, 256, 64)
    ),
    "prostt5": FeatureSpec(
        "prostt5", "Rostlab/ProstT5", 1024, "sequence", (512, 256, 128), (512, 256, 64)
    ),
    "esmif1": FeatureSpec(
        "esmif1", "esm_if1_gvp4_t16_142M_UR50", 512, "structure", (256, 128), (256, 64)
    ),
    "proteinmpnn": FeatureSpec(
        "proteinmpnn", "v_48_020", 128, "structure", (256, 128), (64,)
    ),
}

ALIASES = {
    "prostt": "prostt5",
    "prost": "prostt5",
    "esm-if1": "esmif1",
    "mpnn": "proteinmpnn",
}


def canonical_feature_name(name: str) -> str:
    normalized = name.strip().lower()
    return ALIASES.get(normalized, normalized)


def get_feature_spec(name: str) -> FeatureSpec:
    name = canonical_feature_name(name)
    try:
        return FEATURE_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(FEATURE_SPECS))
        raise ValueError(f"Unknown feature '{name}'. Choose one of: {choices}") from exc

