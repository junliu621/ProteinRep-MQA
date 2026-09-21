from .deepaccnet import DeepAccNetFusion, calculate_lddt, deepaccnet_loss
from .head import ResidueQualityHead
from ..registry import get_feature_spec

__all__ = ["DeepAccNetFusion", "ResidueQualityHead", "calculate_lddt", "create_fusion_model", "deepaccnet_loss"]


def create_fusion_model(feature: str, **kwargs) -> DeepAccNetFusion:
    spec = get_feature_spec(feature)
    return DeepAccNetFusion(
        spec.dimension,
        spec.fusion_hidden_dims,
        projection_name="mlp" if spec.name == "saprot" else "esm_mlp",
        **kwargs,
    )
