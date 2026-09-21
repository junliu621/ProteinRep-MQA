"""Model quality assessment with pretrained protein representations."""

from .registry import FEATURE_SPECS, FeatureSpec, get_feature_spec

__all__ = ["FEATURE_SPECS", "FeatureSpec", "get_feature_spec"]
__version__ = "0.1.0"

