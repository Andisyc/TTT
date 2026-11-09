"""Source-faithful assets and official FADA experiment configuration."""

from .assets import PreparedScene, materialize_scene
from .config import build_experiment_config

__all__ = ["PreparedScene", "materialize_scene", "build_experiment_config"]
