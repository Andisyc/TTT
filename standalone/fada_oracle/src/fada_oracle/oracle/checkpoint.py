"""Read-only adapter for the UniLab motion-quality Oracle artifact.

Adapted from UniLab training/oracle/checkpoint.py (Apache-2.0).
Modified for standalone Torch inference: no training or environment dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

FORMAT = "unilab_motion_quality_oracle_v1"
OBS_DIM = 98
PRIV_DIM = 205
ACTION_DIM = 29
OBSERVATION_LAYOUT = (
    ("gyro", 3), ("gravity", 3), ("dof_pos", 29), ("dof_vel", 29),
    ("actions", 29), ("command", 3), ("gait_phase", 2),
)
_METADATA_FIELDS = (
    "identity", "algorithm", "iteration", "config", "network_config",
    "obs_normalization", "priv_info_normalization", "observation_layout",
    "privileged_field_slices", "body_names", "actuated_joint_names",
    "asset_sha256", "ctrl_dt", "sim_dt", "action_scale", "default_joint_pos",
    "kp", "kd", "environment_config",
)
_STATE_FIELDS = (
    "actor", "qnet", "qnet_target", "log_alpha", "actor_optimizer",
    "q_optimizer", "alpha_optimizer", "update_count", "obs_normalizer", "amp_scaler",
)


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Oracle {name} must be an integer >= {minimum}")


def _validate_normalizer(state: Any, width: int, label: str) -> None:
    if not isinstance(state, dict) or set(state) != {"_mean", "_var", "_std", "count"}:
        raise ValueError(f"Oracle {label} requires complete normalization statistics")
    for key, value in state.items():
        expected = () if key == "count" else (1, width)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"Oracle {label}.{key} must have shape {expected}")
        if key == "count":
            if value.dtype != torch.int64:
                raise ValueError(f"Oracle {label}.count must be int64")
        elif not value.is_floating_point():
            raise ValueError(f"Oracle {label}.{key} must be floating point")


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate the supported inference family without changing the artifact."""
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT}; bare/legacy state is unsupported")
    metadata, state = payload.get("metadata"), payload.get("state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("Oracle checkpoint requires metadata and training state")
    missing = set(_METADATA_FIELDS).difference(metadata) | set(_STATE_FIELDS).difference(state)
    if missing:
        raise ValueError(f"Oracle checkpoint missing fields: {sorted(missing)}")
    if metadata["identity"] != "motion_quality_oracle_v1" or metadata["algorithm"] != "sac":
        raise ValueError("Oracle checkpoint identity/algorithm is incompatible")
    if not isinstance(metadata["config"], dict) or not isinstance(metadata["environment_config"], dict):
        raise ValueError("Oracle configuration metadata must be mappings")
    if (type(metadata["iteration"]) is not int or metadata["iteration"] < 0
            or type(state["update_count"]) is not int
            or metadata["iteration"] != state["update_count"]):
        raise ValueError("Oracle checkpoint has inconsistent iteration/update count")
    if metadata["obs_normalization"] is not True or metadata["priv_info_normalization"] is not True:
        raise ValueError("Oracle checkpoint must retain both normalization mechanisms")
    layout = metadata["observation_layout"]
    if not isinstance(layout, (list, tuple)) or list(layout) != [list(item) for item in OBSERVATION_LAYOUT]:
        raise ValueError("Oracle observation layout is incompatible with obs98")
    slices = metadata["privileged_field_slices"]
    if not isinstance(slices, (list, tuple)) or not slices:
        raise ValueError("Oracle privileged field slices are missing")
    end = 0
    for field in slices:
        if (not isinstance(field, (list, tuple)) or len(field) != 3
                or not isinstance(field[0], str) or type(field[1]) is not int
                or type(field[2]) is not int or field[1] != end or field[2] <= end):
            raise ValueError("Oracle privileged field slices must be ordered and contiguous")
        end = field[2]
    if end != PRIV_DIM:
        raise ValueError("Oracle privileged field slices must span privilege205")
    network = metadata["network_config"]
    required_network = {"obs_dim", "critic_obs_dim", "action_dim", "actor_hidden_dim", "use_layer_norm", "actor"}
    if not isinstance(network, dict) or required_network.difference(network):
        raise ValueError("Oracle network configuration is incomplete")
    for name, expected in (("obs_dim", OBS_DIM), ("critic_obs_dim", OBS_DIM + PRIV_DIM), ("action_dim", ACTION_DIM)):
        if type(network[name]) is not int or network[name] != expected:
            raise ValueError(f"Oracle {name} must be {expected}")
    _positive_integer(network["actor_hidden_dim"], "actor_hidden_dim", minimum=4)
    if type(network["use_layer_norm"]) is not bool:
        raise ValueError("Oracle use_layer_norm must be boolean")
    actor = network["actor"]
    if not isinstance(actor, dict):
        raise ValueError("Oracle actor configuration must be a mapping")
    if actor.get("priv_info_normalization") is not True or actor.get("fixed_privileged_input") is not False:
        raise ValueError("Oracle actor requires normalized real privileged input")
    _positive_integer(actor.get("priv_info_embed_dim"), "priv_info_embed_dim")
    widths = actor.get("priv_mlp_hidden_dims")
    if not isinstance(widths, (list, tuple)) or not widths:
        raise ValueError("Oracle privileged encoder widths are missing")
    for width in widths:
        _positive_integer(width, "priv_mlp_hidden_dims")
    _validate_normalizer(state["obs_normalizer"], OBS_DIM, "obs_normalizer")
    actor_state = state["actor"]
    if not isinstance(actor_state, dict) or any(not isinstance(key, str) for key in actor_state):
        raise ValueError("Oracle actor state must be a named tensor mapping")
    prefix = "priv_info_normalizer."
    _validate_normalizer(
        {key[len(prefix):]: value for key, value in actor_state.items() if key.startswith(prefix)},
        PRIV_DIM, "priv_info_normalizer",
    )
    return payload


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an artifact on CPU and return its original, unmodified dictionary."""
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    return validate_payload(payload)
