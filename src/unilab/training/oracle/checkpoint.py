"""Versioned, self-contained motion-quality Oracle artifacts and training resume."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.training.oracle.config import OracleInterface, validate_oracle_config


ENVIRONMENT_FIELDS = (
    "observation_layout", "privileged_field_slices", "body_names",
    "actuated_joint_names", "asset_sha256", "ctrl_dt", "sim_dt", "action_scale",
    "default_joint_pos", "kp", "kd",
)
STATE_FIELDS = (
    "actor", "qnet", "qnet_target", "log_alpha", "actor_optimizer",
    "q_optimizer", "alpha_optimizer", "update_count", "obs_normalizer", "amp_scaler",
)


def build_checkpoint_contract(cfg: Any, env: Any, *, obs_dim: int,
                              critic_obs_dim: int, action_dim: int) -> dict[str, Any]:
    """Capture materialized identities before the factory closes its probe env."""
    interface = validate_oracle_config(cfg)
    return {
        **env.get_motion_quality_deployment_profile(),
        "identity": interface.identity,
        "algorithm": "sac",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "network_config": {
            "obs_dim": obs_dim, "critic_obs_dim": critic_obs_dim,
            "action_dim": action_dim, "actor_hidden_dim": int(cfg.algo.actor_hidden_dim),
            "critic_hidden_dim": int(cfg.algo.critic_hidden_dim),
            "use_layer_norm": bool(cfg.algo.use_layer_norm),
            "num_atoms": int(cfg.algo.num_atoms),
            "v_min": float(cfg.algo.value_support_min),
            "v_max": float(cfg.algo.value_support_max),
            "actor": OmegaConf.to_container(cfg.algo.actor, resolve=True),
        },
        "obs_normalization": True,
        "priv_info_normalization": True,
        "observation_layout": [list(item) for item in interface.actor_obs_layout],
    }


def save_oracle_checkpoint(learner: Any, path: str | Path, iteration: int) -> None:
    metadata = {**learner.checkpoint_contract, "iteration": int(iteration)}
    state = learner.get_state_dict()
    if int(state["update_count"]) != int(iteration):
        raise ValueError("Oracle checkpoint iteration and learner update count differ")
    torch.save({"format": OracleInterface().checkpoint_format,
                "metadata": metadata, "state": state}, path)


def read_oracle_checkpoint(path: str | Path) -> dict[str, Any]:
    """Reject other families and incomplete artifacts; never infer missing stats."""
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    interface = OracleInterface()
    if not isinstance(payload, dict) or payload.get("format") != interface.checkpoint_format:
        raise ValueError(f"Expected {interface.checkpoint_format}; bare/legacy state is unsupported")
    meta, state = payload.get("metadata"), payload.get("state")
    if not isinstance(meta, dict) or not isinstance(state, dict):
        raise ValueError("Oracle checkpoint requires metadata and training state")
    required = (*interface.checkpoint_required_fields, "environment_config", "sim_dt")
    missing = set(required).difference(meta) | set(STATE_FIELDS).difference(state)
    if missing:
        raise ValueError(f"Oracle checkpoint missing fields: {sorted(missing)}")
    cfg = OmegaConf.create(meta["config"])
    if not isinstance(cfg, DictConfig):
        raise ValueError("Oracle checkpoint config must be a mapping")
    validate_oracle_config(cfg)
    if meta["identity"] != interface.identity or meta["algorithm"] != "sac":
        raise ValueError("Oracle checkpoint identity/algorithm is incompatible")
    if (type(meta["iteration"]) is not int or meta["iteration"] < 0
            or meta["iteration"] != state["update_count"]):
        raise ValueError("Oracle checkpoint has inconsistent iteration/update count")
    if not meta["obs_normalization"] or not meta["priv_info_normalization"]:
        raise ValueError("Oracle checkpoint must retain both normalization mechanisms")
    if not isinstance(state["obs_normalizer"], dict) or not state["obs_normalizer"]:
        raise ValueError("Oracle checkpoint is missing observation normalization statistics")
    if not isinstance(state["actor"], dict) or not any(
        key.startswith("priv_info_normalizer.") for key in state["actor"]
    ):
        raise ValueError("Oracle checkpoint is missing privileged normalization statistics")
    return payload


def validate_environment_contract(metadata: dict[str, Any], actual: dict[str, Any]) -> None:
    for name in ENVIRONMENT_FIELDS:
        if metadata[name] != actual[name]:
            raise ValueError(f"Oracle checkpoint environment mismatch: {name}")


def resume_oracle_training(runner: Any, cfg: Any, log_dir: str | Path) -> None:
    """Restore before collector startup. Replay and physical state start fresh."""
    path = cfg.oracle.resume_checkpoint
    if path is None:
        return
    payload = read_oracle_checkpoint(path)
    saved, current = payload["metadata"], runner.learner.checkpoint_contract
    validate_environment_contract(saved, current)
    if saved["network_config"] != current["network_config"]:
        raise ValueError("Oracle resume network configuration differs")
    for section in ("env", "reward", "algo", "oracle"):
        previous, requested = deepcopy(saved["config"][section]), deepcopy(current["config"][section])
        ignored = {"max_iterations", "save_interval"} if section == "algo" else set()
        if section == "oracle":
            ignored = {"resume_checkpoint"}
        for key in ignored:
            previous.pop(key, None)
            requested.pop(key, None)
        if previous != requested:
            raise ValueError(f"Oracle resume changes training semantics: {section}")
    # Asset paths are machine-local; its content and ordered joints were checked above.
    previous_env, requested_env = deepcopy(saved["environment_config"]), deepcopy(current["environment_config"])
    for env_cfg in (previous_env, requested_env):
        env_cfg["scene"].pop("model_file", None)
    if previous_env != requested_env:
        raise ValueError("Oracle resume materialized environment configuration differs")
    iteration = saved["iteration"]
    if int(cfg.algo.max_iterations) <= iteration:
        raise ValueError("Oracle resume max_iterations must exceed the saved global iteration")
    if Path(log_dir).resolve() == Path(path).expanduser().resolve().parent:
        raise ValueError("Oracle resume requires a new output directory")
    runner.learner.load_state_dict(payload["state"])
    runner.start_iteration = iteration
    print(f"[Oracle resume] iteration={iteration}; replay/physics reset, replay warmup enabled", flush=True)
