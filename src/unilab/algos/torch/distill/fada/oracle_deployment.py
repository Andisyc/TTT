"""Restore native student controls from embedded Oracle provenance, without a teacher."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT, FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
)
from unilab.envs.locomotion.g1.motion_quality_config import build_motion_quality_reward_config
from unilab.training import BackendAdapter
from unilab.training.oracle.checkpoint import ENVIRONMENT_FIELDS, validate_environment_contract
from unilab.training.oracle.config import OracleInterface


def oracle_student_source_profile(
    checkpoint: Mapping[str, Any], *, observation_contract: str,
) -> dict[str, Any] | None:
    """Distinguish old students from source-bound native students before env creation."""
    runtime = checkpoint.get("runtime_config", {})
    if runtime is None:
        return None
    if not isinstance(runtime, Mapping):
        raise ValueError("Student runtime_config must be a mapping")
    profile = runtime.get("source_profile")
    if profile is None and runtime.get("source_route") != "unilab_oracle":
        return None
    if not isinstance(profile, Mapping):
        raise ValueError("Oracle student requires embedded runtime_config.source_profile")
    if profile.get("kind") != "unilab_oracle" or type(profile.get("version")) is not int or profile["version"] != 1:
        raise ValueError("Unsupported Oracle student source profile kind/version")
    if (observation_contract not in {FADA_G1_STATE_OBSERVATION_CONTRACT, FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT}
            or profile.get("student_observation_contract") != observation_contract):
        raise ValueError("Oracle student source profile observation contract differs from its policy")
    controls = profile.get("control_contract")
    environment = profile.get("environment_config")
    identity = profile.get("teacher_identity")
    if not isinstance(controls, Mapping) or set(ENVIRONMENT_FIELDS) - set(controls):
        raise ValueError("Oracle student source profile has an incomplete control contract")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("scene"), Mapping):
        raise ValueError("Oracle student source profile has no saved environment/scene")
    interface = OracleInterface()
    if (not isinstance(identity, Mapping) or identity.get("format") != interface.checkpoint_format
            or identity.get("identity") != interface.identity):
        raise ValueError("Oracle student source profile has an invalid teacher identity")
    digest = identity.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Oracle student source profile has an invalid teacher checksum")
    # The recorded teacher path is provenance only: deployment never opens that file.
    return dict(profile)


def build_oracle_student_play_env_cfg(
    profile: Mapping[str, Any], cfg: DictConfig, *, root_dir: str | Path,
) -> dict[str, Any]:
    """Apply play-only perturbations on the saved environment, never legacy PD defaults."""
    environment = deepcopy(dict(profile["environment_config"]))
    relocated = OmegaConf.select(cfg, "training.fada.scene_model_file")
    if relocated in (None, ""):
        relocated = OmegaConf.select(cfg, "env.scene.model_file")
    if relocated not in (None, ""):
        model_file = Path(str(relocated)).expanduser()
        environment["scene"]["model_file"] = str(
            (model_file if model_file.is_absolute() else Path(root_dir) / model_file).resolve()
        )
    # Recording conditions may change; policy timing, gains and state layout may not.
    for name in ("commands", "domain_rand", "noise_config", "runtime_payload", "command_gated_phase_contact"):
        patch = OmegaConf.select(cfg, f"env.{name}")
        if patch is not None:
            environment[name] = OmegaConf.to_container(
                OmegaConf.merge(environment.get(name, {}), patch), resolve=True,
            )
    for name in ("max_episode_seconds", "reset_base_qvel_limit", "standing_reset_base_qvel_limit"):
        value = OmegaConf.select(cfg, f"env.{name}")
        if value is not None:
            environment[name] = value
    play_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if not isinstance(play_cfg, DictConfig):
        raise ValueError("Oracle student playback config must be a mapping")
    play_cfg.training.play_only = True
    restored = BackendAdapter(
        play_cfg, root_dir=root_dir, algo_name="distill",
    ).build_play_env_cfg_override(environment)
    protected = {
        "sim_dt", "ctrl_dt", "control_config", "reward_config", "post_step_forward_sensor",
        *(name for name in environment if name.startswith("gait_")),
    }
    for name in protected:
        if restored.get(name) != environment.get(name):
            raise ValueError(f"Oracle student playback cannot override source {name}")
    restored["reward_config"] = build_motion_quality_reward_config(restored["reward_config"])
    return cast(dict[str, Any], restored)


def validate_oracle_student_environment(profile: Mapping[str, Any], env: Any) -> None:
    """Check the actual materialized controls/assets before wrapper reset or inference."""
    validate_environment_contract(
        dict(profile["control_contract"]), env.get_motion_quality_deployment_profile(),
    )
