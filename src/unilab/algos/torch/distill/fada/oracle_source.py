"""Construct an uninitialized Oracle collection source and its portable profile."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.contracts.workflow import file_sha256
from unilab.algos.torch.distill.fada.observation import FADA_G1_STATE_OBSERVATION_CONTRACT
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.g1.motion_quality_config import build_motion_quality_reward_config
from unilab.training import create_env
from unilab.training.oracle.checkpoint import ENVIRONMENT_FIELDS, read_oracle_checkpoint
from unilab.training.oracle.config import OracleInterface
from unilab.training.oracle.teacher import MotionQualityOracleTeacher


def build_oracle_fada_source(
    checkpoint_path: str | Path,
    *,
    num_envs: int,
    device: str = "cpu",
    scene_model_file: str | Path | None = None,
) -> tuple[Any, MotionQualityOracleTeacher, dict[str, Any]]:
    """Return ``(env, frozen_teacher, source_profile)`` without resetting or stepping.

    Sampling keeps the saved source environment, including its randomization;
    playback profiles are not applied. Asset relocation still requires the same
    content and ordered deployment contract. The caller owns ``env.close()`` and
    must prepare any isolated rollout branch before initializing the state.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or num_envs <= 0:
        raise ValueError("Oracle FADA source num_envs must be a positive integer")
    path = Path(checkpoint_path).expanduser().resolve()
    payload = read_oracle_checkpoint(path)
    metadata = payload["metadata"]
    interface = OracleInterface()
    expected_layout = [list(term) for term in interface.actor_obs_layout]
    if metadata["observation_layout"] != expected_layout:
        raise ValueError("Oracle FADA source observation layout differs from g1_fada_state_v2 input")
    network = metadata["network_config"]
    if (network["obs_dim"], network["action_dim"]) != (
        interface.actor_obs_dim, interface.action_dim,
    ):
        raise ValueError("Oracle FADA source requires 98 actor observations and 29 actions")

    source_cfg = OmegaConf.create(metadata["config"])
    if not isinstance(source_cfg, DictConfig):
        raise ValueError("Oracle checkpoint config must be a mapping")
    source_cfg.training.play_only = False
    source_cfg.training.device = str(device)
    source_cfg.oracle.resume_checkpoint = None
    environment_config = deepcopy(metadata["environment_config"])
    if scene_model_file is not None:
        environment_config["scene"]["model_file"] = str(
            Path(scene_model_file).expanduser().resolve()
        )
    profile = {
        "kind": "unilab_oracle",
        "version": 1,
        "teacher_identity": {
            "format": payload["format"],
            "identity": metadata["identity"],
            "iteration": metadata["iteration"],
            "checkpoint_path": str(path),
            "checkpoint_sha256": file_sha256(path),
        },
        "environment_config": deepcopy(environment_config),
        "control_contract": {name: deepcopy(metadata[name]) for name in ENVIRONMENT_FIELDS},
        "student_observation_contract": FADA_G1_STATE_OBSERVATION_CONTRACT,
    }
    environment_config["reward_config"] = build_motion_quality_reward_config(
        environment_config["reward_config"]
    )
    ensure_registries(packages=("unilab.envs.locomotion.g1",))
    env = create_env(
        source_cfg, num_envs=int(num_envs), env_cfg_override=environment_config,
    )
    try:
        teacher = MotionQualityOracleTeacher(payload, env, str(device))
        return env, teacher, profile
    except BaseException as error:
        try:
            env.close()
        except BaseException as cleanup_error:
            error.add_note(f"Oracle source environment cleanup also failed: {cleanup_error!r}")
        raise
