"""Independent Oracle inference, including both saved normalizers and real privilege."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.offpolicy.worker import resolve_offpolicy_actor_priv_info
from unilab.base.observations import get_obs_dims, split_obs_dict
from unilab.envs.locomotion.g1.motion_quality_config import build_motion_quality_reward_config
from unilab.envs.locomotion.g1.walk_config import G1WalkEnvCfg
from unilab.training import create_env, log_playback_plan
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.oracle.checkpoint import read_oracle_checkpoint, validate_environment_contract
from unilab.training.oracle.config import validate_oracle_config


class OracleInferencePolicy(nn.Module):
    """Deterministic raw-observation interface; normalization never learns in play."""

    def __init__(self, payload: dict[str, Any], env: Any, device: str) -> None:
        super().__init__()
        meta, state = payload["metadata"], payload["state"]
        validate_environment_contract(meta, env.get_motion_quality_deployment_profile())
        network = meta["network_config"]
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        if (obs_dim, critic_dim, env.action_space.shape[0]) != (
            network["obs_dim"], network["critic_obs_dim"], network["action_dim"],
        ):
            raise ValueError("Oracle checkpoint observation/action dimensions differ")
        self.actor = build_actor(
            "hora_sac", obs_dim, network["action_dim"], network["actor_hidden_dim"],
            network["use_layer_norm"], device,
            priv_info_dim=critic_dim - obs_dim, **network["actor"],
        )
        self.normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        self.actor.load_state_dict(state["actor"], strict=True)
        self.normalizer.load_state_dict(state["obs_normalizer"], strict=True)
        self.eval()
        self.requires_grad_(False)

    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        return self.actor.explore(
            self.normalizer(obs, update=False), priv_info, deterministic=True,
        )


def build_oracle_play_environment(
    payload: dict[str, Any],
    cfg: DictConfig | None = None,
    *,
    num_envs: int | None = None,
    device: str | None = None,
    env_patch: dict[str, Any] | None = None,
) -> tuple[Any, DictConfig]:
    """Restore the artifact environment, then apply explicit playback-only settings."""
    from unilab.training.offpolicy.playback import default_device

    play_cfg = OmegaConf.create(payload["metadata"]["config"])
    if not isinstance(play_cfg, DictConfig):
        raise ValueError("Oracle checkpoint config must be a mapping")
    # These are operation-only overrides; network, control, and rewards come from the artifact.
    if cfg is not None:
        for key in (
            "device", "play_env_num", "play_steps", "play_render_mode", "cam_distance",
            "cam_elevation", "cam_azimuth",
        ):
            play_cfg.training[key] = cfg.training[key]
    if num_envs is not None:
        play_cfg.training.play_env_num = num_envs
    if device is not None:
        play_cfg.training.device = device
    play_cfg.training.play_only = True
    play_cfg.oracle.resume_checkpoint = None
    env_override = deepcopy(payload["metadata"]["environment_config"])
    # Relocate assets without weakening the checkpoint's content/order checks.
    model_file = OmegaConf.select(cfg, "env.scene.model_file") if cfg is not None else None
    env_override["scene"]["model_file"] = (
        str(Path(model_file).expanduser().resolve())
        if model_file is not None else G1WalkEnvCfg().scene.model_file
    )
    adapter = BackendAdapter(play_cfg, root_dir=Path(__file__).parents[4], algo_name="sac")
    env_override = adapter.build_play_env_cfg_override(env_override)
    if env_patch is not None:
        env_override = cast(dict[str, Any], OmegaConf.to_container(
            OmegaConf.merge(env_override, env_patch), resolve=True,
        ))
    env_override["reward_config"] = build_motion_quality_reward_config(env_override["reward_config"])
    play_cfg.training.device = default_device(torch, play_cfg.training.device)
    env = cast(Any, create_env(
        play_cfg, num_envs=int(play_cfg.training.play_env_num), env_cfg_override=env_override,
    ))
    return env, play_cfg


def play_oracle(cfg: DictConfig) -> str | None:
    """Restore the saved training config, then apply only the nominal play profile."""
    validate_oracle_config(cfg)
    path = cfg.training.play_checkpoint_path
    if not path:
        raise ValueError("Oracle play requires training.play_checkpoint_path")
    payload = read_oracle_checkpoint(path)
    env, play_cfg = build_oracle_play_environment(payload, cfg)
    device = str(play_cfg.training.device)
    try:
        policy = OracleInferencePolicy(payload, env, device)
        env.init_state()
        current: tuple[np.ndarray, np.ndarray] | None = None

        def observe(obs_dict: dict[str, np.ndarray], info: dict) -> np.ndarray:
            nonlocal current
            obs_np, critic_np = split_obs_dict(obs_dict)
            obs_np = np.asarray(obs_np, dtype=np.float32)
            priv_np = resolve_offpolicy_actor_priv_info(
                algo_type="hora_sac", obs_np=obs_np,
                critic_np=np.asarray(critic_np, dtype=np.float32), info=info,
            )
            if priv_np is None:
                raise ValueError("Oracle playback requires real per-frame privileged observations")
            current = obs_np, priv_np
            return obs_np

        def initialize() -> np.ndarray:
            obs_dict, info = env.reset(np.arange(env.num_envs, dtype=np.int32))
            return observe(obs_dict, info)

        def step(_obs: np.ndarray) -> np.ndarray:
            if current is None:
                raise RuntimeError("Oracle playback has not been initialized")
            obs_np, priv_np = current
            actions = policy(
                torch.as_tensor(obs_np, device=device),
                torch.as_tensor(priv_np, device=device),
            ).cpu().numpy()
            state = env.step(actions)
            return observe(state.obs, state.info)

        with torch.inference_mode():
            return env.run_playback_mode(
                play_render_mode=play_cfg.training.play_render_mode,
                play_steps=int(play_cfg.training.play_steps),
                output_video=str(Path(path).expanduser().parent / "play_video.mp4"),
                initialize=initialize, step=step,
                camera_kwargs={name: play_cfg.training[name] for name in (
                    "cam_distance", "cam_elevation", "cam_azimuth",
                )},
                on_plan=log_playback_plan,
            )
    finally:
        env.close()
