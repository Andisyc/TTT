"""Frozen motion-quality Oracle at the existing FADA collection seam.

Use ``load_motion_quality_oracle_teacher(path, env, device=...)`` with a
materialized, contract-compatible UniLab env, then pass the returned module as
``teacher_policy`` to the existing source-window collector. Teacher inputs stay
98D plus real privilege; the collector owns student-state projection. This is
not the official PPO loader and does not enable the retired training entrypoint.
For shadow labels, the collection owner prepares
``env.prepare_isolated_rollout_branch()`` before ``env.init_state()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info
from unilab.training.oracle.checkpoint import read_oracle_checkpoint
from unilab.training.oracle.playback import OracleInferencePolicy


class MotionQualityOracleTeacher(OracleInferencePolicy):
    """Inference-only teacher; actions remain in the checkpoint's native U space."""

    def __init__(self, payload: dict[str, Any], env: Any, device: str) -> None:
        super().__init__(payload, env, device)
        network = payload["metadata"]["network_config"]
        self.obs_dim = int(network["obs_dim"])
        self.critic_obs_dim = int(network["critic_obs_dim"])
        self.priv_info_dim = self.critic_obs_dim - self.obs_dim
        self.action_dim = int(network["action_dim"])
        self.checkpoint_identity = {
            "format": payload["format"],
            "identity": payload["metadata"]["identity"],
            "iteration": payload["metadata"]["iteration"],
        }

    @torch.inference_mode()
    def actions_from_env_observation(
        self,
        obs: Mapping[str, np.ndarray],
        info: Mapping[str, Any] | None,
    ) -> np.ndarray:
        """Use unprojected env observations, without updating either normalizer."""
        actor_obs, critic_obs, priv_info = split_hora_obs_with_priv_info(
            dict(obs), None if info is None else dict(info),
        )
        actor_obs = np.asarray(actor_obs, dtype=np.float32)
        if actor_obs.ndim != 2 or actor_obs.shape[1] != self.obs_dim:
            raise ValueError(f"Oracle teacher requires raw observations shaped (*, {self.obs_dim})")
        batch_size = actor_obs.shape[0]
        if critic_obs is None or critic_obs.shape != (batch_size, self.critic_obs_dim):
            raise ValueError(f"Oracle teacher requires critic observations shaped (*, {self.critic_obs_dim})")
        if priv_info is None or priv_info.shape != (batch_size, self.priv_info_dim):
            raise ValueError(f"Oracle teacher requires real privilege shaped (*, {self.priv_info_dim})")
        priv_info = np.asarray(priv_info, dtype=np.float32)
        if not np.isfinite(actor_obs).all() or not np.isfinite(priv_info).all():
            raise ValueError("Oracle teacher inputs must be finite")
        device = next(self.actor.parameters()).device
        actions = self(
            torch.as_tensor(actor_obs, device=device),
            torch.as_tensor(priv_info, device=device),
        ).cpu().numpy()
        if actions.shape != (batch_size, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError("Oracle teacher produced invalid actions")
        return actions


def load_motion_quality_oracle_teacher(
    checkpoint_path: str | Path,
    env: Any,
    *,
    device: str | torch.device = "cpu",
) -> MotionQualityOracleTeacher:
    """Validate artifact and environment before exposing a frozen collection teacher."""
    return MotionQualityOracleTeacher(
        read_oracle_checkpoint(checkpoint_path), env, str(device),
    )
