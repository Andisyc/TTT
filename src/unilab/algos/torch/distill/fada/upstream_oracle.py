"""UniLab Oracle inference at the original FADA expert boundary.

This module never imports Holosoma, constructs physics, or changes a checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info
from unilab.training.oracle.checkpoint import read_oracle_checkpoint
from unilab.training.oracle.teacher import MotionQualityOracleTeacher


class UpstreamOracleAlgorithm:
    """The inference/load surface consumed by upstream ExpertPolicyWrapper."""

    actor_obs_keys = {"actor_obs": ("obs",), "priv_info": ("priv_info",)}

    def __init__(self, checkpoint: str | Path, env: Any, device: str) -> None:
        self.env = env
        self.device = str(device)
        self._network_config: dict[str, Any] | None = None
        self.load(str(checkpoint))

    def load(self, checkpoint: str) -> None:
        payload = read_oracle_checkpoint(checkpoint)
        network = payload["metadata"]["network_config"]
        if self._network_config is not None and network != self._network_config:
            raise ValueError("intermediate Oracle network differs from the final Oracle")
        # Construct before publication: a failed load leaves the resident teacher intact.
        teacher = MotionQualityOracleTeacher(payload, self.env, self.device)
        self.teacher = teacher
        self._network_config = network
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())

    def get_inference_policy(self):
        return self.inference

    @torch.inference_mode()
    def inference(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.teacher(observations["actor_obs"], observations["priv_info"])

    def _eval_mode(self) -> None:
        self.teacher.eval()


def upstream_oracle_observation(state: Any, *, device: str | torch.device) -> dict[str, torch.Tensor]:
    """Preserve the teacher's raw input, including real privileged information."""
    actor, critic, privilege = split_hora_obs_with_priv_info(state.obs, state.info)
    if critic is None or privilege is None:
        raise ValueError("upstream Oracle requires real critic and privileged observations")
    arrays = {"obs": actor, "priv_info": privilege}
    result = {}
    for name, value in arrays.items():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"invalid upstream Oracle {name}")
        result[name] = torch.tensor(array, dtype=torch.float32, device=device)
    return result
