"""The official FADA teacher interface, without a training or environment owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .checkpoint import read_checkpoint
from .policy import OraclePolicy


class OracleAlgorithm:
    """Load and atomically replace a frozen teacher compatible with FADA inference."""

    actor_obs_keys = {"actor_obs": ("oracle_obs",), "priv_info": ("oracle_priv",)}

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.teacher: OraclePolicy
        self.checkpoint: str
        self.metadata: dict[str, Any]
        self._network_config: dict[str, Any]
        self.load(checkpoint)

    def load(self, path: str | Path) -> None:
        """Keep the resident teacher and path untouched if validation/loading fails."""
        resolved = str(Path(path).expanduser().resolve())
        payload = read_checkpoint(resolved)
        network = payload["metadata"]["network_config"]
        if hasattr(self, "_network_config") and network != self._network_config:
            raise ValueError("Oracle replacement network configuration differs")
        candidate = OraclePolicy(payload, device=self.device)
        metadata = deepcopy(payload["metadata"])
        network_config = deepcopy(network)
        self.teacher = candidate
        self.checkpoint = resolved
        self.metadata = metadata
        self._network_config = network_config

    def _eval_mode(self) -> None:
        self.teacher.eval()

    def get_inference_policy(self) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
        self._eval_mode()

        @torch.no_grad()
        def inference(observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return self.teacher(
                observations["actor_obs"].to(self.device),
                observations["priv_info"].to(self.device),
            )

        return inference
