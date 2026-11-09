"""Standalone deterministic extraction of the UniLab HORA SAC teacher.

Adapted from UniLab algos/torch/hora/sac_models.py, common/actor_factory.py,
common/normalization.py, and training/oracle/playback.py (Apache-2.0).
Modified to retain only frozen inference, preserving checkpoint module names,
SiLU/LayerNorm topology, both normalizers, and deterministic action semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .checkpoint import ACTION_DIM, OBS_DIM, PRIV_DIM, validate_payload


class EmpiricalNormalization(nn.Module):
    """Saved empirical statistics; inference can never update these buffers."""

    def __init__(self, shape: int, device: str | torch.device = "cpu", eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, shape, device=device))
        self.register_buffer("_var", torch.ones(1, shape, device=device))
        self.register_buffer("_std", torch.ones(1, shape, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = False) -> torch.Tensor:
        if center:
            return torch.as_tensor((x - self._mean) / (self._std + self.eps))
        return torch.as_tensor(x / (self._std + self.eps))


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    use_layer_norm: bool,
    device: str | torch.device,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for next_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, next_dim, device=device))
        if use_layer_norm:
            layers.append(nn.LayerNorm(next_dim, device=device))
        layers.append(nn.SiLU())
        current_dim = next_dim
    return nn.Sequential(*layers), current_dim


class HoraSACActor(nn.Module):
    """Checkpoint-compatible HORA actor with only its deterministic action path."""

    def __init__(self, network: dict[str, Any], device: str | torch.device = "cpu") -> None:
        super().__init__()
        config = network["actor"]
        hidden_dim = network["actor_hidden_dim"]
        embed_dim = config["priv_info_embed_dim"]
        layer_norm = network["use_layer_norm"]
        self.priv_info_normalizer = EmpiricalNormalization(PRIV_DIM, device)
        self.priv_encoder, priv_out_dim = _build_mlp(
            PRIV_DIM, config["priv_mlp_hidden_dims"], layer_norm, device,
        )
        self.priv_projection = (
            nn.Linear(priv_out_dim, embed_dim, device=device)
            if priv_out_dim != embed_dim else nn.Identity()
        )
        self.actor_trunk, trunk_out_dim = _build_mlp(
            OBS_DIM + embed_dim, (hidden_dim, hidden_dim // 2, hidden_dim // 4),
            layer_norm, device,
        )
        self.action_mean_head = nn.Linear(trunk_out_dim, ACTION_DIM, device=device)
        # Required for a strict load, even though deterministic inference ignores variance.
        self.action_logstd_head = nn.Linear(trunk_out_dim, ACTION_DIM, device=device)
        self.register_buffer("action_scale", torch.ones(ACTION_DIM, device=device))
        self.register_buffer("action_bias", torch.zeros(ACTION_DIM, device=device))

    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        normalized = self.priv_info_normalizer(priv_info, update=False)
        encoded = self.priv_encoder(normalized)
        privileged_embedding = torch.tanh(self.priv_projection(encoded))
        latent = self.actor_trunk(torch.cat([obs, privileged_embedding], dim=-1))
        mean = torch.clamp(self.action_mean_head(latent), -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class OraclePolicy(nn.Module):
    """Frozen raw obs98/privilege205 to deterministic actions29."""

    def __init__(self, payload: dict[str, Any], device: str | torch.device = "cpu") -> None:
        super().__init__()
        validate_payload(payload)
        self.actor = HoraSACActor(payload["metadata"]["network_config"], device)
        self.normalizer = EmpiricalNormalization(OBS_DIM, device)
        self.actor.load_state_dict(payload["state"]["actor"], strict=True)
        self.normalizer.load_state_dict(payload["state"]["obs_normalizer"], strict=True)
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> OraclePolicy:
        # An official algorithm may toggle modes; this teacher must remain frozen.
        return super().train(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != OBS_DIM:
            raise ValueError(f"Oracle actor observations must have shape [batch, {OBS_DIM}]")
        if priv_info.ndim != 2 or priv_info.shape != (obs.shape[0], PRIV_DIM):
            raise ValueError(f"Oracle privileged observations must have shape [batch, {PRIV_DIM}]")
        return self.actor(self.normalizer(obs, update=False), priv_info)
