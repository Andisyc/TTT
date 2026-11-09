"""Explicit Holosoma rollout surface over a UniLab-owned G1 environment."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada.upstream_oracle import upstream_oracle_observation


class _CommandConfig:
    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def locomotion_command_resampling_time(self) -> float:
        return float(self.env.cfg.commands.resampling_time)

    @locomotion_command_resampling_time.setter
    def locomotion_command_resampling_time(self, value: float) -> None:
        self.env.cfg.commands.resampling_time = float(value)


class _Commands:
    def __init__(self, owner: UpstreamEnvironment) -> None:
        self.owner = owner
        self.command_cfg = _CommandConfig(owner.env)
        self.allow_eval_randomization = True
        self.commands = torch.empty((owner.num_envs, 3), device=owner.device)

    def get_state(self, name: str) -> Any:
        if name == "locomotion_gait":
            return SimpleNamespace(phase=self.owner.tensor(self.owner.env.state.info["gait_phase"]))
        if name == "locomotion_command":
            return self
        raise KeyError(f"unsupported upstream command state: {name}")


class _SimulatorObservation:
    def __init__(self, owner: UpstreamEnvironment) -> None:
        self.owner = owner

    @property
    def dof_pos(self) -> torch.Tensor:
        return self.owner.tensor(self.owner.env.get_dof_pos())

    @property
    def dof_vel(self) -> torch.Tensor:
        return self.owner.tensor(self.owner.env.get_dof_vel())

    @property
    def robot_root_states(self) -> torch.Tensor:
        owner, env = self.owner, self.owner.env
        quat = owner.tensor(env.get_base_quat())[:, (1, 2, 3, 0)]
        angular = owner.tensor(env.get_body_ang_vel_w(owner.base_body_ids))[:, 0]
        return torch.cat((owner.tensor(env.get_base_pos()), quat,
                          owner.tensor(env.get_base_lin_vel()), angular), dim=1)


class UpstreamEnvironment:
    """Translate only the original trainer's rollout API, without a new sampler."""

    def __init__(self, env: Any, *, device: str) -> None:
        self.env = env
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)
        self.dt = float(env.cfg.ctrl_dt)
        self.robot_config = SimpleNamespace(
            actions_dim=29, control=SimpleNamespace(clip_actions=False),
        )
        self.base_body_ids = env.get_body_ids(("pelvis",))
        self.default_dof_pos = self.tensor(np.asarray(env.default_angles).reshape(1, 29))
        self.command_manager = _Commands(self)
        self.simulator = _SimulatorObservation(self)

    def tensor(self, value: Any) -> torch.Tensor:
        return torch.tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    @property
    def base_quat(self) -> torch.Tensor:
        return self.tensor(self.env.get_base_quat())[:, (1, 2, 3, 0)]

    def synchronize(self) -> dict[str, torch.Tensor]:
        self.command_manager.commands.copy_(self.tensor(self.env.state.info["commands"][:, :3]))
        return upstream_oracle_observation(self.env.state, device=self.device)

    def reset_all(self) -> dict[str, torch.Tensor]:
        self.env.reset_all()
        return self.synchronize()

    def step(self, inputs: dict[str, torch.Tensor]):
        actions = inputs["actions"]
        if actions.shape != (self.num_envs, 29) or not bool(torch.isfinite(actions).all()):
            raise ValueError("upstream actions must be finite and shaped (num_envs, 29)")
        self.env.state.info["commands"][:, :3] = self.command_manager.commands.detach().cpu().numpy()
        state = self.env.step(actions.detach().cpu().numpy())
        rewards = self.tensor(state.reward)
        dones = torch.tensor(state.terminated | state.truncated, device=self.device, dtype=torch.bool)
        return self.synchronize(), rewards, dones, {"to_log": state.info.get("log", {})}

    def capture(self) -> dict[str, Any]:
        return {
            "environment": self.env.capture_rollout_snapshot(),
            "backend": self.env.capture_transfer_rollout_state(),
            "resample_time": self.command_manager.command_cfg.locomotion_command_resampling_time,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        # Model parameters must precede the ordinary physics/task/RNG restore.
        self.env.restore_transfer_rollout_state(snapshot["backend"])
        self.env.restore_rollout_snapshot(snapshot["environment"])
        self.command_manager.command_cfg.locomotion_command_resampling_time = snapshot["resample_time"]
        self.synchronize()

    def close(self) -> None:
        self.env.close()
