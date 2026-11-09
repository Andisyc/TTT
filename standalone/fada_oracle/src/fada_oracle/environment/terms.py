"""Observation and reset terms registered only in the independent source task."""
from __future__ import annotations

import math

import torch
from holosoma.managers.command.terms.locomotion import LocomotionCommand, LocomotionGait

from .features import sample_commands


def oracle_obs(env):
    return env.oracle_observation()


def oracle_priv(env):
    return env.source_views().privileged(env)


def source_physics(env, env_ids=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    env.source_views().randomize(env_ids, env.oracle_metadata["environment_config"]["domain_rand"])


def source_failure(env, *, max_tilt_deg, min_base_height):
    views = env.source_views()
    up_name = env.oracle_metadata["environment_config"]["sensor"]["upvector"]
    up_z = views.sensor(up_name)[:, 2]
    clearance = views.xpos[:, views.base_id, 2] - views.floor_z
    return (up_z < math.cos(math.radians(float(max_tilt_deg)))) | (clearance < float(min_base_height))


class SourceCommand(LocomotionCommand):
    def _resample(self, env_ids):
        if env_ids.numel():
            settings = self.env.oracle_metadata["environment_config"]["commands"]
            self.commands[env_ids] = sample_commands(
                env_ids.numel(), self.command_ranges, standing=self.stand_prob,
                straight=float(settings["rel_straight_envs"]),
                small_xy_threshold=float(settings["small_xy_threshold"]), device=self.env.device,
            )


class SourceGait(LocomotionGait):
    """Source raw radians: fixed-frequency continuous clock, including standing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mean_gait_freq = float(cfg.params["gait_frequency"])

    def setup(self):
        super().setup()
        self.reset(None)

    def reset(self, env_ids):
        idx = self._ensure_index_tensor(env_ids)
        phase = torch.empty(idx.numel(), device=self.env.device).uniform_(0, 2 * torch.pi)
        self.phase_offset[idx, 0] = phase
        self.phase_offset[idx, 1] = phase + torch.pi
        self.phase[idx] = self.phase_offset[idx]
        self.gait_freq[idx] = self.mean_gait_freq
        self.phase_dt[idx] = 2 * torch.pi * self.env.dt * self.mean_gait_freq

    def step(self):
        phase = self.phase_offset + self.env.episode_length_buf[:, None] * self.phase_dt
        self.phase.copy_(torch.remainder(phase, 2 * torch.pi))

    def resample_frequency(self, env_ids):
        idx = self._ensure_index_tensor(env_ids)
        self.gait_freq[idx] = self.mean_gait_freq
        self.phase_dt[idx] = 2 * torch.pi * self.env.dt * self.mean_gait_freq
