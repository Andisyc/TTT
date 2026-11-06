"""Pure G1 command construction and command-owned gait decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import zero_small_xy_commands


@dataclass(frozen=True)
class G1CommandGaitState:
    commands: np.ndarray
    is_null: np.ndarray
    intensity: np.ndarray
    frequency: np.ndarray


def _command_matrix(commands: np.ndarray) -> np.ndarray:
    values = np.asarray(commands, dtype=get_global_dtype())
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"commands must have shape (N, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("commands must be finite")
    return values


def canonicalize_g1_commands(
    commands: np.ndarray,
    *,
    xy_dead_zone: float,
    yaw_dead_zone: float,
) -> np.ndarray:
    """Map the joint planar/yaw dead zone to one exact null command."""

    xy_dead_zone = float(xy_dead_zone)
    yaw_dead_zone = float(yaw_dead_zone)
    if (
        not np.isfinite([xy_dead_zone, yaw_dead_zone]).all()
        or min(xy_dead_zone, yaw_dead_zone) < 0.0
    ):
        raise ValueError("command dead zones must be finite and non-negative")
    canonical = np.array(_command_matrix(commands), copy=True)
    null = (np.linalg.norm(canonical[:, :2], axis=1) < xy_dead_zone) & (
        np.abs(canonical[:, 2]) < yaw_dead_zone
    )
    canonical[null] = 0.0
    return canonical


def apply_g1_command_dead_zone(commands: np.ndarray, command_cfg: Any) -> np.ndarray:
    """Apply the standalone ablation without changing phase or reward state."""

    if not getattr(command_cfg, "dead_zone_enabled", False):
        return commands
    return canonicalize_g1_commands(
        commands,
        xy_dead_zone=float(command_cfg.dead_zone_xy),
        yaw_dead_zone=float(command_cfg.dead_zone_yaw),
    )


def resolve_g1_command_gait_state(
    commands: np.ndarray,
    *,
    xy_dead_zone: float,
    yaw_dead_zone: float,
    linear_intensity_span: float,
    yaw_intensity_span: float,
    min_frequency: float,
    max_frequency: float,
) -> G1CommandGaitState:
    """Resolve gait state from an already canonical command matrix.

    Dead-zone mutation belongs to command producers. Consumers use exact zero as
    the sole null identity and only use dead-zone widths to scale intensity.
    """

    canonical = _command_matrix(commands)
    parameters = np.asarray(
        [linear_intensity_span, yaw_intensity_span, min_frequency, max_frequency],
        dtype=np.float64,
    )
    if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
        raise ValueError("command intensity spans and gait frequencies must be finite and positive")
    if max_frequency < min_frequency:
        raise ValueError("max_frequency must be at least min_frequency")

    is_null = np.all(canonical == 0.0, axis=1)
    linear_excess = np.maximum(
        np.linalg.norm(canonical[:, :2], axis=1) - float(xy_dead_zone), 0.0
    ) / float(linear_intensity_span)
    yaw_excess = np.maximum(np.abs(canonical[:, 2]) - float(yaw_dead_zone), 0.0) / float(
        yaw_intensity_span
    )
    intensity = np.clip(np.maximum(linear_excess, yaw_excess), 0.0, 1.0)
    intensity[is_null] = 0.0
    frequency = float(min_frequency) + (float(max_frequency) - float(min_frequency)) * intensity
    frequency[is_null] = 0.0
    return G1CommandGaitState(
        commands=np.asarray(canonical, dtype=get_global_dtype()),
        is_null=np.asarray(is_null, dtype=np.bool_),
        intensity=np.asarray(intensity, dtype=get_global_dtype()),
        frequency=np.asarray(frequency, dtype=get_global_dtype()),
    )


def sample_g1_walk_commands(env: Any, num_samples: int) -> np.ndarray:
    low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
    high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
    # Use the process RNG seeded by apply_training_seed, including transition draws.
    commands = np.asarray(
        np.random.uniform(low=low, high=high, size=(num_samples, 3)),
        dtype=get_global_dtype(),
    )
    phase_contact_cfg = getattr(env.cfg, "command_gated_phase_contact", None)
    command_gated = bool(getattr(phase_contact_cfg, "enabled", False))
    if not command_gated and not getattr(env.cfg.commands, "dead_zone_enabled", False):
        zero_small_xy_commands(
            commands,
            threshold=float(getattr(env.cfg.commands, "small_xy_threshold", 0.0)),
        )
    standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
    transition_prob = float(getattr(env.cfg.commands, "rel_transition_envs", 0.0))
    straight_prob = float(getattr(env.cfg.commands, "rel_straight_envs", 0.0))
    standing_prob = min(max(standing_prob, 0.0), 1.0)
    transition_prob = min(max(transition_prob, 0.0), max(1.0 - standing_prob, 0.0))
    straight_prob = min(
        max(straight_prob, 0.0), max(1.0 - standing_prob - transition_prob, 0.0)
    )
    draw = np.random.uniform(size=(num_samples,))
    if transition_prob > 0.0:
        transition_low = np.asarray(
            env.cfg.commands.transition_vel_limit[0], dtype=get_global_dtype()
        )
        transition_high = np.asarray(
            env.cfg.commands.transition_vel_limit[1], dtype=get_global_dtype()
        )
        transition = (draw >= standing_prob) & (draw < standing_prob + transition_prob)
        if np.any(transition):
            commands[transition] = np.asarray(
                np.random.uniform(
                    low=transition_low,
                    high=transition_high,
                    size=(int(np.sum(transition)), 3),
                ),
                dtype=get_global_dtype(),
            )
    if straight_prob > 0.0:
        start = standing_prob + transition_prob
        straight = (draw >= start) & (draw < start + straight_prob)
        num_straight = int(np.count_nonzero(straight))
        if num_straight > 0:
            forward_low = max(
                float(low[0]),
                float(getattr(env.cfg.commands, "small_xy_threshold", 0.0)),
                float(np.finfo(commands.dtype).tiny),
            )
            commands[straight, 0] = np.random.uniform(
                low=forward_low, high=float(high[0]), size=num_straight
            )
            commands[straight, 1:3] = 0.0
    if command_gated:
        commands = canonicalize_g1_commands(
            commands,
            xy_dead_zone=float(phase_contact_cfg.command_xy_dead_zone),
            yaw_dead_zone=float(phase_contact_cfg.command_yaw_dead_zone),
        )
    if standing_prob > 0.0:
        commands[draw < standing_prob] = 0.0
    if getattr(env.cfg.commands, "heading_command", False):
        commands[:, 2] = 0.0
    commands = apply_g1_command_dead_zone(commands, env.cfg.commands)
    return np.asarray(commands, dtype=get_global_dtype())


def command_resample_mask(steps: np.ndarray, *, interval_steps: int) -> np.ndarray:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    step_values = np.asarray(steps)
    return (step_values > 0) & ((step_values % int(interval_steps)) == 0)


def freeze_inactive_gait_phase(
    gait_phase: np.ndarray,
    active: np.ndarray,
    stand_phase: np.ndarray,
) -> np.ndarray:
    phase = np.array(gait_phase, copy=True)
    phase[~np.asarray(active, dtype=bool)] = np.asarray(stand_phase, dtype=phase.dtype)
    return phase
