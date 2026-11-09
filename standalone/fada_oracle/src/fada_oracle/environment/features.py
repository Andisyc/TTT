"""Standalone Torch assembly of the source G1 walk teacher features.

Adapted from UniLab envs/locomotion/g1/{walk_observation,
walk_observation_bindings,fada_privileged,walk_commands}.py and
locomotion/common/base.py (Apache-2.0). No UniLab runtime imports are required.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


_PRIVILEGED_WIDTHS = (
    ("base_linear_velocity", 3),
    ("foot_contact_resultants", 6),
    ("foot_contact_flags", 2),
    ("terrain_heights", 9),
    ("root_clearance", 1),
    ("kp_scale", 29),
    ("kd_scale", 29),
    ("normalized_torque", 29),
    ("ground_friction", 1),
    ("base_com_shift", 3),
    ("added_base_mass", 1),
    ("body_mass_scale", 31),
    ("dof_position_bias", 29),
    ("torque_rfi", 29),
    ("control_delay", 1),
    ("push_interval", 1),
    ("push_velocity", 1),
)


def _batch_field(name: str, value: Tensor, width: int, reference: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape (batch, {width}), got {tuple(value.shape)}")
    if value.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} batch size differs from the preceding fields")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must share the preceding fields' dtype and device")


def _noise_setting(config: Mapping[str, float] | Any, name: str) -> float:
    return float(config[name] if isinstance(config, Mapping) else getattr(config, name))


def actor_observation(
    *,
    gyro: Tensor,
    upvector: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    default_joint_pos: Tensor,
    last_actions: Tensor,
    commands: Tensor,
    gait_phase: Tensor,
    noise_config: Mapping[str, float] | Any,
    add_noise: bool = True,
) -> Tensor:
    """Assemble batched obs98 using the source walk-profile scales.

    ``upvector`` is the torso frame-z axis expressed in WORLD coordinates,
    not body-frame projected gravity. Actions and the two gait phases are
    retained verbatim. Noise is added in sensor units before feature scaling.
    """
    for name, value, width in (
        ("gyro", gyro, 3),
        ("upvector", upvector, 3),
        ("dof_pos", dof_pos, 29),
        ("dof_vel", dof_vel, 29),
        ("last_actions", last_actions, 29),
        ("commands", commands, 3),
        ("gait_phase", gait_phase, 2),
    ):
        _batch_field(name, value, width, gyro)
    if (
        not isinstance(default_joint_pos, Tensor)
        or default_joint_pos.shape not in ((29,), (1, 29), tuple(dof_pos.shape))
        or default_joint_pos.dtype != dof_pos.dtype
        or default_joint_pos.device != dof_pos.device
    ):
        raise ValueError("default_joint_pos must match joint dtype/device and broadcast over (batch, 29)")

    diff = dof_pos - default_joint_pos
    level = _noise_setting(noise_config, "level") if add_noise else 0.0

    def noisy(value: Tensor, scale_name: str) -> Tensor:
        if level <= 0.0:
            return value
        # Source BaseNoise: data + uniform(-1, 1) * level * per-source scale.
        noise = torch.rand_like(value).mul_(2.0).sub_(1.0)
        noise.mul_(level).mul_(_noise_setting(noise_config, scale_name))
        return noise.add_(value)

    return torch.cat(
        (
            noisy(gyro, "scale_gyro") * 0.25,
            -noisy(upvector, "scale_gravity"),
            noisy(diff, "scale_joint_angle"),
            noisy(dof_vel, "scale_joint_vel") * 0.05,
            last_actions,
            commands,
            gait_phase,
        ),
        dim=1,
    )


def pack_privileged(fields: Mapping[str, Tensor], metadata: dict) -> Tensor:
    """Pack actual runtime fields into the checkpoint's exact privilege205.

    Physics-derived values belong to the runtime: in particular, normalized
    torque is tau/tau_max and the zero-mass world body's mass scale is one.
    This function neither estimates those values nor substitutes missing data.
    """
    slices = metadata.get("privileged_field_slices")
    if not isinstance(slices, (list, tuple)) or len(slices) != len(_PRIVILEGED_WIDTHS):
        raise ValueError("privileged_field_slices must declare the exact source privilege205 schema")
    cursor = 0
    for entry, (name, width) in zip(slices, _PRIVILEGED_WIDTHS):
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 3
            or entry[0] != name
            or type(entry[1]) is not int
            or type(entry[2]) is not int
            or entry[1] != cursor
            or entry[2] != cursor + width
        ):
            raise ValueError(f"privileged_field_slices must preserve source order and contiguous width for {name}")
        cursor += width
    missing = [name for name, _ in _PRIVILEGED_WIDTHS if name not in fields]
    if missing:
        raise ValueError(f"missing required privileged fields: {', '.join(missing)}")

    arrays: list[Tensor] = []
    reference = fields[slices[0][0]]
    for name, start, stop in slices:
        value = fields[name]
        _batch_field(name, value, stop - start, reference)
        arrays.append(value)
    packed = torch.cat(arrays, dim=1)
    if not torch.isfinite(packed).all():
        raise ValueError("privileged fields must contain only finite values")
    return packed


def sample_commands(count, ranges, *, standing, straight, small_xy_threshold, device):
    """Source's mutually exclusive standing/straight/general command mixture."""
    low = torch.tensor([ranges[name][0] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")], device=device)
    high = torch.tensor([ranges[name][1] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")], device=device)
    commands = low + torch.rand((count, 3), device=device) * (high - low)
    small = torch.linalg.vector_norm(commands[:, :2], dim=1) < small_xy_threshold
    commands[small, :2] = 0
    draw = torch.rand(count, device=device)
    straight_mask = (draw >= standing) & (draw < standing + straight)
    if straight_mask.any():
        forward_low = max(float(low[0]), small_xy_threshold, torch.finfo(commands.dtype).tiny)
        forward = forward_low + torch.rand(int(straight_mask.sum()), device=device) * (high[0] - forward_low)
        commands[straight_mask, 0] = forward
        commands[straight_mask, 1:] = 0
    commands[draw < standing] = 0
    return commands


__all__ = ["actor_observation", "pack_privileged", "sample_commands"]
