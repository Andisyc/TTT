"""NumPy motion-quality rewards, adapted from FADA revision 290b3a9c.

Frames, norm reductions and clearance denominator follow the reference, not
historical G1 reward aliases. Contact history is explicitly in control ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.base.backend.base import SimBackend
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_matrix_from_quat, np_yaw_from_quat
from unilab.envs.locomotion.g1.motion_quality_config import MotionQualityRewardConfig


@dataclass(frozen=True)
class MotionQualitySample:
    commands: np.ndarray
    phase: np.ndarray
    base_quat: np.ndarray
    base_pos: np.ndarray
    base_lin_vel_w: np.ndarray
    base_ang_vel_w: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    dof_acc: np.ndarray
    torques: np.ndarray
    actions: np.ndarray
    last_actions: np.ndarray
    feet_pos_w: np.ndarray
    feet_vel_w: np.ndarray
    feet_quat_w: np.ndarray
    clearance_heights: np.ndarray
    clearance_vel_w: np.ndarray
    feet_contact: np.ndarray
    undesired_contact: np.ndarray


def compute_motion_quality_terms(
    cfg: MotionQualityRewardConfig,
    state: MotionQualitySample,
    *,
    default_angles: np.ndarray,
    pose_weights: np.ndarray,
    soft_joint_limits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute raw per-env terms; time scaling belongs only to the reducer."""
    rotation = np_matrix_from_quat(state.base_quat)
    linvel = np.einsum("nji,nj->ni", rotation, state.base_lin_vel_w)
    angvel = np.einsum("nji,nj->ni", rotation, state.base_ang_vel_w)
    yaw = np_yaw_from_quat(state.base_quat)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    vx, vy = state.base_lin_vel_w[:, 0], state.base_lin_vel_w[:, 1]
    yaw_velocity = np.column_stack((cosine * vx + sine * vy, -sine * vx + cosine * vy))
    relative_feet_velocity = np.einsum(
        "nji,nfj->nfi", rotation, state.feet_vel_w - state.base_lin_vel_w[:, None, :]
    )
    feet_rotation = np_matrix_from_quat(state.feet_quat_w.reshape(-1, 4)).reshape(-1, 2, 3, 3)
    feet_tilt = np.linalg.norm(feet_rotation[:, :, 2, :2], axis=2)
    leg_phase = np.remainder(state.phase / (2 * np.pi), 1.0)
    expected_contact = leg_phase < cfg.gait_duty_factor
    moving = np.linalg.norm(state.commands[:, :3], axis=1) > cfg.min_command_threshold
    feet_delta = state.feet_pos_w[:, 0] - state.feet_pos_w[:, 1]
    feet_width = np.abs(cosine * feet_delta[:, 1] - sine * feet_delta[:, 0])
    clearance_speed = np.linalg.norm(state.clearance_vel_w[:, :, :2], axis=2)
    clearance_error = np.square(state.clearance_heights - cfg.feet_clearance_target)
    lower, upper = soft_joint_limits[:, 0], soft_joint_limits[:, 1]
    return {
        "tracking_lin_vel": np.exp(-np.sum(np.square(state.commands[:, :2] - yaw_velocity), axis=1) / cfg.tracking_sigma),
        "tracking_ang_vel": np.exp(-np.square(state.commands[:, 2] - angvel[:, 2]) / cfg.tracking_sigma),
        "alive": np.ones(state.dof_pos.shape[0], dtype=state.dof_pos.dtype),
        "penalty_lin_vel_z": np.square(linvel[:, 2]),
        "penalty_ang_vel_xy": np.sum(np.square(angvel[:, :2]), axis=1),
        "penalty_dof_vel": np.sum(np.square(state.dof_vel), axis=1),
        "penalty_dof_acc": np.sum(np.square(state.dof_acc), axis=1),
        "penalty_action_rate": np.sum(np.square(state.actions - state.last_actions), axis=1),
        "penalty_dof_pos_limits": np.sum(np.maximum(lower - state.dof_pos, 0) + np.maximum(state.dof_pos - upper, 0), axis=1),
        "energy": np.sum(np.abs(state.dof_vel * state.torques), axis=1),
        "pose": np.sum(pose_weights * np.square(state.dof_pos - default_angles), axis=1),
        "penalty_orientation": np.sum(np.square(rotation[:, 2, :2]), axis=1),
        "base_height": np.square(state.base_pos[:, 2] - cfg.base_height_target),
        "gait": np.sum(expected_contact == state.feet_contact, axis=1) * moving,
        "feet_slide": np.sum(np.linalg.norm(relative_feet_velocity[:, :, :2], axis=2) * state.feet_contact, axis=1),
        "feet_clearance": np.exp(-np.sum(clearance_error * np.tanh(cfg.feet_clearance_tanh_mult * clearance_speed), axis=1) / cfg.feet_clearance_std),
        "penalty_feet_ori": np.sum(feet_tilt, axis=1),
        "penalty_close_feet_xy": (feet_width < cfg.close_feet_threshold).astype(state.dof_pos.dtype),
        "undesired_contacts": np.sum(state.undesired_contact, axis=1),
    }


def reduce_motion_quality_terms(
    cfg: MotionQualityRewardConfig, terms: dict[str, np.ndarray], ctrl_dt: float
) -> np.ndarray:
    """Fail closed on missing terms and multiply by dt exactly once, without clipping."""
    if terms.keys() != cfg.scales.keys():
        raise ValueError("motion-quality reward data does not match configured terms")
    if not np.isfinite(ctrl_dt) or ctrl_dt <= 0:
        raise ValueError("reward control period must be finite and positive")
    total = np.zeros_like(terms["alive"], dtype=get_global_dtype())
    for name, weight in cfg.scales.items():
        value = terms[name]
        if value.shape != total.shape or not np.isfinite(value).all():
            raise ValueError(f"invalid motion-quality reward data: {name}")
        if weight != 0:
            total += weight * value
    total *= ctrl_dt
    return total


class MotionQualityRewardProvider:
    """Cache metadata at init; all mutable trajectory state lives in env.info."""

    def __init__(self, backend: SimBackend, cfg: MotionQualityRewardConfig,
                 default_angles: np.ndarray, ctrl_dt: float, base_name: str):
        if backend.backend_type != "mujoco":
            raise ValueError("motion-quality Oracle currently requires MuJoCo")
        self.backend, self.cfg, self.ctrl_dt = backend, cfg, ctrl_dt
        self.default_angles = np.asarray(default_angles)
        self.pose_weights = np.asarray(cfg.pose_weights, dtype=get_global_dtype())
        if self.pose_weights.shape != self.default_angles.shape:
            raise ValueError("motion-quality pose weights must match the actuator order")
        self.base_id = backend.get_body_ids([base_name])
        self.feet_ids = backend.get_body_ids(["left_ankle_roll_link", "right_ankle_roll_link"])
        self.site_ids = backend.get_site_ids(["left_foot", "right_foot"])
        body_names = backend.get_body_names()
        self.body_ids = np.arange(1, len(body_names), dtype=np.int32)
        self.feet_columns = np.asarray([int(np.flatnonzero(self.body_ids == i)[0]) for i in self.feet_ids])
        self.undesired_columns = np.asarray([
            i for i, body_id in enumerate(self.body_ids)
            if "ankle" not in body_names[body_id] and "foot_contact_point" not in body_names[body_id]
        ], dtype=np.intp)
        limits = backend.get_joint_range()
        if limits is None or limits.shape != (len(default_angles), 2):
            raise ValueError("motion-quality rewards require ordered joint position limits")
        center = np.mean(limits, axis=1)
        half = 0.5 * (limits[:, 1] - limits[:, 0]) * cfg.soft_dof_pos_limit
        self.soft_joint_limits = np.column_stack((center - half, center + half))
        # Resolve mandatory telemetry now, not by probing capability in the hot path.
        backend.get_actuator_forces()
        backend.get_body_contact_force_norms(self.body_ids)
        backend.get_site_lin_vel_w(self.site_ids)

    def reset_info(self, dof_vel: np.ndarray) -> dict[str, np.ndarray]:
        rows = dof_vel.shape[0]
        period = 1 / self.cfg.gait_frequency
        width = self.cfg.gait_period_randomization_width
        return {
            "mq_gait_period": np.asarray(np.random.uniform(period - width, period + width, rows), dtype=get_global_dtype()),
            "mq_prev_dof_vel": dof_vel.copy(),
            "mq_contact_history": np.zeros((rows, self.cfg.contact_history_steps, len(self.body_ids)), dtype=get_global_dtype()),
        }

    def compute(self, info: dict) -> np.ndarray:
        backend, cfg = self.backend, self.cfg
        dof_vel = backend.get_dof_vel()
        history = info["mq_contact_history"]
        history[:, 1:] = history[:, :-1]
        history[:, 0] = backend.get_body_contact_force_norms(self.body_ids)
        contacts = np.max(history, axis=1) > cfg.contact_force_threshold
        feet_pos, feet_quat, feet_vel, _ = backend.get_body_state_w(self.feet_ids)
        torque = backend.get_actuator_forces()
        if torque.shape != dof_vel.shape:
            raise ValueError("actuator force order must match G1 joint velocity order")
        clearance_pos = np.stack((backend.get_sensor_data("left_foot_pos"), backend.get_sensor_data("right_foot_pos")), axis=1)
        sample = MotionQualitySample(
            commands=info["commands"], phase=info["gait_phase"],
            base_quat=backend.get_base_quat(), base_pos=backend.get_base_pos(),
            base_lin_vel_w=backend.get_base_lin_vel(),
            base_ang_vel_w=backend.get_body_ang_vel_w(self.base_id)[:, 0],
            dof_pos=backend.get_dof_pos(), dof_vel=dof_vel,
            dof_acc=(dof_vel - info["mq_prev_dof_vel"]) / self.ctrl_dt,
            torques=torque, actions=info["current_actions"], last_actions=info["last_actions"],
            feet_pos_w=feet_pos, feet_vel_w=feet_vel, feet_quat_w=feet_quat,
            clearance_heights=clearance_pos[:, :, 2],
            clearance_vel_w=backend.get_site_lin_vel_w(self.site_ids),
            feet_contact=contacts[:, self.feet_columns],
            undesired_contact=contacts[:, self.undesired_columns],
        )
        terms = compute_motion_quality_terms(cfg, sample, default_angles=self.default_angles,
                                             pose_weights=self.pose_weights, soft_joint_limits=self.soft_joint_limits)
        reward = reduce_motion_quality_terms(cfg, terms, self.ctrl_dt)
        np.copyto(info["mq_prev_dof_vel"], dof_vel)
        info["torques"] = torque
        log = info.setdefault("log", {})
        if int(info["steps"][0]) % 4 == 0:
            for name, value in terms.items():
                log[f"reward/{name}"] = float(np.mean(value) * cfg.scales[name])
        return reward
