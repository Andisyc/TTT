"""Framework reward bindings for :class:`G1WalkEnv`.

The adapter owns no mutable state. It exposes the method-shaped reward surface
required by the environment reward registry while calculations remain in the
walk reward and math owners.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.fada_privileged import split_net_contact_sensor
from unilab.envs.locomotion.g1.walk_config import (
    G1RewardConfig,
    GaitConstraintConfig,
    RewardModeConfig,
)
from unilab.envs.locomotion.g1.walk_math import (
    command_phase_height_cost,
    command_phase_height_targets,
    compute_aggregated_foot_contact,
    compute_aggregated_foot_contact_count,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_forward_command_mask,
    compute_forward_progress_failure,
    compute_forward_speed_gate,
    compute_planar_command_progress_ratio,
    compute_planar_command_speed_gate,
    compute_gait_phase_contact_violation,
    compute_gait_phase_contrast_violation,
    compute_gait_phase_height_violation,
    compute_tracking_gate,
    original_command_height_targets,
    phase_height_targets_v3,
)
from unilab.envs.locomotion.g1.walk_reward import (
    command_direction_speed_deficit,
    tracking_planar_speed,
    tracking_planar_speed_blended,
    normalized_corridor_violation,
    null_foot_force_balance_l1,
    null_torque_relaxation_l2,
    phase_contact_mismatch_cost,
    phase_stance_targets,
    resolve_stand_height_target,
    stand_action_l2,
    stand_contact_balance_l1,
    stand_dof_vel_l2,
    stand_fall_l2,
    stand_height_deficit_l1,
    stand_height_margin_l2,
    stand_lin_vel_xy_l2,
    stand_still_l1,
    stand_tilt_l2,
    stand_tilt_margin_l2,
    stand_yaw_vel_l2,
)

LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


class G1WalkRewardBindings:
    def _init_reward_functions(self):
        if self._motion_quality_provider is not None:
            return
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": self._reward_under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "upright": rewards.upright,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "penalty_lateral_displacement": self._reward_lateral_displacement,
            "penalty_yaw_drift": self._reward_yaw_drift,
            "penalty_lateral_corridor_violation": self._reward_lateral_corridor_violation,
            "penalty_yaw_corridor_violation": self._reward_yaw_corridor_violation,
            "base_height": rewards.base_height,
            "track_base_height_exp_smooth": rewards.track_base_height_exp_smooth,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "stand_still": self._reward_stand_still,
            "stand_action_l2": self._reward_stand_action_l2,
            "stand_dof_vel_l2": self._reward_stand_dof_vel_l2,
            "stand_lin_vel_xy_l2": self._reward_stand_lin_vel_xy_l2,
            "stand_yaw_vel_l2": self._reward_stand_yaw_vel_l2,
            "stand_tilt_l2": self._reward_stand_tilt_l2,
            "stand_tilt_margin_l2": self._reward_stand_tilt_margin_l2,
            "stand_fall_l2": self._reward_stand_fall_l2,
            "stand_base_height_deficit_l1": self._reward_stand_base_height_deficit_l1,
            "stand_support_height_margin_l2": self._reward_stand_support_height_margin_l2,
            "stand_both_feet_contact": self._reward_stand_both_feet_contact,
            "stand_foot_contact_balance": self._reward_stand_foot_contact_balance,
            "stand_feet_slide_l2": self._reward_stand_feet_slide_l2,
            "stand_feet_x_l2": self._reward_stand_feet_x_l2,
            "stand_feet_y_width_l2": self._reward_stand_feet_y_width_l2,
            "stand_feet_yaw_l2": self._reward_stand_feet_yaw_l2,
            "stand_base_feet_center_x_l2": self._reward_stand_base_feet_center_x_l2,
            "stand_base_feet_center_y_l2": self._reward_stand_base_feet_center_y_l2,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "phase_contact": self._reward_phase_contact,
            "null_foot_force_balance": self._reward_null_foot_force_balance,
            "null_torque_relaxation": self._reward_null_torque_relaxation,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }
        unknown = self._reward_cfg.scales.keys() - self._reward_fns.keys()
        if unknown:
            raise ValueError(f"unknown G1 reward.scales entries: {sorted(unknown)}")

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._reward_cfg
        if cfg.tracking_lin_relative_blend:
            return tracking_planar_speed_blended(
                ctx.info["commands"], ctx.linvel, ctx.tracking_sigma,
                cfg.tracking_lin_min_command, cfg.tracking_lin_relative_until, cfg.tracking_lin_absolute_from,
            )
        scale = self._reward_cfg.tracking_lin_error_scale
        if scale is None:
            return rewards.tracking_lin_vel(ctx)
        return tracking_planar_speed(ctx.info["commands"], ctx.linvel, scale)

    def _reward_under_speed(self, ctx: RewardContext) -> np.ndarray:
        if self._reward_cfg.under_speed_mode == "forward":
            return rewards.under_speed(ctx)
        return command_direction_speed_deficit(
            ctx.info["commands"], ctx.linvel, self._reward_cfg.under_speed_min_command
        )

    def _episode_frame_errors(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        try:
            initial_position = np.asarray(info["episode_start_base_pos"], dtype=get_global_dtype())
            initial_yaw = np.asarray(info["episode_start_base_yaw"], dtype=get_global_dtype())
        except KeyError as exc:
            raise ValueError(
                "trajectory-precision rewards require reset-time episode frame state"
            ) from exc
        current_position = np.asarray(self.get_base_pos(), dtype=get_global_dtype())
        current_yaw = np.asarray(np_yaw_from_quat(self.get_base_quat()), dtype=get_global_dtype())
        expected_position_shape = (self._num_envs, 3)
        if initial_position.shape != expected_position_shape:
            raise ValueError(
                "episode_start_base_pos must have shape "
                f"{expected_position_shape}, got {initial_position.shape}"
            )
        if initial_yaw.shape != (self._num_envs,):
            raise ValueError(
                "episode_start_base_yaw must have shape "
                f"({self._num_envs},), got {initial_yaw.shape}"
            )
        delta = current_position[:, :2] - initial_position[:, :2]
        lateral = -np.sin(initial_yaw) * delta[:, 0] + np.cos(initial_yaw) * delta[:, 1]
        yaw_drift = np_wrap_to_pi(current_yaw - initial_yaw)
        return lateral, yaw_drift

    def _straight_line_command_mask(self, info: dict[str, Any]) -> np.ndarray:
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        return np.asarray(
            (commands[:, 0] > float(self._cfg.commands.small_xy_threshold))
            & (np.abs(commands[:, 1]) < float(self._cfg.commands.dead_zone_xy))
            & (np.abs(commands[:, 2]) < float(self._cfg.commands.dead_zone_yaw)),
            dtype=get_global_dtype(),
        )

    def _reward_lateral_displacement(self, ctx: RewardContext) -> np.ndarray:
        lateral, _ = self._episode_frame_errors(ctx.info)
        return np.asarray(
            np.square(lateral) * self._straight_line_command_mask(ctx.info),
            dtype=get_global_dtype(),
        )

    def _reward_yaw_drift(self, ctx: RewardContext) -> np.ndarray:
        _, yaw_drift = self._episode_frame_errors(ctx.info)
        return np.asarray(
            np.square(yaw_drift) * self._straight_line_command_mask(ctx.info),
            dtype=get_global_dtype(),
        )

    @staticmethod
    def _normalized_corridor_violation(
        error: np.ndarray,
        tolerance: float,
        max_violation: float | None = None,
    ) -> np.ndarray:
        return np.asarray(
            normalized_corridor_violation(error, tolerance, max_violation),
            dtype=get_global_dtype(),
        )

    def _reward_lateral_corridor_violation(self, ctx: RewardContext) -> np.ndarray:
        lateral, _ = self._episode_frame_errors(ctx.info)
        violation = self._normalized_corridor_violation(
            lateral,
            self._reward_cfg.straight_line_lateral_tolerance_m,
            getattr(self._reward_cfg, "straight_line_corridor_max_violation", None),
        )
        return np.asarray(
            violation * self._straight_line_command_mask(ctx.info),
            dtype=get_global_dtype(),
        )

    def _reward_yaw_corridor_violation(self, ctx: RewardContext) -> np.ndarray:
        _, yaw_drift = self._episode_frame_errors(ctx.info)
        violation = self._normalized_corridor_violation(
            yaw_drift,
            self._reward_cfg.straight_line_yaw_tolerance_rad,
            getattr(self._reward_cfg, "straight_line_corridor_max_violation", None),
        )
        return np.asarray(
            violation * self._straight_line_command_mask(ctx.info),
            dtype=get_global_dtype(),
        )

    def _forward_progress_failure(self, info: dict[str, Any]) -> np.ndarray:
        cfg = self._cfg.forward_progress_termination
        if not cfg.enabled:
            return np.zeros((self._num_envs,), dtype=np.bool_)
        try:
            initial_position = info["episode_start_base_pos"]
            initial_yaw = info["episode_start_base_yaw"]
            steps = info["steps"]
            commands = info["commands"]
        except KeyError as error:
            raise ValueError(
                "forward-progress termination requires episode frame, steps, and commands"
            ) from error
        failure, average_speed = compute_forward_progress_failure(
            self.get_base_pos(),
            initial_position,
            initial_yaw,
            steps,
            commands,
            ctrl_dt=float(self._cfg.ctrl_dt),
            grace_steps=int(cfg.grace_steps),
            min_command_forward_speed=float(cfg.min_command_forward_speed),
            min_average_forward_speed=float(cfg.min_average_forward_speed),
        )
        info["forward_progress_average_speed"] = average_speed
        info["forward_progress_failure"] = failure
        return failure

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def _gait_constraint_cfg(self) -> GaitConstraintConfig:
        cfg = getattr(self._reward_cfg, "gait_constraint", GaitConstraintConfig())
        if isinstance(cfg, dict):
            cfg = GaitConstraintConfig(**cfg)
            self._reward_cfg.gait_constraint = cfg
        return cast(GaitConstraintConfig, cfg)

    def _reward_mode_cfg(self) -> RewardModeConfig:
        cfg = getattr(self._reward_cfg, "mode", RewardModeConfig())
        if isinstance(cfg, dict):
            cfg = RewardModeConfig(**cfg)
            self._reward_cfg.mode = cfg
        return cfg

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        height_target: float | np.ndarray | None = info.get("height_commands")
        if height_target is None:
            height_target = info.get("commands_height", self._reward_cfg.base_height_target)
        height_target_arr = np.asarray(height_target, dtype=get_global_dtype())
        if height_target_arr.ndim == 0:
            height_target = float(height_target_arr)
        elif height_target_arr.ndim == 2 and height_target_arr.shape[1] == 1:
            height_target = height_target_arr[:, 0]
        else:
            height_target = height_target_arr
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=height_target,
            base_height=self._terrain_relative_base_height(),
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        if self._motion_quality_provider is not None:
            return self._motion_quality_provider.compute(info)
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = self._compute_mode_reward(ctx, cfg)
        reward = self._apply_gait_constraint_bridge(ctx, reward)
        self._log_current_action_authority(info)
        return reward

    def _compute_mode_reward(self, ctx: RewardContext, cfg: G1RewardConfig) -> np.ndarray:
        mode_cfg = self._reward_mode_cfg()
        if not mode_cfg.enabled:
            return rewards.run_reward_dispatch(
                scales=cfg.scales,
                fns=self._reward_fns,
                ctx=ctx,
                info=ctx.info,
                enable_log=self._enable_reward_log,
                ctrl_dt=self._cfg.ctrl_dt,
            )

        walk_mask = self._gait_enabled_mask(ctx.info)
        stand_mask = np.asarray(1.0 - walk_mask, dtype=get_global_dtype())
        stand_recovery_mask = self._stand_recovery_mask(ctx, stand_mask)
        stand_static_mask = np.asarray(stand_mask - stand_recovery_mask, dtype=get_global_dtype())
        self._reset_mode_reward_log(ctx.info)
        walk_terms = self._combine_mode_terms(mode_cfg.balance_common_terms, mode_cfg.walk_terms)
        if mode_cfg.standing_enabled:
            stand_terms = self._combine_mode_terms(
                mode_cfg.balance_common_terms, mode_cfg.stand_terms
            )
            stand_recovery_terms = self._combine_mode_terms(
                mode_cfg.balance_common_terms, mode_cfg.stand_recovery_terms
            )
            stand_reward = self._run_masked_mode_reward_dispatch(
                ctx, cfg, stand_terms, stand_static_mask, mode_cfg.stand_scale_overrides
            )
            stand_recovery_reward = self._run_masked_mode_reward_dispatch(
                ctx,
                cfg,
                stand_recovery_terms,
                stand_recovery_mask,
                mode_cfg.stand_recovery_scale_overrides,
            )
        else:
            stand_reward = np.zeros((ctx.num_envs,), dtype=get_global_dtype())
            stand_recovery_reward = np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        walk_reward = self._run_masked_mode_reward_dispatch(
            ctx, cfg, walk_terms, walk_mask, mode_cfg.walk_scale_overrides
        )
        reward = np.asarray(
            stand_reward + stand_recovery_reward + walk_reward, dtype=get_global_dtype()
        )
        self._log_reward_mode(
            ctx.info,
            stand_static_mask,
            stand_recovery_mask,
            walk_mask,
            stand_reward,
            stand_recovery_reward,
            walk_reward,
        )
        return reward

    @staticmethod
    def _combine_mode_terms(common_terms: list[str], mode_terms: list[str]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for name in [*common_terms, *mode_terms]:
            if name in seen:
                continue
            seen.add(name)
            terms.append(name)
        return terms

    def _reset_mode_reward_log(self, info: dict) -> None:
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            info["log"] = {}

    def _run_masked_mode_reward_dispatch(
        self,
        ctx: RewardContext,
        cfg: G1RewardConfig,
        terms: list[str],
        mask: np.ndarray,
        scale_overrides: dict[str, float] | None = None,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        term_set = set(terms)
        scales = {name: scale for name, scale in cfg.scales.items() if name in term_set}
        if scale_overrides:
            scales.update(
                {name: scale for name, scale in scale_overrides.items() if name in term_set}
            )
        reward = np.zeros((ctx.num_envs,), dtype=dtype)
        step_count = ctx.info.get("steps", np.zeros((ctx.num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and int(step_count[0]) % 4 == 0
        log = ctx.info.get("log", {})
        mode_mask = np.asarray(mask, dtype=dtype)

        for name, scale in scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = np.asarray(rew * scale * mode_mask, dtype=dtype)
            reward += weighted_rew
            if should_log:
                key = f"reward/{name}"
                log[key] = float(log.get(key, 0.0) + np.mean(weighted_rew))

        ctx.info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _log_reward_mode(
        self,
        info: dict,
        stand_static_mask: np.ndarray,
        stand_recovery_mask: np.ndarray,
        walk_mask: np.ndarray,
        stand_reward: np.ndarray,
        stand_recovery_reward: np.ndarray,
        walk_reward: np.ndarray,
    ) -> None:
        if not self._enable_reward_log:
            return
        log = info.get("log", {})
        log["mode/stand_reward_mask"] = float(np.mean(stand_static_mask + stand_recovery_mask))
        log["mode/stand_static_reward_mask"] = float(np.mean(stand_static_mask))
        log["mode/stand_recovery_reward_mask"] = float(np.mean(stand_recovery_mask))
        log["mode/walk_reward_mask"] = float(np.mean(walk_mask))
        log["reward/mode_stand_frac"] = float(np.mean(stand_static_mask + stand_recovery_mask))
        log["reward/mode_stand_static_frac"] = float(np.mean(stand_static_mask))
        log["reward/mode_stand_recovery_frac"] = float(np.mean(stand_recovery_mask))
        log["reward/mode_walk_frac"] = float(np.mean(walk_mask))
        log["reward/stand_total"] = float(np.mean(stand_static_mask * stand_reward))
        log["reward/stand_recovery_total"] = float(
            np.mean(stand_recovery_mask * stand_recovery_reward)
        )
        log["reward/walk_total"] = float(np.mean(walk_mask * walk_reward))
        info["log"] = log

    def _apply_gait_constraint_bridge(self, ctx: RewardContext, reward: np.ndarray) -> np.ndarray:
        cfg = self._gait_constraint_cfg()
        if not cfg.enabled:
            return reward

        components = self._compute_gait_constraint_components(ctx, cfg)
        excess = np.maximum(components["total"] - cfg.epsilon, 0.0)
        gated_cost = excess * components["gate"]
        self._log_gait_constraint(ctx.info, components, gated_cost)
        return np.asarray(
            reward - cfg.penalty_scale * gated_cost * self._cfg.ctrl_dt,
            dtype=get_global_dtype(),
        )

    def _compute_gait_constraint_components(
        self, ctx: RewardContext, cfg: GaitConstraintConfig
    ) -> dict[str, np.ndarray]:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        height = compute_gait_phase_height_violation(
            left_foot[:, 2], right_foot[:, 2], gait_phase, swing_height
        )
        contrast = compute_gait_phase_contrast_violation(
            left_foot[:, 2], right_foot[:, 2], gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        contact = compute_gait_phase_contact_violation(
            left_contact, right_contact, gait_phase, swing_height
        )
        total = np.asarray(
            cfg.height_weight * height
            + cfg.contrast_weight * contrast
            + cfg.contact_weight * contact,
            dtype=get_global_dtype(),
        )

        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        command_active = self._command_active_mask(ctx.info)
        gait_enabled = self._gait_enabled_mask(ctx.info)
        gate = (
            np.ones_like(gait_enabled, dtype=get_global_dtype())
            if cfg.apply_in_stand_mode
            else gait_enabled
        )
        if cfg.apply_when_tracking:
            gate = gate * compute_tracking_gate(
                commands,
                ctx.linvel,
                ctx.gyro,
                tracking_sigma=self._reward_cfg.tracking_sigma,
                threshold=cfg.tracking_threshold,
            )

        return {
            "height": height,
            "contrast": contrast,
            "contact": contact,
            "total": total,
            "gate": np.asarray(gate, dtype=get_global_dtype()),
            "command_active": command_active,
            "gait_enabled": gait_enabled,
        }

    def _log_gait_constraint(
        self, info: dict, components: dict[str, np.ndarray], gated_cost: np.ndarray
    ) -> None:
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if not self._enable_reward_log or int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        log["constraint/gait_height"] = float(np.mean(components["height"]))
        log["constraint/gait_contrast"] = float(np.mean(components["contrast"]))
        log["constraint/gait_contact"] = float(np.mean(components["contact"]))
        log["constraint/gait_total"] = float(np.mean(components["total"]))
        log["constraint/gait_gated_cost"] = float(np.mean(gated_cost))
        log["mode/command_active"] = float(np.mean(components["command_active"]))
        log["mode/gait_enabled"] = float(np.mean(components["gait_enabled"]))
        log["mode/gait_constraint_gate"] = float(np.mean(components["gate"]))
        info["log"] = log

    def _reward_feet_phase(self, ctx: RewardContext):
        """Reward gait phase tracking by encouraging the expected swing-foot height."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        if self._reward_cfg.feet_phase_mode == "absolute_phase_height":
            left_target, right_target = compute_feet_phase_height_targets(
                gait_phase, swing_height
            )
            left_error = np.square(left_foot[:, 2] - left_target)
            right_error = np.square(right_foot[:, 2] - right_target)
            reward = np.exp(
                -(left_error + right_error)
                / self._reward_cfg.feet_phase_tracking_sigma
            )
            gate = compute_planar_command_speed_gate(
                ctx.info["commands"],
                self._reward_cfg.min_planar_command_speed_for_gait_reward,
            )
            progress = compute_planar_command_progress_ratio(
                ctx.info["commands"],
                ctx.linvel,
            )
            return np.asarray(reward * gate * progress, dtype=get_global_dtype())
        if self._reward_cfg.feet_phase_mode in {
            "filtered_command_height",
            "command_height",
            "phase_height",
        }:
            if self._reward_cfg.feet_phase_mode == "phase_height":
                target = phase_height_targets_v3(
                    gait_phase,
                    ctx.info["commands"],
                    swing_height,
                    self._reward_cfg.feet_phase_command_speed_scale,
                    self._reward_cfg.feet_phase_turn_length,
                )
            elif self._reward_cfg.feet_phase_mode == "command_height":
                commands = ctx.info["commands"]
                demand = np.sqrt(
                    np.sum(commands[:, :2] ** 2, axis=1)
                    + (self._reward_cfg.feet_phase_turn_length * commands[:, 2]) ** 2
                )
                amplitude = swing_height * demand / self._reward_cfg.feet_phase_command_speed_scale
                target = amplitude[:, None] * ((1.0 + np.sin(gait_phase)) / 2.0) ** 2
            else:
                target = command_phase_height_targets(
                    gait_phase, ctx.info["command_phase_amplitude"], swing_height
                )
            # Canonical flat G1: foot site is 2 mm below the sole reference.
            actual = np.column_stack(
                [
                    left_foot[:, 2]
                    + 0.002 * self._backend.get_sensor_data("left_foot_upvector")[:, 2],
                    right_foot[:, 2]
                    + 0.002 * self._backend.get_sensor_data("right_foot_upvector")[:, 2],
                ]
            )
            cost = command_phase_height_cost(
                actual, target, self._reward_cfg.feet_phase_height_scale
            )
            if self._reward_cfg.feet_phase_mode == "phase_height":
                return np.exp(-0.5 * cost)
            return cost
        if self._reward_cfg.feet_phase_mode == "relative_command_height":
            left_target, right_target = original_command_height_targets(
                gait_phase,
                ctx.info["commands"],
                swing_height,
                self._reward_cfg.feet_phase_command_speed_scale,
                self._reward_cfg.feet_phase_turn_length,
            )
        else:
            left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        stance_z = np.minimum(left_foot[:, 2], right_foot[:, 2])
        left_height = left_foot[:, 2] - stance_z
        right_height = right_foot[:, 2] - stance_z
        left_error = np.square(left_height - left_target)
        right_error = np.square(right_height - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        stance_z = np.minimum(left_foot[:, 2], right_foot[:, 2])
        actual_delta = (left_foot[:, 2] - stance_z) - (right_foot[:, 2] - stance_z)
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _exact_null_command_mask(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(ctx.info["commands"], dtype=get_global_dtype())
        if commands.ndim != 2 or commands.shape[1] != 3:
            raise ValueError("commands must have shape (N, 3)")
        published = ctx.info.get("command_is_null")
        exact = np.all(commands == 0.0, axis=1)
        if published is None:
            return exact
        published_mask = np.asarray(published, dtype=np.bool_)
        if published_mask.shape != exact.shape or not np.array_equal(published_mask, exact):
            raise ValueError("published command_is_null disagrees with canonical command")
        return published_mask

    def _net_foot_force_z(self) -> tuple[np.ndarray, np.ndarray]:
        left_force, _ = split_net_contact_sensor(
            self._backend.get_sensor_data("left_foot_net_contact")
        )
        right_force, _ = split_net_contact_sensor(
            self._backend.get_sensor_data("right_foot_net_contact")
        )
        return (
            np.asarray(left_force[:, 2], dtype=get_global_dtype()),
            np.asarray(right_force[:, 2], dtype=get_global_dtype()),
        )

    def _reward_phase_contact(self, ctx: RewardContext):
        cfg = self._cfg.command_gated_phase_contact
        left_force_z, right_force_z = self._net_foot_force_z()
        cost = phase_contact_mismatch_cost(
            np.asarray(ctx.info["gait_phase"], dtype=get_global_dtype()),
            left_force_z,
            right_force_z,
            duty_factor=float(cfg.duty_factor),
            contact_force_threshold=float(cfg.contact_force_threshold),
            is_null=(
                self._exact_null_command_mask(ctx)
                if self._reward_cfg.feet_phase_mode == "fixed_contact"
                else None
            ),
        )
        return np.asarray(cost, dtype=get_global_dtype())

    def _reward_null_foot_force_balance(self, ctx: RewardContext):
        cfg = self._cfg.command_gated_phase_contact
        left_force_z, right_force_z = self._net_foot_force_z()
        return null_foot_force_balance_l1(
            left_force_z,
            right_force_z,
            self._exact_null_command_mask(ctx),
            epsilon=float(cfg.force_balance_epsilon),
        )

    def _reward_null_torque_relaxation(self, ctx: RewardContext):
        torques = ctx.info.get("torques")
        if torques is None:
            raise ValueError("v024 torque relaxation requires proven actuator torque")
        if self._fada_tau_max is None:
            raise ValueError("v024 torque relaxation requires actuator force limits")
        return null_torque_relaxation_l2(
            np.asarray(torques, dtype=get_global_dtype()),
            np.asarray(self._fada_tau_max, dtype=get_global_dtype()),
            self._exact_null_command_mask(ctx),
        )

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        if self._reward_cfg.feet_orientation_stance_only:
            stance = phase_stance_targets(
                ctx.info["gait_phase"],
                duty_factor=float(self._cfg.command_gated_phase_contact.duty_factor),
                is_null=self._exact_null_command_mask(ctx),
            )
            # Gate by requested support, not measured contact: lifting a support
            # foot must not erase its orientation cost. Null commands keep both.
            return (
                np.sum(np.square(left_foot_quat[:, 1:3]), axis=1) * stance[:, 0]
                + np.sum(np.square(right_foot_quat[:, 1:3]), axis=1) * stance[:, 1]
            )
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _stand_mode_mask(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(1.0 - self._gait_enabled_mask(ctx.info), dtype=get_global_dtype())

    def _reward_stand_still(self, ctx: RewardContext):
        return stand_still_l1(ctx.dof_pos, self.default_angles, self._stand_mode_mask(ctx))

    def _reward_stand_action_l2(self, ctx: RewardContext):
        actions = ctx.info.get("current_actions", np.zeros_like(ctx.dof_pos))
        return stand_action_l2(actions, self._stand_mode_mask(ctx))

    def _reward_stand_dof_vel_l2(self, ctx: RewardContext):
        assert ctx.dof_vel is not None
        return stand_dof_vel_l2(ctx.dof_vel, self._stand_mode_mask(ctx))

    def _reward_stand_lin_vel_xy_l2(self, ctx: RewardContext):
        return stand_lin_vel_xy_l2(ctx.linvel, self._stand_mode_mask(ctx))

    def _reward_stand_yaw_vel_l2(self, ctx: RewardContext):
        return stand_yaw_vel_l2(ctx.gyro, self._stand_mode_mask(ctx))

    def _reward_stand_tilt_l2(self, ctx: RewardContext):
        return stand_tilt_l2(ctx.gravity, self._stand_mode_mask(ctx))

    def _reward_stand_tilt_margin_l2(self, ctx: RewardContext):
        return stand_tilt_margin_l2(
            ctx.gravity,
            self._stand_mode_mask(ctx),
            soft_limit_deg=float(self._reward_cfg.stand_recovery_tilt_deg_threshold),
            hard_limit_deg=float(self._reward_cfg.max_tilt_deg),
        )

    def _reward_stand_fall_l2(self, ctx: RewardContext):
        return stand_fall_l2(
            ctx.gravity,
            ctx.base_height,
            self._stand_mode_mask(ctx),
            max_tilt_deg=float(self._reward_cfg.max_tilt_deg),
            min_base_height=float(self._reward_cfg.min_base_height),
        )

    def _foot_contact_counts(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            compute_aggregated_foot_contact_count(self._backend, LEFT_FOOT_CONTACT_SENSORS),
            compute_aggregated_foot_contact_count(self._backend, RIGHT_FOOT_CONTACT_SENSORS),
        )

    def _stand_support_relative_height(self) -> np.ndarray:
        dtype = get_global_dtype()
        base_z = np.asarray(self._backend.get_base_pos()[:, 2], dtype=dtype)
        left_foot = np.asarray(self._backend.get_sensor_data("left_foot_pos"), dtype=dtype)
        right_foot = np.asarray(self._backend.get_sensor_data("right_foot_pos"), dtype=dtype)
        left_contact = compute_aggregated_foot_contact(
            self._backend, LEFT_FOOT_CONTACT_SENSORS
        ).astype(dtype)
        right_contact = compute_aggregated_foot_contact(
            self._backend, RIGHT_FOOT_CONTACT_SENSORS
        ).astype(dtype)
        contact_count = left_contact + right_contact
        eps = float(self._reward_cfg.stand_foot_contact_balance_epsilon)
        contacted_foot_z = np.where(
            contact_count > eps,
            (left_foot[:, 2] * left_contact + right_foot[:, 2] * right_contact)
            / np.maximum(contact_count, eps),
            0.5 * (left_foot[:, 2] + right_foot[:, 2]),
        )
        return np.asarray(base_z - contacted_foot_z, dtype=dtype)

    def _reward_stand_support_height_margin_l2(self, ctx: RewardContext):
        target = self._stand_height_target(ctx)
        support_height = self._stand_support_relative_height()
        return stand_height_margin_l2(
            target,
            support_height,
            self._stand_mode_mask(ctx),
            margin=float(self._reward_cfg.stand_support_height_margin),
        )

    def _reward_stand_base_height_deficit_l1(self, ctx: RewardContext):
        target = self._stand_height_target(ctx)
        margin = float(self._reward_cfg.stand_support_height_margin)
        if ctx.base_height is None:
            base_height = self._terrain_relative_base_height()
        else:
            base_height = np.asarray(ctx.base_height, dtype=get_global_dtype())
        return stand_height_deficit_l1(
            target,
            base_height,
            self._stand_mode_mask(ctx),
            margin=margin,
        )

    def _stand_height_target(self, ctx: RewardContext) -> np.ndarray:
        """Resolve the per-environment standing target with scalar legacy fallback."""
        return resolve_stand_height_target(ctx.base_height_target, num_envs=ctx.num_envs)

    def _reward_stand_both_feet_contact(self, ctx: RewardContext):
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        missing = (
            2.0 - left_contact.astype(get_global_dtype()) - right_contact.astype(get_global_dtype())
        )
        return np.asarray(missing * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_foot_contact_balance(self, ctx: RewardContext):
        left_count, right_count = self._foot_contact_counts()
        return stand_contact_balance_l1(
            left_count,
            right_count,
            self._stand_mode_mask(ctx),
            epsilon=float(self._reward_cfg.stand_foot_contact_balance_epsilon),
        )

    def _reward_stand_feet_slide_l2(self, ctx: RewardContext):
        left_vel = self._backend.get_sensor_data("left_foot_linvel")
        right_vel = self._backend.get_sensor_data("right_foot_linvel")
        left_contact = compute_aggregated_foot_contact(
            self._backend, LEFT_FOOT_CONTACT_SENSORS
        ).astype(get_global_dtype())
        right_contact = compute_aggregated_foot_contact(
            self._backend, RIGHT_FOOT_CONTACT_SENSORS
        ).astype(get_global_dtype())
        slide = (
            np.sum(np.square(left_vel[:, :2]), axis=1) * left_contact
            + np.sum(np.square(right_vel[:, :2]), axis=1) * right_contact
        )
        return np.asarray(slide * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_feet_x_l2(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        delta = self._feet_delta_in_base_yaw_frame(left_foot, right_foot)
        x_delta = delta[:, 0]
        target = float(self._reward_cfg.stand_feet_x_target)
        return np.asarray(
            np.square(x_delta - target) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_feet_y_width_l2(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        delta = self._feet_delta_in_base_yaw_frame(left_foot, right_foot)
        width = np.abs(delta[:, 1])
        target = float(self._reward_cfg.stand_feet_y_width_target)
        return np.asarray(
            np.square(width - target) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_feet_yaw_l2(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        base_yaw = np_yaw_from_quat(self._backend.get_base_quat())
        left_yaw = np_wrap_to_pi(np_yaw_from_quat(left_foot_quat) - base_yaw)
        right_yaw = np_wrap_to_pi(np_yaw_from_quat(right_foot_quat) - base_yaw)
        relative_yaw = np_wrap_to_pi(left_yaw - right_yaw)
        return np.asarray(
            np.square(relative_yaw) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_base_feet_center_x_l2(self, ctx: RewardContext):
        delta = self._base_delta_from_feet_center_in_base_yaw_frame()
        target = float(self._reward_cfg.stand_base_feet_center_x_target)
        return np.asarray(
            np.square(delta[:, 0] - target) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_base_feet_center_y_l2(self, ctx: RewardContext):
        delta = self._base_delta_from_feet_center_in_base_yaw_frame()
        target = float(self._reward_cfg.stand_base_feet_center_y_target)
        return np.asarray(
            np.square(delta[:, 1] - target) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _feet_delta_in_base_yaw_frame(
        self, left_foot: np.ndarray, right_foot: np.ndarray
    ) -> np.ndarray:
        delta = np.asarray(left_foot[:, :2] - right_foot[:, :2], dtype=get_global_dtype())
        return self._rotate_xy_to_base_yaw_frame(delta)

    def _base_delta_from_feet_center_in_base_yaw_frame(self) -> np.ndarray:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        base_pos = self._backend.get_base_pos()
        feet_center = 0.5 * (left_foot[:, :2] + right_foot[:, :2])
        delta = np.asarray(base_pos[:, :2] - feet_center, dtype=get_global_dtype())
        return self._rotate_xy_to_base_yaw_frame(delta)

    def _rotate_xy_to_base_yaw_frame(self, delta: np.ndarray) -> np.ndarray:
        base_yaw = np_yaw_from_quat(self._backend.get_base_quat())
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        return np.stack(
            [
                cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
                -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
            ],
            axis=1,
        )

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )
