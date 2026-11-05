"""Framework bindings extracted from :mod:`joystick` by responsibility.

The concrete environment remains the sole owner of mutable state.
"""

from __future__ import annotations

import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    sample_height_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.calibration_fault import (
    G1ActionExecutionFaultConfig,
    apply_action_execution_fault,
)
from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    G1FADAPrivilegedCheckpointLayoutIdentity,
    G1FADAPrivilegedObservationConfig,
    apply_fada_pd_target_perturbation,
    build_fada_reset_info,
    build_g1_fada_checkpoint_layout_identity,
    pack_fada_runtime_observation,
)
from unilab.envs.locomotion.g1.walk_commands import (
    apply_g1_command_dead_zone,
    canonicalize_g1_commands,
    command_resample_mask,
    freeze_inactive_gait_phase,
    resolve_g1_command_gait_state,
    sample_g1_walk_commands,
)
from unilab.envs.locomotion.g1.walk_config import G1CommandGatedPhaseContactConfig
from unilab.envs.locomotion.g1.walk_control import (
    advance_gait_phase,
    estimate_clipped_pd_torque,
    select_authority_actions,
    step_command_gait_phase,
    transition_command_gait_phase,
)
from unilab.envs.locomotion.g1.walk_math import (  # noqa: F401
    build_upper_body_pose_weights,
    command_phase_amplitude,
    compute_aggregated_foot_contact,
    compute_aggregated_foot_contact_count,
    compute_command_active_mask,
    compute_external_command_mask,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_forward_command_mask,
    compute_forward_progress_failure,
    compute_forward_speed_gate,
    compute_gait_phase_contact_violation,
    compute_gait_phase_contrast_violation,
    compute_gait_phase_height_violation,
    compute_tracking_gate,
    sample_gait_phase_pairs,
    sample_reset_base_qvel,
    update_command_phase_amplitude,
)


class G1WalkControlBindings:
    def _fixed_command_clock_enabled(self) -> bool:
        return self._cfg.gait_clock_mode == "command_fixed"

    def _command_gated_phase_contact_cfg(self) -> G1CommandGatedPhaseContactConfig:
        cfg = getattr(
            self._cfg,
            "command_gated_phase_contact",
            G1CommandGatedPhaseContactConfig(),
        )
        if isinstance(cfg, dict):
            cfg = G1CommandGatedPhaseContactConfig(**cfg)
            self._cfg.command_gated_phase_contact = cfg
        return cfg

    def _command_gated_phase_contact_enabled(self) -> bool:
        return bool(self._command_gated_phase_contact_cfg().enabled)

    def _resolve_command_gait_state(self, commands: np.ndarray):
        cfg = self._command_gated_phase_contact_cfg()
        return resolve_g1_command_gait_state(
            commands,
            xy_dead_zone=cfg.command_xy_dead_zone,
            yaw_dead_zone=cfg.command_yaw_dead_zone,
            linear_intensity_span=cfg.linear_intensity_span,
            yaw_intensity_span=cfg.yaw_intensity_span,
            min_frequency=cfg.min_frequency,
            max_frequency=cfg.max_frequency,
        )

    def _canonicalize_command_gait_state(self, commands: np.ndarray):
        cfg = self._command_gated_phase_contact_cfg()
        canonical = canonicalize_g1_commands(
            commands,
            xy_dead_zone=cfg.command_xy_dead_zone,
            yaw_dead_zone=cfg.command_yaw_dead_zone,
        )
        return self._resolve_command_gait_state(canonical)

    def _command_active_mask(self, info: dict) -> np.ndarray:
        commands = info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        if self._fixed_command_clock_enabled():
            return ~np.all(commands == 0.0, axis=1)
        if self._command_gated_phase_contact_enabled():
            return np.asarray(~self._resolve_command_gait_state(commands).is_null)
        cfg = self._gait_constraint_cfg()
        return compute_command_active_mask(
            commands,
            xy_threshold=cfg.command_xy_threshold,
            yaw_threshold=cfg.command_yaw_threshold,
        )

    def _current_command_gait_mask(self, info: dict) -> np.ndarray | None:
        commands = info.get("commands")
        if commands is None:
            return None

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        if commands_arr.ndim == 1:
            commands_arr = commands_arr[None, :]
        if commands_arr.ndim != 2:
            raise ValueError(f"commands must have shape (N, C), got {commands_arr.shape}")
        if self._fixed_command_clock_enabled():
            return ~np.all(commands_arr == 0.0, axis=1)
        if self._command_gated_phase_contact_enabled():
            return np.asarray(~self._resolve_command_gait_state(commands_arr).is_null)
        cfg = self._gait_constraint_cfg()
        return compute_command_active_mask(
            commands_arr,
            xy_threshold=cfg.command_xy_threshold,
            yaw_threshold=cfg.command_yaw_threshold,
        )

    def _gait_enabled_mask(self, info: dict) -> np.ndarray:
        command_mask = self._current_command_gait_mask(info)
        if command_mask is not None:
            info["gait_enabled"] = command_mask
            return command_mask

        if "gait_enabled" not in info:
            return self._command_active_mask(info)

        mask = np.asarray(info["gait_enabled"], dtype=get_global_dtype())
        if mask.ndim == 2 and mask.shape[1] == 1:
            mask = mask[:, 0]
        commands = info.get("commands")
        expected_size = np.asarray(commands).shape[0] if commands is not None else self._num_envs
        if mask.shape != (expected_size,):
            raise ValueError(f"gait_enabled must have shape ({expected_size},), got {mask.shape}")
        return np.asarray(mask > 0.5, dtype=get_global_dtype())

    def _stand_recovery_mask(self, ctx: RewardContext, stand_mask: np.ndarray) -> np.ndarray:
        speed_xy = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        if ctx.gravity is None:
            tilt_deg = np.zeros_like(speed_xy)
        else:
            tilt_deg = np.rad2deg(np.arccos(np.clip(ctx.gravity[:, 2], -1.0, 1.0)))
        recovery = (speed_xy > float(self._reward_cfg.stand_recovery_lin_vel_xy_threshold)) | (
            tilt_deg > float(self._reward_cfg.stand_recovery_tilt_deg_threshold)
        )
        recovery_mask = np.asarray(recovery, dtype=get_global_dtype()) * stand_mask
        ctx.info["stand_recovery_active"] = recovery_mask
        return recovery_mask

    def _dynamic_mode_mask(self, info: dict) -> np.ndarray:
        gait_enabled = self._gait_enabled_mask(info)
        recovery = np.asarray(
            info.get("stand_recovery_active", np.zeros_like(gait_enabled)),
            dtype=get_global_dtype(),
        )
        if recovery.ndim == 2 and recovery.shape[1] == 1:
            recovery = recovery[:, 0]
        if recovery.shape != gait_enabled.shape:
            recovery = np.zeros_like(gait_enabled)
        return np.asarray(np.maximum(gait_enabled, recovery), dtype=get_global_dtype())

    def _stand_phase_array(self) -> np.ndarray:
        if self._command_gated_phase_contact_enabled():
            return np.asarray(
                self._command_gated_phase_contact_cfg().stand_phase,
                dtype=get_global_dtype(),
            )
        cfg = self._gait_constraint_cfg()
        stand_phase = np.asarray(cfg.stand_phase, dtype=get_global_dtype())
        if stand_phase.shape != (2,):
            raise ValueError(f"gait_constraint.stand_phase must have shape (2,), got {stand_phase}")
        return stand_phase

    def _actions_for_execution(self, actions: np.ndarray, info: dict) -> np.ndarray:
        active = self._dynamic_mode_mask(info).astype(bool)
        exec_actions = select_authority_actions(
            actions,
            active,
            enabled=bool(self._cfg.stand_action_authority),
        )
        self._log_action_authority(info, actions, exec_actions, active)
        return exec_actions

    def _log_action_authority(
        self,
        info: dict,
        raw_actions: np.ndarray,
        exec_actions: np.ndarray,
        active: np.ndarray,
    ) -> None:
        if not self._enable_reward_log:
            return
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        stand = ~active.astype(bool)
        log["reward/action_authority_stand_frac"] = float(np.mean(stand))
        log["reward/raw_action_l1"] = float(np.mean(np.sum(np.abs(raw_actions), axis=1)))
        log["reward/executed_action_l1"] = float(np.mean(np.sum(np.abs(exec_actions), axis=1)))
        if np.any(stand):
            log["reward/stand_raw_action_l1"] = float(
                np.mean(np.sum(np.abs(raw_actions[stand]), axis=1))
            )
            log["reward/stand_executed_action_l1"] = float(
                np.mean(np.sum(np.abs(exec_actions[stand]), axis=1))
            )
        else:
            log["reward/stand_raw_action_l1"] = 0.0
            log["reward/stand_executed_action_l1"] = 0.0
        info["log"] = log

    def _log_current_action_authority(self, info: dict) -> None:
        raw_actions = info.get("current_actions")
        exec_actions = info.get("executed_actions")
        if raw_actions is None or exec_actions is None:
            return
        active = self._dynamic_mode_mask(info).astype(bool)
        self._log_action_authority(info, raw_actions, exec_actions, active)

    def _advance_command_state_for_reward(self, info: dict) -> None:
        if not self._command_gated_phase_contact_enabled():
            return
        commands = info.get("commands")
        if commands is None:
            return
        state = self._canonicalize_command_gait_state(commands)
        cfg = self._command_gated_phase_contact_cfg()
        previous_null = np.asarray(info.get("command_is_null", state.is_null), dtype=np.bool_)
        if previous_null.shape != state.is_null.shape:
            raise ValueError("command_is_null must match command rows")
        phase = np.asarray(
            info.get("gait_phase", np.zeros((state.commands.shape[0], 2))),
            dtype=get_global_dtype(),
        )
        info["commands"] = state.commands
        info["gait_phase"] = step_command_gait_phase(
            phase,
            was_null=previous_null,
            is_null=state.is_null,
            frequency=state.frequency,
            ctrl_dt=float(self._cfg.ctrl_dt),
            startup_phase=np.asarray(cfg.startup_phase, dtype=get_global_dtype()),
            stand_phase=np.asarray(cfg.stand_phase, dtype=get_global_dtype()),
        )
        info["command_is_null"] = state.is_null
        info["gait_enabled"] = np.asarray(~state.is_null, dtype=get_global_dtype())
        info["command_intensity"] = state.intensity
        info["gait_frequency"] = state.frequency

    def _update_commands(self, info: dict) -> None:
        """Compatibility entrypoint; runtime uses the explicit transaction name."""

        self._commit_command_state_for_next_observation(info)

    def _commit_command_state_for_next_observation(self, info: dict) -> None:
        """Sample only after scoring the command observed by the completed action."""
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype()).copy()
        self._sample_next_commands(info, commands_arr)
        if self._command_gated_phase_contact_enabled():
            self._publish_command_state_transition(
                info,
                commands_arr,
                was_null=self._resolve_command_gait_state(commands).is_null,
            )
        else:
            self._publish_ungated_commands(info, commands_arr)

    def _sample_next_commands(self, info: dict, commands_arr: np.ndarray) -> None:
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            completed_steps = steps.astype(np.uint64, copy=False) + 1
            resample_mask = command_resample_mask(completed_steps, interval_steps=interval_steps)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                commands_arr[resample_mask] = sample_g1_walk_commands(self, num_resample)
                if getattr(self._cfg.commands, "heading_command", False):
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, num_resample)
                    info["heading_commands"] = heading_commands
        if getattr(self._cfg.commands, "heading_command", False):
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands_arr.shape[0]:
                stiffness = float(getattr(self._cfg.commands, "heading_control_stiffness", 0.5))
                apply_heading_yaw_feedback(
                    commands_arr, base_quat, heading_commands, stiffness=stiffness
                )

    def _synchronize_external_command_for_observation(self, info: dict) -> None:
        """Commit an externally written command without advancing or resampling."""

        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        if not self._command_gated_phase_contact_enabled():
            self._publish_ungated_commands(info, commands_arr)
            return
        previous_null = np.asarray(
            info.get("command_is_null", np.all(commands_arr == 0.0, axis=1)),
            dtype=np.bool_,
        )
        if previous_null.shape != (commands_arr.shape[0],):
            raise ValueError("command_is_null must match command rows")
        self._publish_command_state_transition(info, commands_arr, was_null=previous_null)

    def _publish_command_state_transition(
        self,
        info: dict,
        commands: np.ndarray,
        *,
        was_null: np.ndarray,
    ) -> None:
        next_state = self._canonicalize_command_gait_state(commands)
        cfg = self._command_gated_phase_contact_cfg()
        phase = np.asarray(info["gait_phase"], dtype=get_global_dtype())
        info["gait_phase"] = transition_command_gait_phase(
            phase,
            was_null=was_null,
            is_null=next_state.is_null,
            startup_phase=np.asarray(cfg.startup_phase, dtype=get_global_dtype()),
            stand_phase=np.asarray(cfg.stand_phase, dtype=get_global_dtype()),
        )
        info["commands"] = next_state.commands
        info["command_is_null"] = next_state.is_null
        info["gait_enabled"] = np.asarray(~next_state.is_null, dtype=get_global_dtype())
        info["command_intensity"] = next_state.intensity
        info["gait_frequency"] = next_state.frequency

    def _publish_ungated_commands(self, info: dict, commands_arr: np.ndarray) -> None:
        commands_arr = apply_g1_command_dead_zone(commands_arr, self._cfg.commands)
        small_xy_threshold = float(getattr(self._cfg.commands, "small_xy_threshold", 0.0))
        if small_xy_threshold > 0.0:
            # Single chokepoint: training samples are already snapped (idempotent);
            # this covers external writers (keyboard teleop, deployment command nodes).
            commands_arr = np.array(commands_arr, copy=True)
            zero_small_xy_commands(commands_arr, threshold=small_xy_threshold)
        if self._fixed_command_clock_enabled():
            is_null = np.all(commands_arr == 0.0, axis=1)
            was_null = np.asarray(
                info.get("command_is_null", np.all(info["commands"] == 0.0, axis=1)),
                dtype=np.bool_,
            )
            clock = self._cfg.command_gated_phase_contact
            info["gait_phase"] = transition_command_gait_phase(
                info["gait_phase"], was_null=was_null, is_null=is_null,
                startup_phase=np.asarray(clock.startup_phase, dtype=get_global_dtype()),
                stand_phase=np.asarray(clock.stand_phase, dtype=get_global_dtype()),
            )
            info["command_is_null"] = is_null
            info["gait_enabled"] = np.asarray(~is_null, dtype=get_global_dtype())
            info["commands"] = commands_arr
            return
        info["commands"] = commands_arr
        cfg = self._gait_constraint_cfg()
        if cfg.enabled and cfg.freeze_phase_in_stand_mode:
            gait_phase = info.get(
                "gait_phase", np.zeros((commands_arr.shape[0], 2), dtype=get_global_dtype())
            )
            info["gait_phase"] = freeze_inactive_gait_phase(
                np.asarray(gait_phase, dtype=get_global_dtype()),
                self._gait_enabled_mask(info) > 0.5,
                self._stand_phase_array(),
            )

    def _ensure_heading_commands(self, info: dict, num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_heading_commands(self, num_obs)
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        if self._fixed_command_clock_enabled():
            # Also handle commands written directly by interactive playback.
            self._publish_ungated_commands(state.info, state.info["commands"])
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        if (
            getattr(getattr(self._cfg, "reward_config", None), "feet_phase_mode", "legacy")
            == "filtered_command_height"
        ):
            target = command_phase_amplitude(
                state.info["commands"],
                self._reward_cfg.feet_phase_command_speed_scale,
                self._reward_cfg.feet_phase_turn_length,
            )
            state.info["command_phase_amplitude"] = update_command_phase_amplitude(
                state.info["command_phase_amplitude"],
                target,
                self._cfg.ctrl_dt,
                self._reward_cfg.feet_phase_settling_tau,
            )

        if not self._command_gated_phase_contact_enabled():
            gait_phase = state.info.get(
                "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
            )
            gait_enabled = bool(getattr(self._cfg, "gait_phase_enabled", True))
            gait_cfg = self._gait_constraint_cfg() if gait_enabled else None
            fixed_clock = self._fixed_command_clock_enabled()
            freeze_inactive = bool(
                fixed_clock
                or (gait_cfg is not None and gait_cfg.enabled and gait_cfg.freeze_phase_in_stand_mode)
            )
            state.info["gait_phase"] = advance_gait_phase(
                np.asarray(gait_phase, dtype=get_global_dtype()),
                active=(
                    ~state.info["command_is_null"] if fixed_clock
                    else self._dynamic_mode_mask(state.info).astype(bool) if freeze_inactive
                    else np.ones((self._num_envs,), dtype=bool)
                ),
                delta=(
                    2.0 * np.pi * self._cfg.ctrl_dt / state.info["mq_gait_period"]
                    if self._motion_quality_provider is not None else self._gait_phase_delta
                ),
                enabled=gait_enabled,
                freeze_inactive=freeze_inactive,
                stand_phase=(
                    np.asarray(self._cfg.command_gated_phase_contact.stand_phase, dtype=get_global_dtype())
                    if fixed_clock else self._stand_phase_array() if freeze_inactive
                    else np.zeros((2,), dtype=get_global_dtype())
                ),
            )

        authority_actions = self._actions_for_execution(actions, state.info)
        state.info["authority_actions"] = authority_actions
        exec_actions = apply_action_execution_fault(
            authority_actions,
            getattr(self._cfg, "action_execution_fault", None),
            num_envs=self._num_envs,
        )
        previous_exec_actions = np.asarray(
            state.info.get("executed_actions", exec_actions), dtype=get_global_dtype()
        )
        if self._fada_privileged_enabled():
            delay = np.asarray(state.info["fada_control_delay"]).reshape(self._num_envs, 1)
            exec_actions = np.where(delay > 0.5, previous_exec_actions, exec_actions)
        state.info["executed_actions"] = exec_actions
        ctrl: np.ndarray = (
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )
        if self._fada_privileged_enabled():
            if self._fada_tau_max is None or self._fada_base_kp is None:
                raise ValueError("FADA PD perturbation baselines were not initialized")
            rfi_fraction = float(self._cfg.domain_rand.torque_rfi_fraction)
            torque_rfi = np.random.uniform(
                -rfi_fraction * self._fada_tau_max,
                rfi_fraction * self._fada_tau_max,
                size=(self._num_envs, self._num_action),
            )
            state.info["fada_torque_rfi"] = np.asarray(torque_rfi, dtype=get_global_dtype())
            kp = np.asarray(state.info["fada_kp_scale"]) * self._fada_base_kp[None, :]
            ctrl = apply_fada_pd_target_perturbation(
                ctrl,
                dof_position_bias=np.asarray(state.info["fada_dof_position_bias"]),
                torque_rfi=torque_rfi,
                kp=kp,
                tau_max=self._fada_tau_max,
            )
        if self._command_gated_phase_contact_enabled():
            state.info["command_gated_actuator_target"] = np.asarray(
                ctrl, dtype=get_global_dtype()
            ).copy()
            state.info["command_gated_actuator_target_valid"] = np.ones(
                (self._num_envs,), dtype=np.bool_
            )
        if self._debug_action_trace_enabled():
            state.info["_g1_action_trace_ctrl"] = ctrl
        return ctrl

    def _refresh_command_gated_torque(
        self,
        info: dict,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        if not self._command_gated_phase_contact_enabled():
            return
        target = info.get("command_gated_actuator_target")
        valid = np.asarray(
            info.get("command_gated_actuator_target_valid", np.zeros(self._num_envs)),
            dtype=np.bool_,
        )
        if target is None or valid.shape != (self._num_envs,) or not np.all(valid):
            raise ValueError("v024 actual step requires a valid final actuator target")
        if self._fada_base_kp is None or self._fada_base_kd is None or self._fada_tau_max is None:
            raise ValueError("v024 torque requires cached PD gains and actuator force limits")
        kp_scale = np.asarray(info.get("fada_kp_scale"), dtype=get_global_dtype())
        kd_scale = np.asarray(info.get("fada_kd_scale"), dtype=get_global_dtype())
        expected_shape = (self._num_envs, self._num_action)
        if kp_scale.shape != expected_shape or kd_scale.shape != expected_shape:
            raise ValueError("v024 torque requires per-row FADA Kp/Kd scales")
        kp = kp_scale * np.asarray(self._fada_base_kp)[None, :]
        kd = kd_scale * np.asarray(self._fada_base_kd)[None, :]
        info["torques"] = estimate_clipped_pd_torque(
            np.asarray(target, dtype=get_global_dtype()),
            np.asarray(dof_pos, dtype=get_global_dtype()),
            np.asarray(dof_vel, dtype=get_global_dtype()),
            kp,
            kd,
            np.asarray(self._fada_tau_max, dtype=get_global_dtype()),
        )
        valid.fill(False)
        info["command_gated_actuator_target_valid"] = valid
