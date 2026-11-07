"""Offline configuration boundary for the independent motion-quality SAC Oracle.

This module does not build an environment, resolve a runtime, or load a model.
The recipe is intentionally separate from historical FADA checkpoint identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from omegaconf import DictConfig, OmegaConf

ORACLE_RUNTIME_IMPL = "motion_quality_privileged_sac_v1"
ORACLE_RUNTIME_RESOLVER = (
    "unilab.training.oracle.runtime:resolve_motion_quality_oracle_runtime"
)


@dataclass(frozen=True)
class OracleInterface:
    """Consumer contract; joint/body ordering is bound when the asset is materialized."""

    identity: str = "motion_quality_oracle_v1"
    observation_schema: str = "g1_walk_privileged_v1"
    checkpoint_format: str = "unilab_motion_quality_oracle_v1"
    privileged_schema: str = "g1_fada_privileged_v1"
    actor_obs_layout: tuple[tuple[str, int], ...] = (
        ("gyro", 3), ("gravity", 3), ("dof_pos", 29), ("dof_vel", 29),
        ("actions", 29), ("command", 3), ("gait_phase", 2),
    )
    action_dim: int = 29
    ctrl_dt: float = 0.02
    action_scale: float = 1.0
    checkpoint_required_fields: tuple[str, ...] = (
        "identity", "algorithm", "iteration", "config", "network_config",
        "obs_normalization", "priv_info_normalization", "observation_layout",
        "privileged_field_slices", "body_names", "actuated_joint_names",
        "asset_sha256", "ctrl_dt", "action_scale", "default_joint_pos", "kp", "kd",
    )

    @property
    def actor_obs_dim(self) -> int:
        return sum(width for _, width in self.actor_obs_layout)


def _value(cfg: DictConfig, path: str) -> Any:
    missing = object()
    value = OmegaConf.select(cfg, path, default=missing)
    if value is missing:
        raise ValueError(f"missing Oracle configuration: {path}")
    return value


def _require(cfg: DictConfig, path: str, expected: Any) -> None:
    value = _value(cfg, path)
    if value != expected or (isinstance(expected, bool) and not isinstance(value, bool)):
        raise ValueError(f"{path} must be {expected!r}, got {value!r}")


def _number(cfg: DictConfig, path: str, *, positive: bool = False) -> float:
    value = _value(cfg, path)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")
    return float(value)


def _range(cfg: DictConfig, path: str, lower: float, upper: float) -> None:
    value = _value(cfg, path)
    if not OmegaConf.is_list(value) or len(value) != 2:
        raise ValueError(f"{path} must contain [lower, upper]")
    lo = _number(cfg, f"{path}.0")
    hi = _number(cfg, f"{path}.1")
    if not lower <= lo <= hi <= upper:
        raise ValueError(f"{path} must be ordered and within [{lower}, {upper}]")


def validate_oracle_config(cfg: DictConfig) -> OracleInterface:
    """Validate the recipe, not training readiness or learned-policy quality.

    Returns the fixed policy interface without mutating the composed configuration.
    Physical randomization may be narrowed, but not widened beyond this profile.
    The configured resolver binds this recipe to its independent training runtime.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError("Oracle configuration must be a composed DictConfig")
    resume = OmegaConf.select(cfg, "oracle.resume_checkpoint")
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("oracle.resume_checkpoint must be a non-empty path or null")
    interface = OracleInterface()
    for path, expected in {
        "oracle.identity": interface.identity,
        "oracle.observation_schema": interface.observation_schema,
        "oracle.checkpoint_format": interface.checkpoint_format,
        "oracle.reward_profile": "unitree_motion_quality_v1",
        "algo.algo": "sac",
        "algo.runtime_impl": ORACLE_RUNTIME_IMPL,
        "algo.runtime_resolver": ORACLE_RUNTIME_RESOLVER,
        "algo.load_run": "-1",
        "algo.actor_warm_start_checkpoint": None,
        "algo.actor_warm_start_adapter": None,
        "algo.obs_normalization": True,
        "algo.actor.fixed_privileged_input": False,
        "algo.actor.priv_info_normalization": True,
        "training.task_name": "G1WalkFlat",
        "training.sim_backend": "mujoco",
        "training.num_gpus": 1,
        "training.no_sync_collection": False,
        "training.env_steps_per_sync": 1,
        "training.replay_prefetch_mode": "one_tick",
        "env.ctrl_dt": interface.ctrl_dt,
        "env.control_config.action_scale": interface.action_scale,
        "env.control_config.simulate_action_latency": False,
        "env.mode_observation": False,
        "env.gait_phase_enabled": True,
        "env.gait_phase_init_mode": "offset_phase",
        "env.gait_clock_mode": "continuous",
        "env.command_gated_phase_contact.enabled": False,
        "env.fada_privileged_observation.enabled": True,
        "env.fada_privileged_observation.schema": interface.privileged_schema,
        "env.commands.heading_command": False,
        "env.commands.dead_zone_enabled": False,
        "env.commands.small_xy_threshold": 0.0,
        "env.commands.rel_transition_envs": 0.0,
        "env.curriculum.enabled": False,
    }.items():
        _require(cfg, path, expected)

    # These overrides would change the scene, observation width, or PD owner.
    for path in ("env.scene", "env.asset", "env.sensor", "env.height_command"):
        if OmegaConf.select(cfg, path) is not None:
            raise ValueError(f"{path} cannot override the shared G1 task in this profile")

    for name in (
        "randomize_body_mass", "random_com", "randomize_gravity", "randomize_dof_armature",
        "randomize_dof_position_bias", "randomize_control_delay", "push_robots",
        "actuator_strength.enabled", "actuator_strength.curriculum_enabled",
        "actuator_strength.group_curriculum_enabled",
    ):
        _require(cfg, f"env.domain_rand.{name}", False)
    _require(cfg, "env.domain_rand.torque_rfi_fraction", 0.0)
    for name, lower, upper in (
        ("kp_multiplier_range", 0.9, 1.1),
        ("kd_multiplier_range", 0.9, 1.1),
        ("ground_friction_multiplier_range", 0.8, 1.2),
        ("added_mass_range", -1.5, 1.5),
    ):
        _range(cfg, f"env.domain_rand.{name}", lower, upper)
    for name in ("randomize_kp", "randomize_kd", "randomize_ground_friction", "randomize_base_mass"):
        _require(cfg, f"env.domain_rand.{name}", True)

    limits = _value(cfg, "env.commands.vel_limit")
    if (
        not OmegaConf.is_list(limits)
        or len(limits) != 2
        or any(not OmegaConf.is_list(row) or len(row) != 3 for row in limits)
    ):
        raise ValueError("env.commands.vel_limit must be shaped [2, 3]")
    for axis in range(3):
        lo = _number(cfg, f"env.commands.vel_limit.0.{axis}")
        hi = _number(cfg, f"env.commands.vel_limit.1.{axis}")
        if not -1.0 <= lo < hi <= 1.0:
            raise ValueError("command limits must be increasing and within [-1, 1]")
    _number(cfg, "env.commands.resampling_time", positive=True)
    if not 0 <= _number(cfg, "env.commands.rel_standing_envs") < 1:
        raise ValueError("env.commands.rel_standing_envs must be in [0, 1)")

    for path in (
        "algo.num_envs", "algo.batch_size", "algo.learning_starts", "algo.updates_per_step",
        "algo.max_iterations", "algo.save_interval", "algo.replay_buffer_n", "algo.policy_frequency",
        "algo.actor.priv_info_embed_dim",
    ):
        value = _number(cfg, path, positive=True)
        if not value.is_integer():
            raise ValueError(f"{path} must be an integer")
    widths = _value(cfg, "algo.actor.priv_mlp_hidden_dims")
    if not OmegaConf.is_list(widths) or not widths:
        raise ValueError("algo.actor.priv_mlp_hidden_dims must be a non-empty list")
    for index in range(len(widths)):
        value = _number(cfg, f"algo.actor.priv_mlp_hidden_dims.{index}", positive=True)
        if not value.is_integer():
            raise ValueError("privileged encoder widths must be integers")
    if not 0 < _number(cfg, "algo.gamma") <= 1:
        raise ValueError("algo.gamma must be in (0, 1]")
    if _number(cfg, "algo.value_support_min") >= _number(cfg, "algo.value_support_max"):
        raise ValueError("critic value support must be increasing")

    for name in (
        "gait_frequency", "gait_period_randomization_width", "tracking_sigma", "base_height_target",
        "contact_force_threshold", "feet_clearance_target", "feet_clearance_std",
        "feet_clearance_tanh_mult", "close_feet_threshold",
    ):
        _number(cfg, f"reward.{name}", positive=True)
    if OmegaConf.select(cfg, "reward.tracking_ang_sigma") is not None:
        _number(cfg, "reward.tracking_ang_sigma", positive=True)
    if not 0.5 < _number(cfg, "reward.gait_duty_factor") < 1:
        raise ValueError("reward.gait_duty_factor must be in (0.5, 1)")
    if _number(cfg, "reward.gait_period_randomization_width") >= 1 / _number(cfg, "reward.gait_frequency"):
        raise ValueError("gait period randomization must leave a positive minimum period")
    scales = _value(cfg, "reward.scales")
    if not isinstance(scales, DictConfig) or not scales:
        raise ValueError("reward.scales must be a non-empty mapping")
    for name in scales:
        _number(cfg, f"reward.scales.{name}")
    for name in ("tracking_lin_vel", "tracking_ang_vel", "gait", "feet_clearance"):
        _number(cfg, f"reward.scales.{name}", positive=True)
    for name in ("penalty_ang_vel_xy", "penalty_orientation"):
        if _number(cfg, f"reward.scales.{name}") >= 0:
            raise ValueError(f"reward.scales.{name} must penalize body motion")
    return interface
