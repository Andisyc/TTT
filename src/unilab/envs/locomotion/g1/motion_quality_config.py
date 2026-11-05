"""Independent motion-quality reward configuration; historical profiles stay unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar

from unilab.envs.locomotion.g1.walk_config import (
    G1WalkRewardConfig,
    GaitConstraintConfig,
    RewardModeConfig,
)


MOTION_QUALITY_TERMS = frozenset({
    "tracking_lin_vel", "tracking_ang_vel", "alive", "penalty_lin_vel_z",
    "penalty_ang_vel_xy", "penalty_dof_vel", "penalty_dof_acc", "penalty_action_rate",
    "penalty_dof_pos_limits", "energy", "pose", "penalty_orientation", "base_height",
    "gait", "feet_slide", "feet_clearance", "penalty_feet_ori", "penalty_close_feet_xy",
    "undesired_contacts",
})


@dataclass
class MotionQualityRewardConfig(G1WalkRewardConfig):
    """Reuse G1 safety/pose settings, not its historical reward dispatch."""

    profile: ClassVar[str] = "unitree_motion_quality_v1"
    gait_period_randomization_width: float = 0.2
    gait_duty_factor: float = 0.55
    contact_force_threshold: float = 1.0
    contact_history_steps: int = 3
    min_command_threshold: float = 0.1
    feet_clearance_target: float = 0.05
    feet_clearance_std: float = 0.05
    feet_clearance_tanh_mult: float = 2.0
    soft_dof_pos_limit: float = 0.95

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scales.keys() != MOTION_QUALITY_TERMS:
            raise ValueError(
                f"motion-quality reward terms mismatch: missing={sorted(MOTION_QUALITY_TERMS - self.scales.keys())}, "
                f"unknown={sorted(self.scales.keys() - MOTION_QUALITY_TERMS)}"
            )
        for name in (
            "contact_force_threshold", "feet_clearance_target", "feet_clearance_std",
            "feet_clearance_tanh_mult", "base_height_target", "close_feet_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"reward.{name} must be finite and positive")
        if not 0 <= self.gait_period_randomization_width < 1 / self.gait_frequency:
            raise ValueError("gait period randomization must leave a positive minimum period")
        if not 0.5 < self.gait_duty_factor < 1:
            raise ValueError("gait_duty_factor must be in (0.5, 1)")
        if isinstance(self.contact_history_steps, bool) or not isinstance(self.contact_history_steps, int) or self.contact_history_steps < 1:
            raise ValueError("contact_history_steps must be a positive integer")
        if not math.isfinite(self.min_command_threshold) or self.min_command_threshold < 0:
            raise ValueError("min_command_threshold must be finite and non-negative")
        if not 0 < self.soft_dof_pos_limit <= 1:
            raise ValueError("soft_dof_pos_limit must be in (0, 1]")
        if not isinstance(self.mode, RewardModeConfig) or not isinstance(self.gait_constraint, GaitConstraintConfig):
            raise ValueError("motion-quality mode and gait configuration was not normalized")
        if self.mode.enabled or self.gait_constraint.enabled:
            raise ValueError("motion-quality rewards do not use historical mode or gait constraints")


def build_motion_quality_reward_config(recipe: dict) -> MotionQualityRewardConfig:
    """Bind the independent recipe to the shared G1 config's safety/pose interface."""
    unknown = recipe.keys() - {item.name for item in fields(MotionQualityRewardConfig)}
    if unknown:
        raise ValueError(f"unknown motion-quality reward parameters: {sorted(unknown)}")
    values = dict(recipe)
    # Inherited phase-height parameters are unused by this profile. Keep them
    # consistent with clearance metadata rather than activating a legacy reward.
    values.setdefault("feet_phase_swing_height", values["feet_clearance_target"])
    values.setdefault("feet_phase_tracking_sigma", values["feet_clearance_std"])
    return MotionQualityRewardConfig(**values)
