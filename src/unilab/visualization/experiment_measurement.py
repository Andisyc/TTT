"""Read-only, single-environment inputs for OAT experiment measurement.

No environment reads, recorder, metric aggregation, or experiment defaults live
here. The producer must copy `before` at action query and `after` after physics,
before autoreset. Both belong to the SAME episode. Never reconstruct terminal
truth from a reset state or from policy `final_observation` alone.

Reference: LeCAR-Lab/FADA at 290b3a9c5c0ebe2e9851b61350617544be05e9bc,
src/holosoma_inference/holosoma_inference/utils/compute_mocap_velocity_metrics.py,
`_compute_tracking_returns`, lines 570-592 (Apache-2.0). Linear RMSE sums XY
squared errors before the temporal mean; angular RMSE uses BODY angular z.
The linear frame there is explicitly body_xy OR yaw_frame_xy. Its mocap
interpolation, synthetic clock, trim, pre-fall slicing, and finite filtering
are NOT an approved OAT measurement protocol and are not applied here.

Future recording owners must separately timestamp actual payload application,
completed updates, reset and interruption; record their values, not scheduled
settings. Terminated/truncated below are raw flags, NOT inferred fall labels.
A run must carry the resolved config, checkpoint identity, source/target action
conventions, and seed. No formal protocol or normalization denominator is set.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Sequence

import numpy as np

from unilab.envs.common.rotation import np_quat_apply_inverse, np_yaw_quat

FADA_METRIC_REVISION = "290b3a9c5c0ebe2e9851b61350617544be05e9bc"
FADA_METRIC_SOURCE = (
    f"https://github.com/LeCAR-Lab/FADA/blob/{FADA_METRIC_REVISION}/"
    "src/holosoma_inference/holosoma_inference/utils/"
    "compute_mocap_velocity_metrics.py#L570-L592"
)


def _index(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _owned_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicalMeasurement:
    """Physical state at one control boundary, not a policy observation.

    Time is simulation time in seconds from run start, including episode resets;
    control_step is episode-local. Quaternion maps body to world, wxyz order.
    Velocities are world-frame m/s and rad/s, through public env/backend APIs.
    """

    run_id: str
    env_index: int
    episode_id: int
    control_step: int
    sim_time_s: float
    base_quat_wxyz: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        for name in ("env_index", "episode_id", "control_step"):
            _index(name, getattr(self, name))
        if not np.isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("sim_time_s must be finite and non-negative")
        for name, size in (("base_quat_wxyz", 4), ("linear_velocity_world", 3), ("angular_velocity_world", 3)):
            object.__setattr__(self, name, _owned_vector(name, getattr(self, name), size))
        if not np.isclose(np.linalg.norm(self.base_quat_wxyz), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("base_quat_wxyz must be a unit quaternion; do not silently normalize it")


@dataclass(frozen=True)
class ControlStepMeasurement:
    """One command/action transition with physical truth before any autoreset.

    velocity_command_body is the 3D velocity read by the action query (m/s,
    m/s, rad/s), not the next command published after step. policy_action is
    the actual saved query action, with no additional clipping here.
    environment_action is separately supplied in its own declared joint order;
    it must not be mistaken for measured physical torque. Source/target scaling,
    defaults, fault and delay semantics belong to resolved run metadata.
    """

    before: PhysicalMeasurement
    after: PhysicalMeasurement
    ctrl_dt: float
    velocity_command_body: np.ndarray
    policy_action: np.ndarray
    environment_action: np.ndarray
    policy_joint_names: tuple[str, ...]
    environment_joint_names: tuple[str, ...]
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.ctrl_dt) or self.ctrl_dt <= 0:
            raise ValueError("ctrl_dt must be finite and positive")
        for name in ("run_id", "env_index", "episode_id"):
            if getattr(self.before, name) != getattr(self.after, name):
                raise ValueError("before and after must belong to the same run, environment and episode; reset state is not terminal truth")
        if self.after.control_step != self.before.control_step + 1:
            raise ValueError("after must be the immediately following control step")
        if not np.isclose(self.after.sim_time_s - self.before.sim_time_s, self.ctrl_dt, rtol=1e-8, atol=1e-9):
            raise ValueError("measurement timestamps must differ by ctrl_dt")
        object.__setattr__(self, "velocity_command_body", _owned_vector("velocity_command_body", self.velocity_command_body, 3))
        for action, names in (("policy_action", "policy_joint_names"), ("environment_action", "environment_joint_names")):
            joint_names = tuple(getattr(self, names))
            if not joint_names or not all(isinstance(name, str) and name for name in joint_names) or len(set(joint_names)) != len(joint_names):
                raise ValueError(f"{names} must contain unique non-empty joint names")
            object.__setattr__(self, names, joint_names)
            object.__setattr__(self, action, _owned_vector(action, getattr(self, action), len(joint_names)))
        for name in ("terminated", "truncated"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class VelocityMetricInputs:
    """Already aligned inputs, not computed metrics or approved statistics.

    Created by velocity_metric_inputs. No trim, smoothing, failed-row exclusion,
    finite filtering, normalization or cross-episode aggregation is performed.
    """

    linear_frame: Literal["body_xy", "yaw_frame_xy"]
    sample_phase: Literal["query", "completed"]
    timestamps_s: np.ndarray
    command_linear_xy: np.ndarray
    actual_linear_xy: np.ndarray
    command_angular_z: np.ndarray
    actual_angular_body_z: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


def velocity_metric_inputs(
    steps: Sequence[ControlStepMeasurement], *,
    linear_frame: Literal["body_xy", "yaw_frame_xy"],
    sample_phase: Literal["query", "completed"],
) -> VelocityMetricInputs:
    """Select explicit timing/frame conventions without choosing a protocol.

    query pairs command_t with state_t; completed pairs command_t with state_t+1
    BEFORE reset. Neither convention is silently selected from code defaults.
    Only a single contiguous episode may be passed. A terminal transition, if
    any, must be last; it remains present and flagged, never silently dropped.
    """
    if linear_frame not in ("body_xy", "yaw_frame_xy"):
        raise ValueError("linear_frame must explicitly select body_xy or yaw_frame_xy")
    if sample_phase not in ("query", "completed"):
        raise ValueError("sample_phase must explicitly select query or completed")
    if not steps:
        raise ValueError("velocity inputs require a non-empty episode window")
    for previous, current in zip(steps, steps[1:]):
        if previous.terminated or previous.truncated:
            raise ValueError("a terminal transition must end the measurement window")
        for name in ("run_id", "env_index", "episode_id", "control_step", "sim_time_s"):
            if getattr(previous.after, name) != getattr(current.before, name):
                raise ValueError("velocity inputs require a contiguous single-episode window")
        if previous.ctrl_dt != current.ctrl_dt:
            raise ValueError("velocity inputs require the same ctrl_dt")
    states = [step.before if sample_phase == "query" else step.after for step in steps]
    quats = np.stack([state.base_quat_wxyz for state in states])
    linear = np.stack([state.linear_velocity_world for state in states])
    angular = np.stack([state.angular_velocity_world for state in states])
    linear_quats = quats if linear_frame == "body_xy" else np_yaw_quat(quats)
    commands = np.stack([step.velocity_command_body for step in steps])
    values = VelocityMetricInputs(
        linear_frame=linear_frame, sample_phase=sample_phase,
        timestamps_s=np.array([state.sim_time_s for state in states]),
        command_linear_xy=commands[:, :2],
        actual_linear_xy=np_quat_apply_inverse(linear_quats, linear)[:, :2],
        command_angular_z=commands[:, 2],
        actual_angular_body_z=np_quat_apply_inverse(quats, angular)[:, 2],
        terminated=np.array([step.terminated for step in steps], dtype=bool),
        truncated=np.array([step.truncated for step in steps], dtype=bool),
    )
    for value in vars(values).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return values
