"""Offline, backend-independent piecewise-constant demo velocity commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FigureEightCommandConfig:
    width_m: float = 6.0
    height_m: float = 3.0
    speed_m_s: float = 0.3
    segment_duration_s: float = 0.8


class FigureEightCommandProgram:
    """Repeat a closed Gerono eight using constant body-frame circular arcs.

    Geometry is sampled once at construction. The first quarter is resampled
    by arc length; analytic tangents keep its turns in one direction. A small
    speed correction closes the quarter, and reflection supplies the rest.
    Traverse the first lobe with left turns and the second with right turns.
    Reference points integrate these arcs, closing both position and heading.
    Episode-local progress belongs to the playback session, not this table.
    """

    commands: np.ndarray
    reference_points: np.ndarray
    ctrl_dt: float
    segment_steps: int
    segment_duration_s: float
    duration_s: float
    total_control_steps: int

    @classmethod
    def from_config(
        cls, config: FigureEightCommandConfig, *, ctrl_dt: float,
        prediction_horizon: int = 1, window_size: int = 0, update_interval: int = 0,
    ) -> FigureEightCommandProgram:
        for name in ("width_m", "height_m", "speed_m_s", "segment_duration_s"):
            value = float(getattr(config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"demo.figure_eight.{name} must be finite and positive")
        if not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("figure-eight ctrl_dt must be finite and positive")
        segment_steps = max(1, int(ceil(config.segment_duration_s / ctrl_dt - 1e-10)))
        required_steps = max(prediction_horizon, prediction_horizon + window_size - 1,
                             prediction_horizon + update_interval)
        if segment_steps < required_steps:
            raise ValueError(
                f"demo.figure_eight segment holds {segment_steps} control steps; "
                f"requires at least {required_steps} for prediction_horizon="
                f"{prediction_horizon}, window_size={window_size}, update_interval={update_interval}"
            )
        segment_dt = segment_steps * float(ctrl_dt)
        parameter = np.linspace(0.0, np.pi / 2.0, 4097)
        a, b = config.width_m / 2.0, config.height_m
        dense = np.column_stack((a * np.sin(parameter), b * np.sin(parameter) * np.cos(parameter)))
        dense[[0, -1], 1] = 0.0
        arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
        if not np.isfinite(arc_length).all() or arc_length[-1] <= 0.0:
            raise ValueError("figure-eight geometry must have finite positive arc length")
        # At least eight arcs per quarter keep even a tiny route an eight.
        count = max(8, int(ceil(arc_length[-1] / (config.speed_m_s * segment_dt))))
        distances = np.linspace(0.0, arc_length[-1], count + 1)
        sampled_parameter = np.interp(distances, arc_length, parameter)
        points = np.column_stack((
            a * np.sin(sampled_parameter),
            b * np.sin(sampled_parameter) * np.cos(sampled_parameter),
        ))
        points[[0, -1], 1] = 0.0
        chords = np.diff(points, axis=0)
        chord_length = np.linalg.norm(chords, axis=1)
        heading = np.arctan2(
            b * np.cos(2.0 * sampled_parameter), a * np.cos(sampled_parameter),
        )
        heading[-1] = -np.pi / 2.0
        turn = np.diff(heading)
        if np.any(chord_length <= 0.0) or np.any(np.abs(turn) >= np.pi):
            raise ValueError("figure-eight arcs are under-resolved; reduce segment_duration_s")
        arc_factor = segment_dt * np.sinc(turn / (2.0 * np.pi))
        velocity = chord_length / arc_factor
        mean_heading = (heading[:-1] + heading[1:]) / 2.0
        displacement = arc_factor[:, None] * np.column_stack((
            np.cos(mean_heading), np.sin(mean_heading),
        ))
        # Preserve analytic turn directions while fitting the quarter endpoint.
        # This is a two-constraint least-norm correction, computed only at init.
        target = np.array([a, 0.0])
        velocity += displacement @ np.linalg.solve(
            displacement.T @ displacement,
            target - displacement.T @ velocity,
        )
        points = np.vstack((np.zeros(2), np.cumsum(displacement * velocity[:, None], axis=0)))
        points[-1] = target
        yaw = -turn / segment_dt
        commands = np.column_stack((
            np.concatenate((velocity, velocity[::-1], velocity, velocity[::-1])),
            np.zeros(4 * count),
            np.concatenate((yaw, yaw[::-1], -yaw, -yaw[::-1])),
        ))
        if not np.isfinite(commands).all() or np.any(commands[:, 0] <= 0.0):
            raise ValueError("figure-eight commands must be finite with positive forward speed")
        route = np.concatenate((points, points[-2::-1] * [1, -1],
                                points[1:] * [-1, 1], points[-2::-1] * [-1, -1]))
        angle = heading[0]
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        program = cls()
        program.commands = commands
        program.reference_points = route @ rotation
        program.reference_points[:, 1] *= -1.0
        program.ctrl_dt = float(ctrl_dt)
        program.segment_steps = segment_steps
        program.segment_duration_s = segment_dt
        program.duration_s = len(commands) * segment_dt
        program.total_control_steps = len(commands) * segment_steps
        program.commands.setflags(write=False)
        program.reference_points.setflags(write=False)
        return program

    def segment_index_at_control_step(self, control_step: int) -> int:
        if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
            raise ValueError("figure-eight control_step must be a non-negative integer")
        return (int(control_step) // self.segment_steps) % len(self.commands)

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[self.segment_index_at_control_step(control_step)].copy()


    def time_at_control_step(self, control_step: int) -> float:
        self.segment_index_at_control_step(control_step)
        return int(control_step) * self.ctrl_dt

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        """Reject competing command owners and canonicalization that changes the route."""
        _validate_demo_commands(self.commands, env_cfg, keyboard=keyboard)




class CalibratedFigureEightCommandProgram:
    """Replay a checkpoint-bound table without reading the robot's pose."""

    commands: np.ndarray
    hold_steps: np.ndarray
    reference_points: np.ndarray
    ctrl_dt: float
    total_control_steps: int
    duration_s: float
    metadata: dict[str, Any]

    @classmethod
    def from_file(
        cls, path: str | Path, *, identity: dict[str, Any], min_segment_steps: int,
    ) -> CalibratedFigureEightCommandProgram:
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            required = {
                "format_version", "commands", "hold_steps", "reference_points",
                "ctrl_dt", "metadata_json",
            }
            if not required.issubset(data.files):
                raise ValueError("Incomplete calibrated Gerono program")
            version = data["format_version"]
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != 1:
                raise ValueError("Unsupported calibrated Gerono program format")
            commands = np.array(data["commands"], dtype=np.float64, copy=True)
            holds = data["hold_steps"].copy()
            reference = np.array(data["reference_points"], dtype=np.float64, copy=True)
            dt = data["ctrl_dt"]
            if dt.shape != () or not np.isfinite(dt) or float(dt) <= 0:
                raise ValueError("Calibrated Gerono ctrl_dt must be finite and positive")
            text = data["metadata_json"]
            if text.shape != () or text.dtype.kind != "U":
                raise ValueError("Calibrated Gerono metadata must be a JSON string")
            metadata = json.loads(str(text))
        if not isinstance(metadata, dict) or metadata.get("identity") != identity:
            raise ValueError(
                "Calibrated Gerono checkpoint/control/geometry identity mismatch; "
                "use the checkpoint and configuration used for measurement"
            )
        if float(dt) != identity.get("ctrl_dt"):
            raise ValueError("Calibrated Gerono table timestep differs from its control identity")
        if commands.ndim != 2 or commands.shape[1] != 3 or len(commands) < 2:
            raise ValueError("Calibrated Gerono commands must have shape (N, 3), N >= 2")
        if not np.isfinite(commands).all() or np.any(commands[:, 0] <= 0):
            raise ValueError("Calibrated Gerono requires finite, forward-moving commands")
        if (holds.shape != (len(commands),) or holds.dtype.kind not in "iu"
                or np.any(holds <= 0) or min_segment_steps < 1
                or np.any(holds < min_segment_steps)):
            raise ValueError("Calibrated Gerono holds violate integer control-step/window limits")
        total = sum(int(value) for value in holds)
        if total > np.iinfo(np.int64).max:
            raise ValueError("Calibrated Gerono cycle exceeds the control-step range")
        if (reference.ndim != 2 or reference.shape != (len(commands) + 1, 2)
                or not np.isfinite(reference).all()
                or not np.allclose(reference[0], reference[-1], rtol=0, atol=1e-9)):
            raise ValueError("Calibrated Gerono reference must be a finite closed (N+1, 2) route")
        program = cls()
        program.commands = commands
        program.hold_steps = holds.astype(np.int64)
        program.reference_points = reference
        program.ctrl_dt = float(dt)
        program.total_control_steps = total
        program.duration_s = total * program.ctrl_dt
        if not np.isfinite(program.duration_s):
            raise ValueError("Calibrated Gerono duration must be finite")
        program.metadata = metadata
        program._segment_ends = np.cumsum(program.hold_steps)
        for array in (commands, program.hold_steps, reference, program._segment_ends):
            array.setflags(write=False)
        return program

    def segment_index_at_control_step(self, control_step: int) -> int:
        if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
            raise ValueError("Gerono control_step must be a non-negative integer")
        step = int(control_step) % self.total_control_steps
        return int(np.searchsorted(self._segment_ends, step, side="right"))

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[self.segment_index_at_control_step(control_step)].copy()

    def time_at_control_step(self, control_step: int) -> float:
        self.segment_index_at_control_step(control_step)
        return int(control_step) * self.ctrl_dt

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        _validate_demo_commands(self.commands, env_cfg, keyboard=keyboard)


@dataclass(frozen=True)
class DoubleCircleCommandConfig:
    forward_m_s: float
    yaw_rate_rad_s: float
    right_duration_s: float
    left_duration_s: float


class DoubleCircleCommandProgram:
    """Two fixed commands with measured hold times, not a feedback controller.

    The reference depicts two desired complete circles. Measured policy response,
    rather than ideal integration of the input commands, determines hold times.
    """

    commands: np.ndarray
    reference_points: np.ndarray
    ctrl_dt: float
    phase_steps: tuple[int, int]
    total_control_steps: int
    duration_s: float

    @classmethod
    def from_config(
        cls, config: DoubleCircleCommandConfig, *, ctrl_dt: float,
    ) -> DoubleCircleCommandProgram:
        for name in ("forward_m_s", "yaw_rate_rad_s", "right_duration_s", "left_duration_s"):
            value = float(getattr(config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"demo.double_circle.{name} must be finite and positive")
        if not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("double-circle ctrl_dt must be finite and positive")
        speed, yaw_rate = float(config.forward_m_s), float(config.yaw_rate_rad_s)
        program = cls()
        program.ctrl_dt = float(ctrl_dt)
        program.phase_steps = (
            max(1, int(ceil(config.right_duration_s / ctrl_dt - 1e-10))),
            max(1, int(ceil(config.left_duration_s / ctrl_dt - 1e-10))),
        )
        program.total_control_steps = sum(program.phase_steps)
        program.duration_s = program.total_control_steps * program.ctrl_dt
        program.commands = np.array([[speed, 0.0, -yaw_rate], [speed, 0.0, yaw_rate]])
        angle = np.linspace(0.0, 2.0 * np.pi, 129)
        radius = speed / yaw_rate
        right = radius * np.column_stack((np.sin(angle), np.cos(angle) - 1.0))
        left = radius * np.column_stack((np.sin(angle), 1.0 - np.cos(angle)))
        right[[0, -1]] = 0.0
        left[[0, -1]] = 0.0
        program.reference_points = np.vstack((right, left[1:]))
        program.commands.setflags(write=False)
        program.reference_points.setflags(write=False)
        return program

    def segment_index_at_control_step(self, control_step: int) -> int:
        if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
            raise ValueError("double-circle control_step must be a non-negative integer")
        return int(int(control_step) % self.total_control_steps >= self.phase_steps[0])

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[self.segment_index_at_control_step(control_step)].copy()

    def time_at_control_step(self, control_step: int) -> float:
        self.segment_index_at_control_step(control_step)
        return int(control_step) * self.ctrl_dt

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        _validate_demo_commands(self.commands, env_cfg, keyboard=keyboard)


def _validate_demo_commands(commands: np.ndarray, env_cfg: Any, *, keyboard: bool) -> None:
    """Keep both demo programs subject to the same command-owner contract."""
    if keyboard:
        raise ValueError("demo velocity commands require interactive.keyboard=false")
    command_cfg = env_cfg.commands
    if float(command_cfg.resampling_time) != 0.0:
        raise ValueError("demo velocity commands require env.commands.resampling_time=0.0")
    if bool(command_cfg.heading_command):
        raise ValueError("demo velocity commands require env.commands.heading_command=false")
    limits = np.asarray(command_cfg.vel_limit, dtype=np.float64)
    if (limits.shape != (2, 3) or not np.isfinite(limits).all()
            or np.any(limits[0] > limits[1])):
        raise ValueError("env.commands.vel_limit must contain ordered finite 3D bounds")
    if np.any(commands < limits[0]) or np.any(commands > limits[1]):
        raise ValueError(
            "demo velocity commands exceed env.commands.vel_limit: "
            f"required min={commands.min(axis=0)}, max={commands.max(axis=0)}"
        )
    from unilab.envs.locomotion.g1.walk_commands import (
        apply_g1_command_dead_zone,
        canonicalize_g1_commands,
    )
    from unilab.envs.locomotion.common.commands import zero_small_xy_commands

    phase_cfg = getattr(env_cfg, "command_gated_phase_contact", None)
    if phase_cfg is not None and bool(phase_cfg.enabled):
        canonical = canonicalize_g1_commands(
            commands, xy_dead_zone=phase_cfg.command_xy_dead_zone,
            yaw_dead_zone=phase_cfg.command_yaw_dead_zone,
        )
    else:
        canonical = apply_g1_command_dead_zone(commands.copy(), command_cfg)
        threshold = float(getattr(command_cfg, "small_xy_threshold", 0.0))
        if threshold > 0.0:
            canonical = canonical.copy()
            zero_small_xy_commands(canonical, threshold=threshold)
    if not np.allclose(canonical, commands, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "demo velocity commands are altered by env command dead zones / "
            "small_xy_threshold; reduce the applicable thresholds"
        )
