"""Straight-line adaptation followed by a fixed-duration, frozen-policy test."""
from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.envs.common.rotation import np_yaw_from_quat
from unilab.visualization.demo_commands import _validate_demo_commands

if TYPE_CHECKING:
    from unilab.visualization.playback_sessions import FADAPlaybackSession


class StraightLineCommandProgram:
    """One body-frame command; reference points only frame the recording camera."""

    def __init__(self, forward_m_s: float, distance_m: float) -> None:
        self.commands = np.array([[forward_m_s, 0.0, 0.0]])
        self.reference_points = np.array([[0.0, 0.0], [distance_m, 0.0]])

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[0]

    def segment_index_at_control_step(self, control_step: int) -> int:
        return 0


def measure_straight_line(positions: np.ndarray, initial_quat: np.ndarray) -> dict[str, float | int | None]:
    """Measure global path shape and direction; entry heading is the fixed reference."""
    yaw = float(np_yaw_from_quat(initial_quat.reshape(1, 4))[0])
    c, s = math.cos(yaw), math.sin(yaw)
    xy = (positions[:, :2] - positions[0, :2]) @ np.array([[c, -s], [s, c]])
    points = positions[:, :2].astype(np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    line_rmse = max_line_deviation = None
    direction_error = None
    sample_count = 0
    if arc[-1] > 0:
        keep = np.r_[True, np.diff(arc) > 0]
        sample_count = 1001
        distances = np.linspace(0.0, arc[-1], sample_count)
        uniform = np.column_stack([
            np.interp(distances, arc[keep], points[keep, axis]) for axis in (0, 1)
        ])
        centered = uniform - uniform.mean(axis=0)
        _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
        residual = centered @ axes[-1]
        line_rmse = float(np.sqrt(np.mean(residual ** 2)))
        max_line_deviation = float(np.max(np.abs(residual)))
        direction = axes[0]
        projection = float(direction @ (points[-1] - points[0]))
        tolerance = 64 * np.finfo(float).eps * max(float(arc[-1]), 1.0)
        if abs(projection) > tolerance and not np.isclose(singular_values[0], singular_values[1], rtol=1e-9, atol=0):
            # Resolve the fitted line's sign using actual net motion, not the desired heading.
            if projection < 0:
                direction = -direction
            cosine = float(direction @ np.array([c, s]))
            direction_error = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return {
        "line_rmse_m": line_rmse,
        "max_line_deviation_m": max_line_deviation,
        "direction_error_deg": direction_error,
        "path_length_m": float(arc[-1]),
        "line_sample_count": sample_count,
        "max_lateral_deviation_m": float(np.max(np.abs(xy[:, 1]))),
        "forward_distance_m": float(xy[-1, 0]),
        "endpoint_lateral_m": float(xy[-1, 1]),
    }


def run_straight_line_trial(
    session: FADAPlaybackSession, *, forward_m_s: float, approach_s: float,
    test_duration_s: float, trajectory_path: Path,
    capture_trial: Callable[[], AbstractContextManager[Any]],
    record_from_start: bool = False,
    keep_updating: bool = False,
) -> dict[str, Any]:
    """Reset once; freeze at test entry unless continuing online adaptation is requested."""
    if not math.isfinite(forward_m_s) or forward_m_s <= 0:
        raise ValueError("straight-line forward speed must be finite and positive")
    if not math.isfinite(approach_s) or approach_s < 0:
        raise ValueError("straight-line approach time must be finite and nonnegative")
    if not math.isfinite(test_duration_s) or test_duration_s <= 0:
        raise ValueError("straight-line test duration must be finite and positive")
    env = session.env
    dt = float(env.cfg.ctrl_dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("straight-line control period must be finite and positive")
    approach_steps = math.ceil(approach_s / dt - 1e-9)
    test_steps = max(1, math.ceil(test_duration_s / dt - 1e-9))
    total = approach_steps + test_steps
    recorded_steps = total if record_from_start else test_steps
    program = StraightLineCommandProgram(forward_m_s, forward_m_s * recorded_steps * dt)
    _validate_demo_commands(program.commands, env.cfg, keyboard=False)
    session.command_program = None
    session.set_autoreset(False)
    session.reset()
    session.set_external_command(program.commands[0])
    positions = np.empty((total + 1, 3))
    quats = np.empty((total + 1, 4))
    mass = np.zeros(total + 1)
    attach_steps = []
    positions[0], quats[0] = env.get_base_pos()[0], env.get_base_quat()[0]
    entry = -1
    reason = "test_completed"
    step = 0
    capture = None
    recording_start = -1
    recorder = session.experiment_recorder
    with ExitStack() as recording:
        if record_from_start:
            session.command_program = program
            session.command_control_steps.fill(0)
            recording_start = 0
            capture = recording.enter_context(capture_trial())
            capture(0)
        for step in range(1, total + 1):
            if step == approach_steps + 1:
                controller = session.controller
                if not keep_updating and isinstance(controller, OnlineAdaptationController):
                    session.bind_controller(controller.playback)
                    session.last_update_stats = None
                entry = step - 1
                session.command_program = program
                session.command_control_steps.fill(0)
                if not record_from_start and session.footprint_recorder is not None:
                    session.footprint_recorder.reset(env.get_foot_contact())
                if recorder is not None:
                    recorder.event(
                        "straight_line_test_started",
                        online_updates=isinstance(session.controller, OnlineAdaptationController),
                    )
                if not record_from_start:
                    recording_start = entry
                    capture = recording.enter_context(capture_trial())
                    capture(0)
            session.step_once()
            positions[step], quats[step] = env.get_base_pos()[0], env.get_base_quat()[0]
            if env.cfg.runtime_payload.enabled:
                info = env.state.info
                mass[step] = info["runtime_payload_added_mass_kg"][0]
                if info["runtime_payload_attached_this_step"][0]:
                    attach_steps.append(step - 1)
            if env.state.terminated[0]:
                reason = "terminated"
            elif env.state.truncated[0]:
                reason = "truncated"
            if capture is not None:
                capture(step - recording_start, final=reason != "test_completed" or step == total)
            if reason != "test_completed":
                break
    if recorder is not None:
        recorder.event("trial_stopped", reason=reason, executed_steps=step, test_start_step=entry)
    positions, quats = positions[:step + 1], quats[:step + 1]
    np.savez_compressed(
        trajectory_path, time_s=np.arange(step + 1) * dt, base_positions=positions,
        base_quats=quats, command=program.commands[0], ctrl_dt=dt, stop_reason=reason,
        payload_mass_kg=mass[:step + 1], payload_event_steps=np.asarray(attach_steps, dtype=np.int64),
        test_start_step=entry, approach_steps=approach_steps, requested_test_steps=test_steps,
        recording_start_step=recording_start,
    )
    complete = reason == "test_completed"
    return {
        "stop_reason": reason, "completed_test": complete, "test_start_step": entry,
        "ctrl_dt": dt, "duration_s": step * dt, "requested_test_duration_s": test_steps * dt,
        "recording_start_step": recording_start,
        "nominal_test_distance_m": forward_m_s * test_steps * dt,
        "geometry_status": "scored" if complete else ("not_entered" if entry == -1 else "incomplete_test"),
        "geometry": measure_straight_line(positions[entry:], quats[entry]) if complete else None,
        "velocity": None,
    }
