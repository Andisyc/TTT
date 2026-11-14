"""Unloaded, fixed-command circle playback and measured geometry."""
from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.visualization.demo_commands import _validate_demo_commands

if TYPE_CHECKING:
    from unilab.visualization.playback_sessions import FADAPlaybackSession


class CircleCommandProgram:
    """One fixed body-frame command; nominal geometry is only for camera framing."""

    def __init__(self, *, forward_m_s: float, yaw_rate_rad_s: float, ctrl_dt: float) -> None:
        if not all(math.isfinite(x) and x > 0 for x in (forward_m_s, ctrl_dt)):
            raise ValueError("circle forward speed and control period must be finite and positive")
        if not math.isfinite(yaw_rate_rad_s) or yaw_rate_rad_s == 0:
            raise ValueError("circle yaw rate must be finite and nonzero")
        self.commands = np.array([[forward_m_s, 0.0, yaw_rate_rad_s]])
        phase = np.linspace(0.0, 2.0 * math.pi, 129)
        radius = forward_m_s / abs(yaw_rate_rad_s)
        self.reference_points = radius * np.column_stack((
            np.sin(phase), math.copysign(1.0, yaw_rate_rad_s) * (1.0 - np.cos(phase)),
        ))

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        _validate_demo_commands(self.commands, env_cfg, keyboard=keyboard)

    def segment_index_at_control_step(self, control_step: int) -> int:
        return 0

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[self.segment_index_at_control_step(control_step)]


def _measure_circle(xy: np.ndarray, dt: float, settle_s: float, yaw_rate: float) -> dict[str, Any]:
    xy = xy[math.ceil(settle_s / dt - 1e-9):].astype(np.float64)
    if len(xy) < 3:
        return {"completed_laps": 0, "geometry_status": "insufficient_samples"}
    points = xy - xy[0]
    system = np.column_stack((2 * points, np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)
    if rank != 3:
        return {"completed_laps": 0, "geometry_status": "degenerate_circle_fit"}
    relative = points - solution[:2]
    radii = np.linalg.norm(relative, axis=1)
    radius = float(radii.mean())
    angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    directed_turn = math.copysign(1.0, yaw_rate) * (angle - angle[0])
    progress = np.maximum.accumulate(directed_turn)
    seams = [xy[0]]
    for lap in (1, 2):
        target = lap * 2 * math.pi
        index = int(np.searchsorted(progress, target))
        if index == len(xy):
            break
        fraction = (target - directed_turn[index - 1]) / (directed_turn[index] - directed_turn[index - 1])
        seams.append(xy[index - 1] + fraction * (xy[index] - xy[index - 1]))
    return {
        "geometry_status": "measured",
        "completed_laps": int(progress[-1] // (2 * math.pi)),
        "net_laps": float(directed_turn[-1] / (2 * math.pi)),
        "radius_m": radius,
        "radial_rmse_m": float(np.sqrt(np.mean((radii - radius)**2))),
        "lap_seam_errors_m": np.linalg.norm(np.diff(seams, axis=0), axis=1).tolist(),
        "mean_planar_speed_m_s": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / ((len(xy) - 1) * dt)),
    }


def run_circle_trial(
    session: FADAPlaybackSession, *, forward_m_s: float, yaw_rate_rad_s: float,
    duration_s: float, settle_s: float, trajectory_path: Path,
    capture_trial: Callable[[], AbstractContextManager[Any]],
) -> dict[str, Any]:
    """Reset once, record all steps, and stop before any episode restart."""
    env = session.env
    dt = float(env.cfg.ctrl_dt)
    program = CircleCommandProgram(forward_m_s=forward_m_s, yaw_rate_rad_s=yaw_rate_rad_s, ctrl_dt=dt)
    program.validate_environment(env.cfg, keyboard=False)
    session.command_program = program
    total = math.ceil(duration_s / dt - 1e-9)
    if total < 1:
        raise ValueError("circle duration must cover at least one control step")
    session.set_autoreset(False)
    session.reset()
    session.set_external_command(program.commands[0])
    positions = np.empty((total + 1, 3))
    quats = np.empty((total + 1, 4))
    positions[0], quats[0] = env.get_base_pos()[0], env.get_base_quat()[0]
    reason = "duration_reached"
    step = 0
    with capture_trial() as capture:
        capture(0)
        for step in range(1, total + 1):
            session.step_once()
            positions[step], quats[step] = env.get_base_pos()[0], env.get_base_quat()[0]
            if env.state.terminated[0]:
                reason = "terminated"
            elif env.state.truncated[0]:
                reason = "truncated"
            capture(step, final=reason != "duration_reached" or step == total)
            if reason != "duration_reached":
                break
    if session.experiment_recorder is not None:
        session.experiment_recorder.event("trial_stopped", reason=reason, executed_steps=step)
    positions, quats = positions[:step + 1], quats[:step + 1]
    np.savez_compressed(
        trajectory_path, time_s=np.arange(step + 1) * dt, base_positions=positions,
        base_quats=quats, command=program.commands[0], ctrl_dt=dt, stop_reason=reason,
    )
    return {
        "stop_reason": reason, "duration_s": step * dt, "settle_s": settle_s,
        **_measure_circle(positions[:, :2], dt, settle_s, yaw_rate_rad_s),
    }
