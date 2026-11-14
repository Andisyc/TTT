"""Single-lap, fixed-command circle playback and measured geometry."""
from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
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


def _measure_circle(xy: np.ndarray, dt: float, yaw_rate: float) -> dict[str, Any]:
    xy = xy.astype(np.float64)
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
    laps = float(directed_turn[-1] / (2 * math.pi))
    completed = max(0, math.floor(laps + 1e-9))
    return {
        "geometry_status": "measured",
        "path_completed_laps": completed,
        "net_laps": laps,
        "radius_m": radius,
        "radial_rmse_m": float(np.sqrt(np.mean((radii - radius)**2))),
        "mean_planar_speed_m_s": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / ((len(xy) - 1) * dt)),
    }


def run_circle_trial(
    session: FADAPlaybackSession, *, forward_m_s: float, yaw_rate_rad_s: float,
    duration_s: float, trajectory_path: Path,
    capture_trial: Callable[[], AbstractContextManager[Any]],
) -> dict[str, Any]:
    """Stop at one net turn in the commanded direction, or at the first failure or timeout."""
    env = session.env
    dt = float(env.cfg.ctrl_dt)
    program = CircleCommandProgram(forward_m_s=forward_m_s, yaw_rate_rad_s=yaw_rate_rad_s, ctrl_dt=dt)
    program.validate_environment(env.cfg, keyboard=False)
    session.command_program = program
    total = math.ceil(duration_s / dt - 1e-9)
    if total < 1:
        raise ValueError("circle duration must cover at least one control step")
    payload = env.cfg.runtime_payload
    if payload.enabled and (payload.attach_time_s != 0 or payload.additional_mass_events):
        raise ValueError("single-lap circle requires the full payload on the first physics step")
    session.set_autoreset(False)
    session.reset()
    session.set_external_command(program.commands[0])
    positions = np.empty((total + 1, 3))
    quats = np.empty((total + 1, 4))
    payload_mass_kg = np.zeros(total + 1)
    applied_steps: list[int] = []
    positions[0], quats[0] = env.get_base_pos()[0], env.get_base_quat()[0]
    reason = "lap_timeout"
    previous_yaw = float(np_yaw_from_quat(quats[:1])[0])
    directed_yaw = 0.0
    turn_direction = math.copysign(1.0, yaw_rate_rad_s)
    step = 0
    with capture_trial() as capture:
        capture(0)
        for step in range(1, total + 1):
            session.step_once()
            positions[step], quats[step] = env.get_base_pos()[0], env.get_base_quat()[0]
            yaw = float(np_yaw_from_quat(quats[step:step + 1])[0])
            directed_yaw += turn_direction * float(np_wrap_to_pi(yaw - previous_yaw))
            previous_yaw = yaw
            if payload.enabled:
                info = env.state.info
                payload_mass_kg[step] = info["runtime_payload_added_mass_kg"][0]
                if info["runtime_payload_attached_this_step"][0]:
                    applied_steps.append(step - 1)
                if step == 1 and (applied_steps != [0] or not np.isclose(payload_mass_kg[step], payload.added_mass_kg)):
                    raise RuntimeError("Configured circle payload was not applied on the first physics step")
            if env.state.terminated[0]:
                reason = "terminated"
            elif env.state.truncated[0]:
                reason = "truncated"
            elif directed_yaw >= math.tau:
                reason = "lap_completed"
            capture(step, final=reason != "lap_timeout" or step == total)
            if reason != "lap_timeout":
                break
    if session.experiment_recorder is not None:
        session.experiment_recorder.event("trial_stopped", reason=reason, executed_steps=step)
    positions, quats = positions[:step + 1], quats[:step + 1]
    np.savez_compressed(
        trajectory_path, time_s=np.arange(step + 1) * dt, base_positions=positions,
        base_quats=quats, command=program.commands[0], ctrl_dt=dt, stop_reason=reason,
        payload_mass_kg=payload_mass_kg[:step + 1],
        payload_event_steps=np.asarray(applied_steps, dtype=np.int64),
        requested_duration_s=duration_s,
        directed_yaw_rad=directed_yaw,
    )
    return {
        "stop_reason": reason, "duration_s": step * dt,
        "completed_lap": reason == "lap_completed",
        "directed_yaw_rad": directed_yaw,
        "shape": (
            _measure_circle(positions[:, :2], dt, yaw_rate_rad_s)
            if reason == "lap_completed" else {"geometry_status": "incomplete_lap"}
        ),
    }
