"""Single-lap, fixed-command circle playback and measured geometry."""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
from unilab.visualization.circle_evaluation import evaluate_circle_shape
from unilab.visualization.demo_commands import _validate_demo_commands
from unilab.visualization.experiment_evaluation import evaluate_experiment

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
    approach_s: float = 0.0,
    startup_s: float = 0.0,
    keep_updating: bool = False,
    test_forward_m_s: float | None = None,
    approach_yaw_rate_rad_s: float | None = None,
    record_from_start: bool = False,
) -> dict[str, Any]:
    """Adapt, then measure one lap at the test speed without resetting learned weights."""
    env = session.env
    dt = float(env.cfg.ctrl_dt)
    program = CircleCommandProgram(
        forward_m_s=forward_m_s if test_forward_m_s is None else test_forward_m_s,
        yaw_rate_rad_s=yaw_rate_rad_s, ctrl_dt=dt,
    )
    program.validate_environment(env.cfg, keyboard=False)
    approach_steps = math.ceil(approach_s / dt - 1e-9)
    startup_steps = math.ceil(startup_s / dt - 1e-9)
    if not 0 <= startup_steps <= approach_steps:
        raise ValueError("circle startup must be within the adaptation phase")
    circle_steps = math.ceil(duration_s / dt - 1e-9)
    approach_command = program.commands[0]
    approach_yaw = yaw_rate_rad_s if approach_yaw_rate_rad_s is None else approach_yaw_rate_rad_s
    if approach_steps and (forward_m_s != approach_command[0] or approach_yaw != yaw_rate_rad_s):
        if not math.isfinite(forward_m_s) or forward_m_s <= 0:
            raise ValueError("circle adaptation forward speed must be finite and positive")
        approach_command = np.array([forward_m_s, 0.0, approach_yaw])
        _validate_demo_commands(approach_command[None, :], env.cfg, keyboard=False)
    total = approach_steps + circle_steps
    session.command_program = None
    if circle_steps < 1:
        raise ValueError("circle duration must cover at least one control step")
    payload = env.cfg.runtime_payload
    mass_levels = tuple(getattr(payload, "requested_mass_levels_kg", ()))
    if mass_levels:
        if not payload.enabled or approach_steps or startup_steps:
            raise ValueError("lap payload progression requires an enabled payload and no approach phase")
        total = (len(mass_levels) + 1) * circle_steps
    lap_start = 0
    lap_records: list[dict[str, Any]] = []
    stage_mass = 0.0
    pending_mass: float | None = None
    attach_step = math.ceil(payload.attach_time_s / dt) if payload.enabled and not mass_levels else None
    if payload.enabled and payload.additional_mass_events:
        raise ValueError("single-lap circle supports only one payload attachment")
    if attach_step is not None and not 0 <= attach_step < total:
        raise ValueError("payload attachment must occur within the trial")
    session.set_autoreset(False)
    session.reset()
    controller = session.controller
    online = controller if isinstance(controller, OnlineAdaptationController) else None
    if startup_steps and online is not None:
        session.bind_controller(online.playback)
    session.set_external_command(approach_command)
    positions = np.empty((total + 1, 3))
    quats = np.empty((total + 1, 4))
    payload_mass_kg = np.zeros(total + 1)
    applied_steps: list[int] = []
    positions[0], quats[0] = env.get_base_pos()[0], env.get_base_quat()[0]
    reason = "lap_timeout"
    directed_yaw = 0.0
    turn_direction = math.copysign(1.0, yaw_rate_rad_s)
    step = 0
    entry_step = -1
    recording_start_step = -1
    previous_yaw = 0.0
    capture = None
    with ExitStack() as recording:
        if record_from_start and approach_steps:
            recording_start_step = 0
            capture = recording.enter_context(capture_trial())
            capture(0)
        for step in range(1, total + 1):
            if startup_steps and step == startup_steps + 1 and online is not None:
                session.bind_controller(online)
            if step == approach_steps + 1:
                controller = session.controller
                if approach_steps and not keep_updating and isinstance(controller, OnlineAdaptationController):
                    session.bind_controller(controller.playback)
                    session.last_update_stats = None
                session.command_program = program
                session.command_control_steps.fill(0)
                session.set_external_command(program.commands[0])
                entry_step = step - 1
                previous_yaw = float(np_yaw_from_quat(quats[entry_step:step])[0])
                if capture is None and approach_steps and session.footprint_recorder is not None:
                    session.footprint_recorder.reset(env.get_foot_contact())
                if session.experiment_recorder is not None:
                    session.experiment_recorder.event(
                        "circle_started", online_updates=isinstance(session.controller, OnlineAdaptationController),
                    )
                if capture is None:
                    recording_start_step = entry_step
                    capture = recording.enter_context(capture_trial())
                    capture(0)
            session.step_once()
            positions[step], quats[step] = env.get_base_pos()[0], env.get_base_quat()[0]
            if entry_step >= 0:
                yaw = float(np_yaw_from_quat(quats[step:step + 1])[0])
                directed_yaw += turn_direction * float(np_wrap_to_pi(yaw - previous_yaw))
                previous_yaw = yaw
            if payload.enabled:
                info = env.state.info
                payload_mass_kg[step] = info["runtime_payload_added_mass_kg"][0]
                if info["runtime_payload_attached_this_step"][0]:
                    applied_steps.append(step - 1)
                if mass_levels:
                    attached = bool(info["runtime_payload_attached_this_step"][0])
                    expected_mass = stage_mass if pending_mass is None else pending_mass
                    if attached != (pending_mass is not None) or payload_mass_kg[step] != expected_mass:
                        raise RuntimeError("lap payload did not match the requested mass at the physics boundary")
                    if pending_mass is not None:
                        stage_mass, pending_mass = pending_mass, None
                if step - 1 == attach_step and (
                    applied_steps != [attach_step] or not np.isclose(payload_mass_kg[step], payload.added_mass_kg)
                ):
                    raise RuntimeError("Configured circle payload was not applied at the scheduled physics step")
            if env.state.terminated[0]:
                reason = "terminated"
            elif env.state.truncated[0]:
                reason = "truncated"
            elif directed_yaw >= math.tau:
                if not mass_levels:
                    reason = "lap_completed"
                else:
                    lap = {"payload_kg": stage_mass, "start_step": lap_start, "end_step": step,
                           "completed": True, "directed_yaw_rad": directed_yaw}
                    lap_records.append(lap)
                    if session.experiment_recorder is not None:
                        session.experiment_recorder.event("payload_lap_completed", **lap)
                    if len(lap_records) > len(mass_levels):
                        reason = "lap_completed"
                    else:
                        pending_mass = mass_levels[len(lap_records) - 1]
                        env.request_runtime_payload_mass(pending_mass)
                        lap_start, directed_yaw = step, 0.0
            elif mass_levels and step - lap_start >= circle_steps:
                reason = "lap_timeout"
                if capture is not None:
                    capture(step - recording_start_step, final=True)
                break
            if capture is not None:
                capture(step - recording_start_step, final=reason != "lap_timeout" or step == total)
            if reason != "lap_timeout":
                break
    if session.experiment_recorder is not None:
        session.experiment_recorder.event("trial_stopped", reason=reason, executed_steps=step)
    positions, quats = positions[:step + 1], quats[:step + 1]
    ladder = {}
    if mass_levels:
        highest = lap_records[-1]["payload_kg"] if lap_records else None
        if reason != "lap_completed":
            lap_records.append({"payload_kg": stage_mass, "start_step": lap_start, "end_step": step,
                                "completed": False, "directed_yaw_rad": directed_yaw})
        ladder = {"laps": lap_records, "highest_completed_payload_kg": highest,
                  "target_payload_kg": mass_levels[-1], "lap_timeout_s": duration_s}
    np.savez_compressed(
        trajectory_path, time_s=np.arange(step + 1) * dt, base_positions=positions,
        base_quats=quats, command=program.commands[0], ctrl_dt=dt, stop_reason=reason,
        payload_mass_kg=payload_mass_kg[:step + 1],
        payload_event_steps=np.asarray(applied_steps, dtype=np.int64),
        requested_duration_s=duration_s,
        approach_steps=approach_steps, approach_command=approach_command if approach_steps else np.empty(0),
        startup_steps=startup_steps,
        circle_start_step=entry_step,
        recording_start_step=recording_start_step,
        directed_yaw_rad=directed_yaw,
        **({"lap_payload_kg": np.array([lap["payload_kg"] for lap in lap_records]),
            "lap_start_steps": np.array([lap["start_step"] for lap in lap_records]),
            "lap_end_steps": np.array([lap["end_step"] for lap in lap_records]),
            "lap_completed": np.array([lap["completed"] for lap in lap_records])} if mass_levels else {}),
    )
    return {
        "stop_reason": reason, "duration_s": step * dt, "ctrl_dt": dt,
        "circle_start_step": entry_step,
        "recording_start_step": recording_start_step,
        "completed_lap": reason == "lap_completed",
        "directed_yaw_rad": directed_yaw,
        **ladder,
        "shape": (
            _measure_circle(positions[entry_step:, :2], dt, yaw_rate_rad_s)
            if reason == "lap_completed" and not mass_levels else {
                "geometry_status": "per_lap_only" if mass_levels else (
                    "not_entered" if entry_step == -1 else "incomplete_lap"
                ),
            }
        ),
    }


def finalize_circle_trial(
    result: dict[str, Any], output: Path, *, reference: Path | None = None,
) -> dict[str, Any]:
    """Evaluate finalized measurements and label their directory without rounding saved metrics."""
    entry = result["circle_start_step"]
    ladder = "laps" in result
    if ladder and reference is not None:
        raise ValueError("a single reference is not used for lap payload progression")
    result["shape_comparison"] = {
        "status": "per_lap_only" if ladder else ("reference_required" if result["completed_lap"] else "DNF"),
        "E_shape_m": None, "reference": str(reference.resolve()) if reference is not None else None,
    }
    if reference is not None:
        result["shape_comparison"] = evaluate_circle_shape(output / "trajectory.npz", reference)
    result["velocity"] = None
    result["velocity_scope"] = "per_lap_only" if ladder else "not_entered"
    result["video_path"] = None
    evaluation = None
    if entry >= 0 and not ladder:
        evaluation = evaluate_experiment({
            "run_dir": str(output / "raw"), "output_dir": str(output / "velocity"),
            "linear_frame": "body_xy", "sample_phase": "completed", "relative_to": "run_start",
            "start_s": entry * result["ctrl_dt"], "end_s": result["duration_s"],
        })
        episode, = evaluation["episodes"]
        if not episode["full_requested_window"]:
            raise RuntimeError("circle velocity recording does not cover the measured segment")
        velocity = episode["velocity_observed_segment"]
        result["velocity"] = velocity
        result["velocity_scope"] = "full_lap" if result["completed_lap"] else "incomplete_lap"
    velocity = result["velocity"] or {}
    fields = {
        "E_shape": result["shape_comparison"]["E_shape_m"],
        "E_lin": velocity.get("E_lin_m_s"), "E_ang": velocity.get("E_ang_rad_s"),
        "E_rad": result["shape"].get("radial_rmse_m"), "R": result["shape"].get("radius_m"),
    }
    suffix = "".join(
        f"_{name}_{value:#.3g}" if value is not None else f"_{name}_NA"
        for name, value in fields.items()
    )
    if ladder:
        highest = result["highest_completed_payload_kg"]
        suffix = f"_max_completed_{highest:g}kg" if highest is not None else "_no_completed_lap"
    if not result["completed_lap"]:
        suffix = f"_incomplete_{result['stop_reason']}" + suffix
    destination = output.with_name(output.name + suffix)
    if destination.exists():
        raise FileExistsError(destination)
    result["output_dir"] = str(destination)
    if result.get("recording_start_step", entry) >= 0:
        video = output / "trial.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        result["video_path"] = str(destination / video.name)
    if evaluation is not None:
        evaluation["run_dir"] = str(destination / "raw")
        evaluation["options"]["run_dir"] = str(destination / "raw")
        evaluation["options"]["output_dir"] = str(destination / "velocity")
        with (output / "velocity" / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(evaluation, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    output.rename(destination)
    return result
