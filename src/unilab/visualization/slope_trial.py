"""One uninterrupted slope trial; terrain truth is used only for evaluation."""
from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.terrains.heightfield_terrains import HfSmoothRampTerrainCfg
from unilab.visualization.demo_commands import _validate_demo_commands
from unilab.visualization.straight_line_trial import StraightLineCommandProgram


def run_slope_trial(
    session: Any, *, ramp: HfSmoothRampTerrainCfg, forward_m_s: float,
    approach_s: float, duration_s: float, output: Path,
    capture_trial: Callable[[], AbstractContextManager[Any]],
    yaw_rate_rad_s: float = 0.0,
) -> dict[str, Any]:
    """Record from reset through success, first terminal step, or fixed timeout."""
    env = session.env
    dt = float(env.cfg.ctrl_dt)
    approach_steps = math.ceil(approach_s / dt - 1e-9)
    total = math.ceil((approach_s + duration_s) / dt - 1e-9)
    if total < 1 or approach_s < 0 or duration_s <= 0:
        raise ValueError("slope trial requires nonnegative approach and positive test duration")
    program = StraightLineCommandProgram(forward_m_s, ramp.finish_x)
    if not math.isfinite(yaw_rate_rad_s):
        raise ValueError("slope yaw command must be finite")
    program.commands[0, 2] = yaw_rate_rad_s
    _validate_demo_commands(program.commands, env.cfg, keyboard=False)
    session.command_program = program
    session.set_autoreset(False)
    session.reset()
    positions = np.empty((total + 1, 3))
    quats = np.empty((total + 1, 4))
    mass = np.zeros(total + 1)
    updates = np.zeros(total + 1, dtype=np.int64)
    payload_events = []
    positions[0], quats[0] = env.get_base_pos()[0], env.get_base_quat()[0]
    entries: dict[str, int] = {}
    boundaries = (
        ("arc_started", ramp.slope_start_x),
        ("arc_finished", ramp.slope_end_x),
    )
    recorder = session.experiment_recorder
    reason = "timeout"
    step = 0
    with capture_trial() as capture:
        capture(0)
        for step in range(1, total + 1):
            if step == approach_steps + 1 and recorder is not None:
                recorder.event("slope_test_started", online_updates=isinstance(session.controller, OnlineAdaptationController))
            session.step_once()
            positions[step], quats[step] = env.get_base_pos()[0], env.get_base_quat()[0]
            if session.last_update_stats is not None:
                updates[step] = updates[step - 1] + 1
            else:
                updates[step] = updates[step - 1]
            info = env.state.info
            mass[step] = info["runtime_payload_added_mass_kg"][0]
            if info["runtime_payload_attached_this_step"][0]:
                payload_events.append(step - 1)
            x, y = positions[step, :2]
            for name, boundary in boundaries:
                if name not in entries and x >= boundary:
                    entries[name] = step
                    if recorder is not None:
                        recorder.event(name, measurement="base_center_crossing", base_position=positions[step].tolist())
            if env.state.terminated[0]:
                reason = "terminated"
            elif env.state.truncated[0]:
                reason = "truncated"
            elif abs(y) >= ramp.size[1] / 2 - 0.25 or abs(x) >= ramp.size[0] / 2 - 0.25:
                reason = "left_course"
            elif x >= ramp.finish_x:
                reason = "course_completed"
            capture(step, final=reason != "timeout" or step == total)
            if reason != "timeout":
                break
    if recorder is not None:
        recorder.event("trial_stopped", reason=reason, executed_steps=step)
    positions, quats = positions[:step + 1], quats[:step + 1]
    np.savez_compressed(
        output / "trajectory.npz", time_s=np.arange(step + 1) * dt,
        base_positions=positions, base_quats=quats, command=program.commands[0], ctrl_dt=dt,
        payload_mass_kg=mass[:step + 1], payload_event_steps=np.asarray(payload_events, dtype=np.int64),
        completed_updates=updates[:step + 1], max_slope_deg=ramp.angle_deg,
        rise_m=ramp.rise_m, horizontal_length_m=ramp.horizontal_length_m,
        approach_steps=approach_steps, slope_start_x=ramp.slope_start_x, slope_end_x=ramp.slope_end_x,
        finish_x=ramp.finish_x, stop_reason=reason,
    )
    start = entries.get("arc_started")
    end = entries.get("arc_finished")
    complete = reason == "course_completed" and start is not None and end is not None
    lateral_rmse = None
    if complete:
        assert start is not None and end is not None
        lateral_rmse = float(np.sqrt(np.mean((positions[start:end + 1, 1] - positions[0, 1])**2)))
    return {
        "stop_reason": reason, "completed_course": complete, "duration_s": step * dt,
        "max_slope_deg": ramp.angle_deg, "rise_m": ramp.rise_m,
        "horizontal_length_m": ramp.horizontal_length_m, "completed_updates": int(updates[step]),
        "events_s": {name: index * dt for name, index in entries.items()},
        "lateral_rmse_m": lateral_rmse,
        "shape_status": "scored" if complete else "incomplete_course",
    }
