"""Run one timed approach and one v14 command cycle in the same playback session."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from unilab.visualization.playback_sessions import FADAPlaybackSession


def run_figure_eight_trial(
    session: FADAPlaybackSession,
    *,
    approach_steps: int,
    payload_kg: float,
    keep_updating: bool,
    capture_trial: Callable[[], AbstractContextManager[Any]],
    trajectory_path: Path,
) -> tuple[str, int]:
    """Record only the cycle; raw experiment recording can span both phases."""
    program = session.command_program
    if program is None:
        raise ValueError("A figure-eight command program is required")
    session.command_program = None
    session.set_autoreset(False)
    session.reset()

    def stop_reason() -> str | None:
        state = session.env.state
        if np.any(state.terminated):
            return "terminated"
        if np.any(state.truncated):
            return "truncated"
        return None

    approach_command = np.array([0.33, 0.0, 0.0])
    reason = None
    for step in range(approach_steps):
        session.set_external_command(approach_command)
        session.step_once()
        if step == 0:
            mass = session.env.state.info["runtime_payload_added_mass_kg"][0]
            if not np.isclose(mass, payload_kg):
                raise RuntimeError(f"Expected {payload_kg:g} kg on the first physics step, got {mass:g} kg")
        if (reason := stop_reason()) is not None:
            break

    positions, quats = [], []
    executed = 0
    entry_step = -1

    def measure() -> None:
        positions.append(session.env.get_base_pos()[0].copy())
        quats.append(session.env.get_base_quat()[0].copy())

    recorder = session.experiment_recorder
    if reason is None:
        if payload_kg > 0 and not keep_updating:
            # The inner controller shares the adapted policy and retains its live histories.
            controller = session.controller
            if controller is None:
                raise RuntimeError("Loaded figure-eight trial requires an online controller")
            session.bind_controller(controller.playback)
            session.last_update_stats = None
        session.command_program = program
        session.command_control_steps.fill(0)
        entry_step = session.step_count
        if session.footprint_recorder is not None:
            session.footprint_recorder.reset(session.env.get_foot_contact())
        if recorder is not None:
            recorder.event("figure_eight_started", online_updates=keep_updating)
        measure()
        with capture_trial() as capture:
            capture(0)
            for executed in range(1, program.total_control_steps + 1):
                session.step_once()
                measure()
                reason = stop_reason()
                capture(executed, final=reason is not None or executed == program.total_control_steps)
                if reason is not None:
                    break
        reason = reason or "command_cycle_completed"

    if recorder is not None:
        recorder.event("trial_stopped", reason=reason, figure_eight_steps=executed)
    np.savez_compressed(
        trajectory_path,
        time_s=np.arange(len(positions)) * program.ctrl_dt,
        base_positions=np.asarray(positions).reshape(-1, 3),
        base_quats=np.asarray(quats).reshape(-1, 4),
        reference_points=program.reference_points,
        commands=program.commands,
        segment_steps=program.segment_steps,
        ctrl_dt=program.ctrl_dt,
        approach_steps=approach_steps,
        figure_eight_start_step=entry_step,
        stop_reason=reason,
    )
    return reason, executed
