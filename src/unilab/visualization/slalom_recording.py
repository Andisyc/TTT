"""Run one physical slalom trial without changing or retraining the policy."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from unilab.visualization.slalom import SlalomCommandProgram, SlalomConfig, SlalomEvaluator


def run_slalom_trial(session: Any, config: SlalomConfig, *, capture=None, log=print, reset=True) -> dict[str, Any]:
    """Measure gates and real pole poses; measurements never generate commands."""
    program = session.command_program
    if not isinstance(program, SlalomCommandProgram) or session.num_envs != 1:
        raise ValueError("Slalom requires one environment and a single-pass slalom program")
    session.set_autoreset(False)
    if reset:
        session.reset()
    dt = float(session.env.cfg.ctrl_dt)
    evaluator = SlalomEvaluator(config, program.pole_centers)
    steps = program.total_control_steps + max(1, math.ceil(config.settle_s/dt))
    base_positions, base_quats, pole_positions, pole_quats, contacts, commands = [], [], [], [], [], []

    def measure():
        base_positions.append(np.asarray(session.env.get_base_pos())[0].copy())
        base_quats.append(np.asarray(session.env.get_base_quat())[0].copy())
        pos, quat, contact = session.env.get_task_pole_state()
        pole_positions.append(np.asarray(pos)[0].copy())
        pole_quats.append(np.asarray(quat)[0].copy())
        contacts.append(np.asarray(contact)[0].copy())

    measure()
    if capture is not None:
        capture(0)
    last_gate = 0
    step = 0
    for step in range(1, steps+1):
        session.step_once()
        measure()
        commands.append(np.asarray(session.info["commands"])[0, :3].copy())
        state = session.env.state
        evaluator.update(
            base_positions[-1][:2], pole_positions[-1], pole_quats[-1], contacts[-1],
            bool(np.any(state.terminated | state.truncated)), dt,
            route_finished=step > program.total_control_steps,
        )
        report = evaluator.report()
        if report["passed_poles"] != last_gate:
            last_gate = report["passed_poles"]
            log(f"[slalom] passed pole {last_gate}/{config.pole_count}; sim_time={step*dt:.2f}s")
        if capture is not None:
            capture(step)
        if report["failed"] or report["success"]:
            break
    report = evaluator.report()
    if not report["success"] and not report["failed"]:
        report["failed"] = True
        report["failure_reasons"].append("route_exhausted_before_gates_finish_or_settling")
    report.update(
        elapsed_s=step*dt, control_steps=step, walking_duration_s=program.duration_s,
        nominal_speed_m_s=config.speed_m_s, command_source="fixed_open_loop_table",
        contact_sampling="post-physics-step samples; not a proof of zero substep impulses",
    )
    if capture is not None:
        capture(step, final=True)
    return {
        "report": report, "base_positions": np.asarray(base_positions),
        "base_quats": np.asarray(base_quats), "pole_positions": np.asarray(pole_positions),
        "pole_quats": np.asarray(pole_quats), "pole_contacts": np.asarray(contacts),
        "commands": np.asarray(commands), "reference_points": program.reference_points,
        "pole_centers": program.pole_centers, "ctrl_dt": np.array(dt),
    }


def run_slalom_demo(cfg: Any, root_dir: str | Path, *, log=print) -> dict[str, Any]:
    """Build the existing FADA session, record once, and retain a truthful task result."""
    from unilab.training import ensure_registries
    from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
    from unilab.visualization.fada_rendering import FADADemoRenderConfig
    from unilab.visualization.interactive_playback import create_fada_playback_session, select_torch_device
    from unilab.visualization.playback_cli import _build_play_args
    from unilab.visualization.playback_controls import _build_playback_config

    config = SlalomConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.demo.slalom, resolve=True)))
    if not config.enabled or str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("Use recording_profile=pole_slalom with task=g1_walk_flat/mujoco")
    if bool(cfg.online_adaptation.enabled) or bool(cfg.env.runtime_payload.enabled):
        raise ValueError("Nominal slalom requires online adaptation and payload disabled")
    if str(cfg.interactive.action_mode) != "policy" or bool(cfg.interactive.keyboard):
        raise ValueError("Slalom recording requires frozen policy playback without keyboard input")
    recording = FADAVideoRecordingConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.video_recording, resolve=True)))
    output = Path(recording.output_path).expanduser()
    if not output.is_absolute():
        output = Path(root_dir)/output
    output = output.resolve()
    result_path, trace_path = output.with_suffix(".json"), output.with_suffix(".npz")
    for path in (output, result_path, trace_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing slalom output: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_registries()
    args = _build_play_args(cfg, algo="fada")
    session, _, checkpoint = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1), cfg=cfg, root_dir=root_dir,
        device=OmegaConf.select(cfg, "training.device") or select_torch_device(), log=log,
    )
    try:
        program = session.command_program
        if not isinstance(program, SlalomCommandProgram):
            raise ValueError("Slalom factory did not return its command program")
        if (recording.duration_s is not None
                and not math.isclose(recording.duration_s, program.duration_s+config.settle_s, abs_tol=1e-9)):
            raise ValueError("A scored slalom recording uses one route plus settling; leave duration_s=null")
        render_config = FADADemoRenderConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.demo.rendering, resolve=True)))
        # Existing capture is reset-free; the trial owner resets before opening it.
        session.set_autoreset(False)
        session.reset()
        from dataclasses import replace
        capture_context = capture_fada_trial(
            session, args, replace(recording, output_path=str(output)),
            label=f"G1 {config.pole_count}-pole slalom", render_config=render_config,
        ) if recording.enabled else nullcontext(None)
        with capture_context as capture:
            data = run_slalom_trial(session, config, capture=capture, log=log, reset=False)
        report = data.pop("report")
        report.update(checkpoint=str(checkpoint), video=str(output) if recording.enabled else None,
                      trajectory=str(trace_path), result=str(result_path))
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, **data, report_json=np.array(json.dumps(report, allow_nan=False)))
        with result_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        log(f"[slalom] {'SUCCESS' if report['success'] else 'FAILED'}: {json.dumps(report, allow_nan=False)}")
        return report
    finally:
        session.wrapped_env.close()
