"""Explicit measure/fit/verify stages; only measurement/verification create a simulator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from unilab.visualization.demo_commands import (
    CalibratedFigureEightCommandProgram, FigureEightCommandConfig, FigureEightCommandProgram,
    _validate_demo_commands,
)
from unilab.visualization.gerono_calibration import (
    ResponseModel, ResponseTrace, build_calibrated_program, calibration_identity,
    load_measurements, save_measurements, save_program,
)


class TrialTerminatedError(RuntimeError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def _pose(env):
    position = np.asarray(env.get_base_pos())[0]
    w, x, y, z = np.asarray(env.get_base_quat())[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([position[0], position[1], yaw], dtype=float)


def collect_response_trial(session, commands, *, reset=True):
    """Record actual applied commands and poses; no reset or feedback inside a trial."""
    commands = np.asarray(commands, dtype=float)
    _validate_demo_commands(commands, session.env.cfg, keyboard=False)
    if reset:
        session.reset()
    dt = float(session.env.cfg.ctrl_dt)
    poses = np.empty((len(commands) + 1, 3))
    applied = np.empty_like(commands)
    poses[0] = _pose(session.env)
    for index, command in enumerate(commands):
        session.set_external_command(command)
        applied[index] = np.asarray(session.info["commands"])[0, :3]
        if not np.allclose(applied[index], command, atol=1e-7, rtol=1e-6):
            raise RuntimeError("Calibration command was changed before policy consumption")
        session.step_once()
        poses[index + 1] = _pose(session.env)
        state = session.env.state
        if np.any(state.terminated) or np.any(state.truncated):
            trace = ResponseTrace(applied[:index+1].copy(), poses[:index+2].copy(), dt)
            raise TrialTerminatedError(f"Calibration trial terminated at {(index+1)*dt:g}s", trace)
    return ResponseTrace(applied, poses, dt)


def _path(root, value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(root) / path


def _live_config(cfg, *, duration_s, program_path=None):
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("Gerono calibration supports task=g1_walk_flat/mujoco only")
    if not bool(OmegaConf.select(cfg, "demo.figure_eight.enabled", default=False)):
        raise ValueError("Use recording_profile=figure_eight for deterministic Gerono calibration")
    for key in ("online_adaptation.enabled", "env.runtime_payload.enabled", "interactive.keyboard",
                "experiment_recording.enabled"):
        if bool(OmegaConf.select(cfg, key, default=False)):
            raise ValueError(f"Calibration requires {key}=false")
    if float(OmegaConf.select(cfg, "env.noise_config.level", default=0)) != 0:
        raise ValueError("Calibration requires env.noise_config.level=0")
    random = OmegaConf.select(cfg, "env.domain_rand", default={})
    for key, value in random.items():
        if (str(key).startswith("randomize_") or key == "push_robots") and bool(value):
            raise ValueError(f"Calibration requires env.domain_rand.{key}=false")
    if float(random.get("torque_rfi_fraction", 0)) != 0 or bool(random.get("actuator_strength", {}).get("enabled", False)):
        raise ValueError("Calibration requires nominal actuator strength and no torque noise")
    runtime = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in {
        "demo.figure_eight.enabled": program_path is not None,
        "demo.figure_eight.calibrated_program_path": str(program_path) if program_path else None,
        "demo.footprints.enabled": False, "demo.rendering.enabled": False,
        "video_recording.enabled": False, "video_recording.duration_s": float(duration_s),
        "interactive.action_mode": "policy", "interactive.keyboard": False,
        "env.max_episode_seconds": max(float(OmegaConf.select(cfg, "env.max_episode_seconds", default=0)), duration_s + 1),
    }.items():
        OmegaConf.update(runtime, key, value, force_add=True)
    return runtime


def _create_session(cfg, root_dir, log):
    from unilab.training import ensure_registries
    from unilab.visualization.interactive_playback import create_fada_playback_session, select_torch_device
    from unilab.visualization.playback_cli import _build_play_args
    from unilab.visualization.playback_controls import _build_playback_config
    ensure_registries()
    args = _build_play_args(cfg, algo="fada")
    session, _, checkpoint = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1), cfg=cfg, root_dir=root_dir,
        device=OmegaConf.select(cfg, "training.device") or select_torch_device(), log=log,
    )
    session.set_autoreset(False)
    return session, checkpoint


def _reference(identity, horizon):
    return FigureEightCommandProgram.from_config(
        FigureEightCommandConfig(**identity["geometry"]), ctrl_dt=identity["ctrl_dt"],
        prediction_horizon=horizon,
    )


def _measure(cfg, root, log):
    options = cfg.gerono_calibration
    output = _path(root, options.measurements_path)
    if output.exists():
        raise FileExistsError(output)
    forward, yaw = np.asarray(options.forward_commands, dtype=float), np.asarray(options.yaw_commands, dtype=float)
    if (forward.ndim != 1 or yaw.ndim != 1 or len(forward) < 2 or len(yaw) < 3
            or not np.isfinite(forward).all() or not np.isfinite(yaw).all() or np.any(forward <= 0)
            or not np.isfinite(options.hold_s) or options.hold_s <= 0):
        raise ValueError("Measure multiple positive forward commands and both yaw directions with positive hold_s")
    runtime = _live_config(cfg, duration_s=6 * float(options.hold_s))
    session, checkpoint = _create_session(runtime, root, log)
    traces, metadata = [], {}
    try:
        from unilab.algos.torch.distill.fada.runtime_policy import get_fada_runtime_spec
        if session.controller is None:
            raise ValueError("Calibration requires a loaded FADA policy controller")
        architecture = get_fada_runtime_spec(session.controller.policy)
        identity = calibration_identity(checkpoint, session.env.cfg, cfg.demo.figure_eight,
                                        task_name=cfg.training.task_name,
                                        observation_contract=architecture.observation_contract)
        dt = float(session.env.cfg.ctrl_dt)
        hold = max(1, int(np.ceil(float(options.hold_s) / dt - 1e-10)))
        probes = [np.tile([v, 0.0, w], (hold, 1)) for v in forward for w in yaw]
        middle_v = float(forward[len(forward) // 2])
        turn = float(np.max(np.abs(yaw)))
        probes.extend([
            np.repeat([[middle_v, 0, w] for w in (0, turn/2, turn, -turn, -turn/2, 0)], hold // 2 or 1, axis=0),
            np.repeat([[float(forward[i]), 0, w] for i, w in
                       ((0, turn/2), (-1, turn/2), (-1, -turn/2), (0, -turn/2))], hold, axis=0),
        ])
        _validate_demo_commands(np.concatenate(probes), session.env.cfg, keyboard=False)
        metadata = {"identity": identity, "complete": False,
                    "train_trial_count": len(forward) * len(yaw),
                    "min_segment_steps": int(architecture.prediction_horizon),
                    "velocity_limits": np.asarray(session.env.cfg.commands.vel_limit).tolist()}
        for index, commands in enumerate(probes):
            log(f"[gerono] trial {index+1}/{len(probes)}: {len(commands)*dt:g}s")
            traces.append(collect_response_trial(session, commands))
        metadata["complete"] = True
        save_measurements(output, traces, metadata)
        log(f"[gerono] measured response saved: {output}")
    except TrialTerminatedError as exc:
        traces.append(exc.trace)
        metadata.update(complete=False, error=str(exc))
        save_measurements(output, traces, metadata)
        raise
    finally:
        session.wrapped_env.close()


def _fit(cfg, root, log):
    options = cfg.gerono_calibration
    output = _path(root, options.program_path)
    if output.exists():
        raise FileExistsError(output)
    traces, metadata = load_measurements(_path(root, options.measurements_path))
    split = int(metadata.get("train_trial_count", 0))
    if not metadata.get("complete") or not 0 < split < len(traces):
        raise ValueError("Fit requires complete measurements and independent transition holdouts")
    model = ResponseModel.fit(traces[:split])
    rmse = model.validation_rmse(traces[split:])
    maximum = np.asarray(options.max_response_rmse, dtype=float)
    if maximum.shape != (3,) or not np.isfinite(maximum).all() or np.any(maximum <= 0):
        raise ValueError("max_response_rmse must contain positive vx/vy/yaw limits")
    if np.any(rmse > maximum):
        raise ValueError(f"Response model fails heldout multi-step prediction: RMSE={rmse.tolist()}, limits={maximum.tolist()}")
    horizon = int(metadata["min_segment_steps"])
    reference = _reference(metadata["identity"], horizon)
    result = build_calibrated_program(
        model, reference, metadata["velocity_limits"], min_segment_steps=horizon,
        time_scales=list(options.time_scales), max_nfev=int(options.max_nfev),
        position_tolerance_m=float(options.position_tolerance_m),
        heading_tolerance_rad=np.deg2rad(float(options.heading_tolerance_deg)),
    )
    report = result.pop("report")
    report.update(heldout_response_rmse=rmse.tolist(), decay=model.decay.tolist(),
                  coefficients=model.coefficients.tolist(), command_bounds=model.command_bounds.tolist())
    save_program(output, **result, metadata={"identity": metadata["identity"], "report": report})
    log(f"[gerono] predicted-only calibrated program saved: {output}; physical verification still required")
    log(json.dumps(report, allow_nan=False))


def _verify(cfg, root, log):
    options = cfg.gerono_calibration
    output = _path(root, options.verification_path)
    if output.exists():
        raise FileExistsError(output)
    cycles = int(options.cycles)
    if cycles < 3:
        raise ValueError("Gerono verification requires at least three continuous cycles")
    path = _path(root, options.program_path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    preflight = CalibratedFigureEightCommandProgram.from_file(path, identity=metadata["identity"], min_segment_steps=1)
    runtime = _live_config(cfg, duration_s=preflight.duration_s * cycles, program_path=path)
    session, _ = _create_session(runtime, root, log)
    try:
        program = session.command_program  # Factory validates against actual checkpoint/control identity.
        if not isinstance(program, CalibratedFigureEightCommandProgram):
            raise ValueError("Verification requires a loaded calibrated Gerono program")
        reference = _reference(metadata["identity"], 1)
        if len(program.commands) != len(reference.commands):
            raise ValueError("Program boundary count differs from its target Gerono geometry")
        commands = np.repeat(np.tile(program.commands, (cycles, 1)), np.tile(program.hold_steps, cycles), axis=0)
        trace = collect_response_trial(session, commands)
        poses = trace.poses.copy()
        origin = poses[0].copy()
        poses[:, :2] -= origin[:2]
        c, s = np.cos(origin[2]), np.sin(origin[2])
        poses[:, :2] = poses[:, :2] @ np.array([[c, -s], [s, c]])
        poses[:, 2] = np.unwrap(poses[:, 2]) - origin[2]
        boundaries = np.cumsum(np.tile(program.hold_steps, cycles))
        desired_yaw = np.tile(np.cumsum(reference.commands[:, 2] * reference.segment_duration_s), cycles)
        difference = poses[boundaries] - np.column_stack((np.tile(program.reference_points[1:], (cycles, 1)), desired_yaw))
        position_error = np.linalg.norm(difference[:, :2], axis=1)
        heading_error = np.abs(difference[:, 2])
        count = len(program.commands)
        passed = bool(position_error.max() <= float(options.position_tolerance_m)
                      and heading_error.max() <= np.deg2rad(float(options.heading_tolerance_deg)))
        report = {"passed": passed, "cycles": cycles,
                  "max_boundary_position_error_m": float(position_error.max()),
                  "max_boundary_heading_error_deg": float(np.degrees(heading_error.max())),
                  "crossings": [{"control_step": int(boundaries[i]), "position_error_m": float(position_error[i]),
                                  "heading_error_deg": float(np.degrees(heading_error[i]))}
                                 for i in range(count//2-1, count*cycles, count//2)]}
        save_measurements(output, [trace], {"identity": metadata["identity"], "complete": True,
                                           "verification_report": report})
        log(json.dumps(report, allow_nan=False))
        if not passed:
            raise ValueError(f"Actual Gerono tracking fails tolerances; trace retained at {output}")
    except TrialTerminatedError as exc:
        save_measurements(output, [exc.trace], {"identity": metadata["identity"], "complete": False, "error": str(exc)})
        raise
    finally:
        session.wrapped_env.close()


def run_gerono_calibration(cfg, root_dir, log=print):
    stage = str(cfg.gerono_calibration.stage)
    stages = {"measure": _measure, "fit": _fit, "verify": _verify}
    if stage not in stages:
        raise ValueError("gerono_calibration.stage must be measure, fit, or verify")
    stages[stage](cfg, Path(root_dir), log)
