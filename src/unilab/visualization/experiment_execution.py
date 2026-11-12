"""One headless straight-line trial, using the existing FADA Session lifecycle.

The five groups differ only in target/reference/anchor selection. Physical and
learning settings must come from an explicit shared protocol. This module does
not implement collection, learning, rendering, fall inference, or new metrics.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.online_controller import FADAOnlineAdaptationConfig
from unilab.training.seed import apply_training_seed
from unilab.visualization.experiment_evaluation import EvaluationWindowUnavailable, evaluate_experiment
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig

GROUP_MODES = {
    "fada_ttt": "fada_ttt",
    "online_reference": "trajectory_lora",
    "action_snapshot": "action_base",
    "no_anchor": "trajectory_base",
    "full_oat": "trajectory_base",
}
_ONLINE_FIELDS = (
    "window_size", "warmup_steps", "update_interval", "gamma", "sigma", "learning_rate",
    "max_grad_norm", "lora_rank", "lora_alpha", "lora_dropout",
)


def _required(options: dict[str, Any], key: str) -> Any:
    if key not in options or options[key] is None:
        raise ValueError(f"experiment {key} must be explicitly supplied")
    return options[key]


def build_experiment_config(options: DictConfig | Mapping[str, Any], *, root_dir: str | Path) -> DictConfig:
    """Compose the task owner, then apply only the explicit trial protocol.

    An empty env_overrides explicitly accepts that task's environment settings;
    no straight_payload profile or hard-coded checkpoint/velocity is inherited.
    The returned config, including the input protocol, is saved by the recorder.
    """
    node = options if isinstance(options, DictConfig) else OmegaConf.create(dict(options))
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError("experiment configuration must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    group = _required(values, "group")
    if group not in GROUP_MODES:
        raise ValueError(f"group must be one of {tuple(GROUP_MODES)}")
    task = _required(values, "task_owner")
    if not isinstance(task, str) or not task.endswith("/mujoco"):
        raise ValueError("task_owner must explicitly select a MuJoCo task/backend owner")
    seed = _required(values, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    checkpoint = Path(_required(values, "checkpoint_path")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(_required(values, "output_dir")).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"trial output directory already exists: {output}")
    command = np.asarray(_required(values, "command"), dtype=np.float64)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("command must be a finite [vx, vy, yaw_rate] vector")
    duration = _required(values, "duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be finite and positive")
    env_overrides = _required(values, "env_overrides")
    if not isinstance(env_overrides, dict):
        raise ValueError("env_overrides must be an explicit mapping ({} accepts the task settings)")
    online = dict(_required(values, "online"))
    for key in _ONLINE_FIELDS:
        if key not in online or (online[key] is None and key != "max_grad_norm"):
            raise ValueError(f"online.{key} must be explicitly supplied")
    if set(online) != set(_ONLINE_FIELDS):
        raise ValueError("online must contain only the shared optimization fields, not mode/enabled overrides")
    # Validate the shared anchor even when this group will deliberately disable it.
    FADAOnlineAdaptationConfig(**online)
    online.update(enabled=True, mode=GROUP_MODES[group])
    if group in ("fada_ttt", "no_anchor"):
        online["sigma"] = 0.0
    FADAOnlineAdaptationConfig(**online)
    payload = dict(_required(values, "payload"))
    if not isinstance(payload.get("enabled"), bool):
        raise ValueError("payload.enabled must explicitly be true or false")
    if payload["enabled"]:
        for key in ("body_name", "added_mass_kg", "attachment_pos_m", "half_size_m", "attach_time_s"):
            if payload.get(key) is None:
                raise ValueError(f"payload.{key} must be explicitly supplied when enabled")
        if not isinstance(payload["body_name"], str) or not payload["body_name"].strip():
            raise ValueError("payload.body_name must be non-empty")
        for key in ("added_mass_kg", "attach_time_s"):
            if not np.isfinite(payload[key]) or payload[key] < 0:
                raise ValueError(f"payload.{key} must be finite and non-negative")
        for key in ("attachment_pos_m", "half_size_m"):
            vector = np.asarray(payload[key], dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all() or (key == "half_size_m" and np.any(vector <= 0)):
                raise ValueError(f"payload.{key} must be a finite 3-vector (half sizes strictly positive)")
    else:
        payload = {"enabled": False}  # Unused null fields must not overwrite task values.
    interval = values.get("progress_interval_s", 1.0)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("progress_interval_s must be positive")
    evaluation = values.get("evaluation") or {"enabled": False}
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("enabled"), bool):
        raise ValueError("evaluation.enabled must explicitly be true or false")
    if evaluation["enabled"]:
        for key in ("linear_frame", "sample_phase", "relative_to", "start_s", "end_s"):
            if evaluation.get(key) is None:
                raise ValueError(f"evaluation.{key} must be explicitly supplied")
        if evaluation["linear_frame"] not in ("body_xy", "yaw_frame_xy") or evaluation["sample_phase"] not in ("query", "completed"):
            raise ValueError("invalid evaluation frame or sample phase")
        if evaluation["relative_to"] not in ("run_start", "payload"):
            raise ValueError("invalid evaluation time origin")
        if not np.isfinite([evaluation["start_s"], evaluation["end_s"]]).all() or evaluation["start_s"] >= evaluation["end_s"]:
            raise ValueError("evaluation requires finite start_s < end_s")
    if GlobalHydra.instance().is_initialized():
        base = compose(config_name="config", overrides=[f"task={task}"])
    else:
        with initialize_config_dir(config_dir=str(Path(root_dir).resolve() / "conf" / "distill"), version_base="1.3"):
            base = compose(config_name="config", overrides=[f"task={task}"])
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    values.update(checkpoint_path=str(checkpoint), output_dir=str(output), command=command.tolist())
    cfg = OmegaConf.merge(cfg, {
        "algo": {"seed": seed},
        "training": {"play_checkpoint_path": str(checkpoint), "play_env_num": 1, "device": values.get("device")},
        "interactive": {"action_mode": "policy", "keyboard": False},
        "online_adaptation": online,
        "experiment_recording": {"enabled": True, "output_dir": str(output / "raw"),
                                 "chunk_steps": values.get("chunk_steps", 256)},
        "video_recording": {"enabled": False},
        "official_playback": {"profile_path": values.get("official_profile_path")},
        "demo": {"figure_eight": {"enabled": False}, "footprints": {"enabled": False}, "rendering": {"enabled": False}},
        "env": env_overrides,
        "experiment_execution": values,
    })
    cfg.env = OmegaConf.merge(cfg.env, {
        "commands": {"vel_limit": [command.tolist(), command.tolist()], "resampling_time": 0.0,
                     "heading_command": False, "rel_standing_envs": 0.0, "rel_transition_envs": 0.0, "rel_straight_envs": 0.0},
        "runtime_payload": payload,
    })
    assert isinstance(cfg, DictConfig)
    return cfg


def _print_progress(message: str) -> None:
    print(message, flush=True)


def run_oat_experiment(
    options: DictConfig | Mapping[str, Any], *, root_dir: str | Path,
    deps: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one selected group/seed, stop on first done, and optionally evaluate.

    deps is the existing playback factory's offline testing seam, not a second
    simulator or collection pipeline. Ctrl-C/SIGTERM/errors propagate after the
    recorder flushes, and the same Session owns policy state and physical steps.
    """
    log = log if log is not None else _print_progress
    cfg = build_experiment_config(options, root_dir=root_dir)
    settings = OmegaConf.to_container(cfg.experiment_execution, resolve=True)
    assert isinstance(settings, dict)
    output = Path(str(settings["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    apply_training_seed(int(cfg.algo.seed))
    playback_cfg = RslRlPlaybackConfig(
        task=str(settings["task_owner"]), load_run="-1", checkpoint=None,
        checkpoint_path=str(settings["checkpoint_path"]), action_mode="policy",
        policy_obs_mode="actor", algo_log_name="distill", log_root=None, num_envs=1, keyboard=False,
    )
    session, _, _ = create_fada_playback_session(
        playback_cfg=playback_cfg, cfg=cfg, root_dir=root_dir, device=cfg.training.device, deps=deps, log=log,
    )
    recorder = session.experiment_recorder
    assert recorder is not None
    result: dict[str, Any] = {"group": settings["group"], "seed": int(cfg.algo.seed),
                              "raw_dir": str(output / "raw"), "evaluation": "not_requested"}
    try:
        with experiment_recording_run(session), ExitStack() as video_stack:
            dt = float(session.env.cfg.ctrl_dt)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("headless trial requires finite positive ctrl_dt")
            timing_dt = settings.get("timing_ctrl_dt_s")
            if timing_dt is not None and not math.isclose(float(timing_dt), dt, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("timing_ctrl_dt_s differs from the checkpoint environment ctrl_dt")
            total = math.ceil(float(settings["duration_s"]) / dt - 1e-9)
            total = max(1, total)
            session.set_autoreset(False)
            session.reset()
            video = settings.get("video") or {}
            capture = None
            if video.get("enabled", False):
                from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
                from unilab.visualization.playback_cli import _build_play_args

                video_path = output / "trial.mp4"
                video_config = FADAVideoRecordingConfig(
                    output_path=str(video_path), duration_s=float(settings["duration_s"]),
                    fps=video.get("fps", 30), width=video.get("width", 1280), height=video.get("height", 720),
                )
                label = f"{video.get('label', settings['group'])} | seed={cfg.algo.seed}"
                capture = video_stack.enter_context(capture_fada_trial(
                    session, _build_play_args(cfg, algo="fada"), video_config, label=label,
                ))
                capture(0)
                result["video_path"] = str(video_path)
            command = np.asarray(settings["command"], dtype=np.float32)
            recorder.event("trial_started", group=settings["group"], seed=int(cfg.algo.seed),
                           requested_duration_s=float(settings["duration_s"]), planned_control_steps=total,
                           planned_duration_s=total * dt, autoreset=False)
            log(f"[OAT] group={settings['group']} mode={cfg.online_adaptation.mode} seed={cfg.algo.seed} "
                f"raw={output / 'raw'} duration={total * dt:g}s steps={total}")
            started = last_report = time.monotonic()
            log(f"[OAT] Progress: step=0/{total} sim_time=0/{total * dt:g}s")
            stop_reason, updates, step = "duration_reached", 0, 0
            for step in range(1, total + 1):
                session.set_external_command(command)
                session.step_once()
                state = session.env.state
                terminated = bool(np.any(state.terminated))
                truncated = bool(np.any(state.truncated))
                if capture is not None:
                    capture(step, final=step == total or terminated or truncated)
                stats = session.last_update_stats
                if stats is not None:
                    updates += 1
                    log(f"[OAT UPDATE] step={step}/{total} sim_time={step * dt:g}s {stats}")
                    session.last_update_stats = None
                now = time.monotonic()
                if now - last_report >= float(settings.get("progress_interval_s", 1.0)) or step == total or terminated or truncated:
                    log(f"[OAT] Progress: step={step}/{total} sim_time={step * dt:g}/{total * dt:g}s "
                        f"updates={updates} wall_time={now - started:.1f}s")
                    last_report = now
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break
            recorder.event("trial_stopped", reason=stop_reason, executed_steps=step, completed_updates=updates,
                           requested_duration_s=float(settings["duration_s"]), actual_duration_s=step * dt)
            result.update(stop_reason=stop_reason, executed_steps=step, planned_steps=total,
                          actual_duration_s=step * dt, completed_updates=updates)
    finally:
        session.wrapped_env.close()
    evaluation: dict[str, Any] = dict(settings.get("evaluation") or {"enabled": False})
    if evaluation.pop("enabled"):
        evaluation.update(run_dir=str(output / "raw"), output_dir=str(output / "evaluation"))
        try:
            summary = evaluate_experiment(evaluation)
        except EvaluationWindowUnavailable as exc:
            if not evaluation.get("allow_incomplete_window"):
                raise
            result.update(evaluation="unavailable", evaluation_reason=str(exc))
            log(f"[OAT] Evaluation unavailable: {exc}")
        else:
            result.update(evaluation="completed", evaluation_dir=str(output / "evaluation"),
                          episode_count=len(summary["episodes"]))
            log(f"[OAT] Evaluation saved: {output / 'evaluation'}")
    else:
        log("[OAT] Collection completed; offline evaluation was not requested.")
    log(f"[OAT] Trial stopped: {result['stop_reason']} steps={result['executed_steps']}/{result['planned_steps']}")
    return result
