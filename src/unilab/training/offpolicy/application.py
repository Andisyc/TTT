"""Application lifecycle for off-policy training and optional playback."""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parents[4]

from unilab.training import (
    apply_configured_training_seed,
    assert_offpolicy_task_choice_matches_algo,
    ensure_registries,
    get_log_root,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker

from .factory import apply_configured_actor_warm_start, build_runner
from .playback import default_device, play_offpolicy


def enable_faulthandler() -> None:
    """Enable fatal-signal Python stack dumps unless explicitly disabled."""
    if os.environ.get("UNILAB_FAULTHANDLER", "1").lower() in {"0", "false", "no", "off"}:
        return
    try:
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=True)
    except Exception as exc:
        print(f"[train_offpolicy] faulthandler unavailable: {exc}", file=sys.stderr)


def build_failure_summary(exc: BaseException, run_summary: Any | None = None) -> dict[str, Any]:
    summary = dict(run_summary) if isinstance(run_summary, dict) else {}
    if summary.get("status") == "completed":
        summary["status"] = "failed"
    else:
        summary.setdefault("status", "failed")
    summary["error_type"] = type(exc).__name__
    summary["error"] = str(exc)
    return summary


def run_offpolicy(cfg: DictConfig) -> None:
    enable_faulthandler()
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = str(
            get_log_root(ROOT_DIR, cfg) / task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = cfg.training.log_dir

    import torch

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name=algo_name,
            task_name=task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=default_device(torch, cfg.training.device),
            seed_info=seed_info,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            runner = None
            try:
                runner = build_runner(algo_name, cfg)
                apply_configured_actor_warm_start(algo_name, cfg, runner)
                if OmegaConf.select(cfg, "oracle.identity") == "motion_quality_oracle_v1":
                    from unilab.training.oracle.checkpoint import resume_oracle_training

                    resume_oracle_training(runner, cfg, log_dir)
                runner.learn(
                    max_iterations=cfg.algo.max_iterations,
                    save_interval=cfg.algo.save_interval,
                    log_dir=log_dir,
                    logger_type=cfg.training.logger,
                )
                run_summary = getattr(runner, "last_run_summary", None)
                if isinstance(run_summary, dict) and run_summary.get("status") not in (
                    None,
                    "completed",
                ):
                    raise RuntimeError(
                        f"Off-policy training ended with status={run_summary.get('status')!r}"
                    )
                if tracker is not None:
                    tracker.update_summary(run_summary)
            except BaseException as exc:
                if tracker is not None:
                    tracker.update_summary(
                        build_failure_summary(exc, getattr(runner, "last_run_summary", None))
                    )
                raise
            finally:
                if runner is not None:
                    runner.close()

        if should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            print("@" * 50)
            play_video_path = play_offpolicy(algo_name, cfg)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()
