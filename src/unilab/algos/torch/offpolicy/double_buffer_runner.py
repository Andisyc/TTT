"""Off-policy runner using the CPU-pinned double-buffer replay pipeline."""

from __future__ import annotations

import queue as queue_module
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.offpolicy.double_buffer_session import (
    DoubleBufferRunOptions,
    DoubleBufferSessionDependencies,
    DoubleBufferTrainingSession,
)
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.algos.torch.offpolicy.worker import off_policy_collector_fn
from unilab.ipc import SharedObsNormStats, SharedWeightSync
from unilab.ipc.async_runner import _SPAWN_CTX
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
    CPUPinnedDoubleBufferReplayPipeline,
)
from unilab.logging import OffPolicyLogger, TraceRecorder

CheckpointSaver = Callable[[Any, Path, int], None]
_TRACE_COMPUTE_STREAM_MARKER = {"explicit_compute_stream": False}


class _CollectorDiedError(RuntimeError):
    """Signal collector death to the run-level lifecycle owner."""


class DoubleBufferOffPolicyRunner(OffPolicyRunner):
    """Configure a double-buffer run and delegate its lifecycle to one session."""

    LEARNER_LOG_INTERVAL = 10

    def __init__(
        self,
        *,
        replay_prefetch_mode: str = "one_tick",
        verbose_metrics: bool = False,
        checkpoint_saver: CheckpointSaver | None = None,
        start_iteration: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if replay_prefetch_mode != "one_tick":
            raise ValueError(
                "DoubleBufferOffPolicyRunner only supports replay_prefetch_mode='one_tick'"
            )
        self.replay_prefetch_mode = replay_prefetch_mode
        self.verbose_metrics = bool(verbose_metrics)
        self.checkpoint_saver = checkpoint_saver
        self.start_iteration = start_iteration
        self.replay_pack_layout = "packed"
        self.replay_pack_executor = "collector_thread"
        self.replay_h2d_submitter = "auto"
        self.replay_transfer_backend: dict[str, object] = {}

    def _save_checkpoint(self, path: str | Path, *, iteration: int) -> None:
        target = Path(path)
        if self.checkpoint_saver is None:
            torch.save(self.learner.get_state_dict(), target)
            return
        self.checkpoint_saver(self.learner, target, int(iteration))

    def _save_iteration_checkpoint(self, log_dir: str | Path, *, iteration: int) -> str:
        """Persist one iteration under the matching canonical filename."""
        iteration_value = int(iteration)
        target = Path(log_dir) / f"model_{iteration_value}.pt"
        self._save_checkpoint(target, iteration=iteration_value)
        return str(target)

    def _fail_collector_died(
        self,
        logger,
        replay_buffer,
        replay_pipeline,
        iteration: int,
        ckpt_path: str | None,
        train_start_wall: float,
    ) -> None:
        logger.log_status("[red]ERROR: Collector died[/]")
        self._sync_logger_replay_counters(logger, replay_buffer)
        logger.close()
        self.last_run_summary = self._make_summary(
            "collector_died",
            iteration,
            logger,
            None,
            None,
            ckpt_path,
            train_start_wall,
            None,
        )
        replay_pipeline.close()
        raise RuntimeError("Collector process died during off-policy training")

    def _safe_put_trainer_done(
        self,
        queue,
        *,
        timeout: float = 5.0,
        label: str = "trainer_done",
    ) -> None:
        """Put a trainer-done token without deadlocking on collector death."""
        if queue is None:
            return
        deadline = time.monotonic() + timeout
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _CollectorDiedError(f"{label} (queue full timeout)")
                queue.put(1, timeout=min(0.5, remaining))
                return
            except queue_module.Full:
                if not self._check_collector_alive():
                    raise _CollectorDiedError(f"{label} (collector dead)")

    def _wait_for_replay_batch_ready(
        self,
        replay_pipeline,
        tick_id: int,
        sample_count: int,
        metrics_queue,
        reward_history,
        latest_reward_components,
        logger,
        trace_recorder,
        replay_buffer,
        ckpt_path: str | None,
        train_start_wall: float,
    ) -> bool:
        if not replay_pipeline.batch_ready(tick_id, sample_count):
            replay_pipeline.start_prepare(tick_id, sample_count)
        while not replay_pipeline.batch_ready(tick_id, sample_count):
            self._drain_metrics(
                metrics_queue,
                reward_history,
                latest_reward_components,
                logger,
                trace_recorder,
            )
            if not self._check_collector_alive():
                self._fail_collector_died(
                    logger,
                    replay_buffer,
                    replay_pipeline,
                    tick_id,
                    ckpt_path,
                    train_start_wall,
                )
            time.sleep(0.1)
        return True

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        options = DoubleBufferRunOptions(
            max_iterations, save_interval, log_dir, logger_type, self.start_iteration,
        )
        dependencies = DoubleBufferSessionDependencies(
            replay_buffer_cls=ReplayBuffer,
            replay_pipeline_cls=CPUPinnedDoubleBufferReplayPipeline,
            weight_sync_cls=SharedWeightSync,
            obs_norm_stats_cls=SharedObsNormStats,
            logger_cls=OffPolicyLogger,
            trace_recorder_cls=TraceRecorder,
            spawn_ctx=_SPAWN_CTX,
            collector_fn=off_policy_collector_fn,
            collector_died_error_cls=_CollectorDiedError,
            time_module=time,
        )
        DoubleBufferTrainingSession(self, options, dependencies).run()

    @staticmethod
    def _make_summary(
        status,
        iteration,
        logger,
        final_reward,
        best_reward,
        ckpt_path,
        train_start_wall,
        trace_path,
    ) -> dict:
        return {
            "status": status,
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": final_reward,
            "best_mean_reward": best_reward,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "trace_path": trace_path,
            "training_wall_time_sec": time.time() - train_start_wall,
        }
