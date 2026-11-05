"""Per-run lifecycle owner for the CPU-pinned double-buffer learner."""

from __future__ import annotations

import os
import queue
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.offpolicy.runner import (
    build_reward_comparison_metrics,
    replay_buffer_ready_for_learning,
)
from unilab.training.seed import derive_worker_seed


@dataclass(frozen=True)
class DoubleBufferRunOptions:
    max_iterations: int
    save_interval: int
    log_dir: str
    logger_type: str
    start_iteration: int = 0


@dataclass(frozen=True)
class DoubleBufferSessionDependencies:
    replay_buffer_cls: Any
    replay_pipeline_cls: Any
    weight_sync_cls: Any
    obs_norm_stats_cls: Any
    logger_cls: Any
    trace_recorder_cls: Any
    spawn_ctx: Any
    collector_fn: Any
    collector_died_error_cls: type[BaseException]
    time_module: Any


class DoubleBufferTrainingSession:
    """Own resources, counters, phases, and cleanup for one ``learn`` call."""

    def __init__(
        self,
        runner: Any,
        options: DoubleBufferRunOptions,
        dependencies: DoubleBufferSessionDependencies,
    ) -> None:
        self.runner = runner
        self.options = options
        self.dependencies = dependencies
        self.train_start_wall = dependencies.time_module.time()
        self.best_mean_reward = float("-inf")
        self.last_mean_reward = 0.0
        self.ckpt_path: str | None = None
        self.iteration = options.start_iteration
        self.reward_history: deque = deque(maxlen=100)
        self.latest_reward_components: dict[str, float] = {}
        self.last_buf_log = 0
        self.write_read_ema = 0.0
        self.reward_stats_ptr = 0
        self.prepared_tick: int | None = None

    def prepare(self) -> None:
        os.makedirs(self.options.log_dir, exist_ok=True)
        self._prepare_trace()
        self._validate_memory_budget()
        self._create_replay_resources()
        self._create_weight_sync()
        self._create_logger()
        self._create_coordination_resources()
        if self.options.start_iteration > 0:
            self._publish_normalizer()
        self._start_collector()
        self.dependencies.time_module.sleep(0.5)
        if self.runner._collector_process:
            print(
                "[DoubleBufferRunner] Collector process alive: "
                f"{self.runner._collector_process.is_alive()}"
            )
        self.training_e2e_start_ns = (
            self.dependencies.time_module.perf_counter_ns() if self.trace_recorder else 0
        )

    def _prepare_trace(self) -> None:
        self.trace_output_path: Path | None = None
        self.trace_recorder = None
        if not self.runner.trace_enabled:
            return
        trace_root = Path(self.runner.trace_output_dir or self.options.log_dir)
        self.trace_output_path = trace_root / "perfetto_offpolicy_timeline.json"
        self.trace_recorder = self.dependencies.trace_recorder_cls("offpolicy_learner")

    def _validate_memory_budget(self) -> None:
        from unilab.ipc.memory_budget import (
            estimate_offpolicy_bytes,
            raise_if_shared_memory_over_budget,
            warn_if_over_budget,
        )

        runner = self.runner
        estimate = estimate_offpolicy_bytes(
            num_envs=runner.num_envs,
            replay_buffer_n=runner.replay_buffer_n,
            obs_dim=runner.obs_dim,
            action_dim=runner.action_dim,
            critic_dim=runner.critic_obs_dim,
            batch_size=runner.batch_size,
            updates_per_step=runner.updates_per_step,
        )
        label = f"Off-policy ({runner.algo_type})"
        warn_if_over_budget(estimate, label=label)
        raise_if_shared_memory_over_budget(estimate, label=label)

    def _create_replay_resources(self) -> None:
        runner = self.runner
        self.replay_buffer = self.dependencies.replay_buffer_cls(
            capacity=runner.replay_buffer_n * runner.num_envs,
            obs_dim=runner.obs_dim,
            action_dim=runner.action_dim,
            device=runner.device,
            defer_gpu=True,
            critic_dim=runner.critic_obs_dim,
            packed_cpu_storage=runner.replay_pack_layout == "packed",
        )
        runner._shared_resources.append(self.replay_buffer)
        self.replay_buffer.trace_recorder = self.trace_recorder
        self.replay_buffer.trace_thread_time = runner.trace_thread_time
        self.replay_buffer.trace_cuda_events = runner.trace_cuda_events
        self.sample_count = runner.batch_size * runner.updates_per_step
        self.collector_pack_request_queue = self.dependencies.spawn_ctx.Queue(maxsize=1)
        self.collector_pack_ready_queue = self.dependencies.spawn_ctx.Queue(maxsize=1)
        packed_width = int(self.replay_buffer._storage.shape[1])
        self.collector_pack_shared_slots = [
            torch.empty((self.sample_count, packed_width), dtype=torch.float32).share_memory_()
            for _ in range(2)
        ]
        verbose_output_dir = None
        if runner.verbose_metrics:
            root = (
                Path(runner.trace_output_dir)
                if runner.trace_output_dir
                else Path(self.options.log_dir)
            )
            verbose_output_dir = str(root)
        self.replay_pipeline = self.dependencies.replay_pipeline_cls(
            self.replay_buffer,
            device=runner.device,
            sample_count=self.sample_count,
            base_seed=int(runner.seed or 0),
            trace_recorder=self.trace_recorder,
            trace_cuda_events=runner.trace_cuda_events,
            verbose=runner.verbose_metrics,
            verbose_output_dir=verbose_output_dir,
            collector_pack_request_queue=self.collector_pack_request_queue,
            collector_pack_ready_queue=self.collector_pack_ready_queue,
            collector_pack_shared_slots=self.collector_pack_shared_slots,
        )
        runner.replay_h2d_submitter = getattr(
            self.replay_pipeline,
            "h2d_submitter",
            runner.replay_h2d_submitter,
        )
        runner.replay_transfer_backend = getattr(
            self.replay_pipeline,
            "transfer_manifest",
            {},
        )

    def _create_weight_sync(self) -> None:
        runner = self.runner
        self.weight_sync = self.dependencies.weight_sync_cls.from_state_dict(
            runner.learner.actor.state_dict(),
            create=True,
        )
        runner._shared_resources.append(self.weight_sync)
        self.weight_sync.trace_recorder = self.trace_recorder
        self.weight_sync.trace_thread_time = runner.trace_thread_time

    def _create_logger(self) -> None:
        runner = self.runner
        options = self.options
        self.logger = self.dependencies.logger_cls(
            algo_name=f"Fast{runner.algo_type.upper()}",
            max_iterations=options.max_iterations,
            num_envs=runner.num_envs,
            env_name=runner.env_name,
            obs_dim=runner.obs_dim,
            action_dim=runner.action_dim,
            log_dir=options.log_dir,
            log_backend=options.logger_type,
        )
        self.logger.set_collection_sync(runner.sync_collection, runner.env_steps_per_sync)
        if hasattr(runner.learner, "use_symmetry") and runner.learner.use_symmetry:
            self.logger.log_status("Symmetry augmentation: enabled")
        self.logger.log_status("Replay pipeline: cpu_pinned_double_buffer")
        self.logger.log_status(f"Replay prefetch mode: {runner.replay_prefetch_mode}")
        self.logger.log_status(f"Replay pack layout: {runner.replay_pack_layout}")
        self.logger.log_status(f"Replay pack executor: {runner.replay_pack_executor}")
        self.logger.log_status(f"Replay H2D submitter: {runner.replay_h2d_submitter}")
        if runner.replay_transfer_backend:
            self.logger.log_status(
                "Replay transfer backend: "
                f"{runner.replay_transfer_backend.get('backend')} "
                f"({runner.replay_transfer_backend.get('device_family')})"
            )
        self.logger.log_status(
            f"Replay learner lightweight: fixed (log_interval={runner.LEARNER_LOG_INTERVAL})"
        )
        if runner.verbose_metrics:
            self.logger.log_status("Verbose metrics: enabled (field-level pack CSV)")
        runner._active_logger = self.logger
        self.logger.start()

    def _create_coordination_resources(self) -> None:
        runner = self.runner
        self.collection_ready_queue = None
        self.trainer_done_queue = None
        if runner.sync_collection:
            self.collection_ready_queue = self.dependencies.spawn_ctx.Queue(maxsize=1)
            self.trainer_done_queue = self.dependencies.spawn_ctx.Queue(maxsize=1)
            runner._safe_put_trainer_done(self.trainer_done_queue, label="init")
            print(
                "[DoubleBufferRunner] Collection sync enabled: "
                f"env_steps_per_sync={runner.env_steps_per_sync}"
            )
        self.metrics_queue = self.dependencies.spawn_ctx.Queue(maxsize=100)
        self.shared_obs_normalizer_stats = None
        if runner.obs_normalization:
            self.shared_obs_normalizer_stats = self.dependencies.obs_norm_stats_cls(
                self.dependencies.spawn_ctx
            )

    def _start_collector(self) -> None:
        runner = self.runner
        weight_param_shapes = {
            key: value.shape for key, value in runner.learner.actor.state_dict().items()
        }
        collector_kwargs = {
            "env_name": runner.env_name,
            "num_envs": runner.num_envs,
            "replay_buffer": self.replay_buffer,
            "weight_sync_name": self.weight_sync.name,
            "weight_sync_lock": self.weight_sync._lock,
            "weight_param_shapes": weight_param_shapes,
            "algo_type": runner.algo_type,
            "actor_hidden_dim": runner.actor_hidden_dim,
            "use_layer_norm": runner.use_layer_norm,
            "learning_starts": runner.learning_starts,
            "metrics_queue": self.metrics_queue,
            "sync_collection": runner.sync_collection,
            "collection_ready_queue": self.collection_ready_queue,
            "trainer_done_queue": self.trainer_done_queue,
            "env_steps_per_sync": runner.env_steps_per_sync,
            "obs_normalization": runner.obs_normalization,
            "shared_obs_normalizer_stats": self.shared_obs_normalizer_stats,
            "sim_backend": runner.sim_backend,
            "env_cfg_override": runner.env_cfg_override,
            "obs_dim": runner.obs_dim,
            "action_dim": runner.action_dim,
            "actor_kwargs": runner.actor_kwargs,
            "seed": derive_worker_seed(runner.seed, worker_index=0),
            "trace_enabled": runner.trace_enabled,
            "trace_thread_time": runner.trace_thread_time,
            "nan_guard_cfg": runner.nan_guard_cfg,
            "collector_pack_request_queue": self.collector_pack_request_queue,
            "collector_pack_ready_queue": self.collector_pack_ready_queue,
            "collector_pack_shared_slots": self.collector_pack_shared_slots,
        }
        runner._start_collector(
            target_fn=self.dependencies.collector_fn,
            kwargs={"stop_event": runner._stop_event, **collector_kwargs},
        )

    def _fail_collector_died(self) -> None:
        self.runner._fail_collector_died(
            self.logger,
            self.replay_buffer,
            self.replay_pipeline,
            self.iteration,
            self.ckpt_path,
            self.train_start_wall,
        )

    def _wait_for_data(self) -> float:
        runner = self.runner
        started = self.dependencies.time_module.time()
        started_ns = self.dependencies.time_module.perf_counter_ns()
        if runner.sync_collection and self.collection_ready_queue:
            self._wait_for_synchronized_data()
        else:
            self._wait_for_asynchronous_data()
        wait_time = self.dependencies.time_module.time() - started
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/wait_for_data",
                category="learner",
                start_ns=started_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
                args={"iteration": self.iteration},
            )
        runner._drain_metrics(
            self.metrics_queue,
            self.reward_history,
            self.latest_reward_components,
            self.logger,
            self.trace_recorder,
        )
        reward_stats_ns = self.dependencies.time_module.perf_counter_ns()
        self.reward_stats_ptr = runner._update_reward_stats_from_replay(
            self.replay_buffer,
            self.reward_stats_ptr,
            int(self.replay_buffer.ptr[0]),
        )
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/update_reward_stats",
                category="learner",
                start_ns=reward_stats_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
            )
        return wait_time

    def _wait_for_synchronized_data(self) -> None:
        runner = self.runner
        while True:
            try:
                self.collection_ready_queue.get(timeout=1.0)
            except queue.Empty:
                if not runner._check_collector_alive():
                    runner._drain_metrics(
                        self.metrics_queue,
                        self.reward_history,
                        self.latest_reward_components,
                        self.logger,
                        self.trace_recorder,
                    )
                    self._fail_collector_died()
                continue
            runner._drain_metrics(
                self.metrics_queue,
                self.reward_history,
                self.latest_reward_components,
                self.logger,
                self.trace_recorder,
            )
            current_size = int(self.replay_buffer.size[0])
            if replay_buffer_ready_for_learning(
                current_size,
                batch_size=runner.batch_size,
                learning_starts=runner.learning_starts,
                num_envs=runner.num_envs,
            ):
                if self.prepared_tick != self.iteration:
                    self.replay_pipeline.start_prepare(self.iteration, self.sample_count)
                    self.prepared_tick = self.iteration
                return
            self._log_buffer_fill(current_size)
            if self.trainer_done_queue:
                runner._safe_put_trainer_done(
                    self.trainer_done_queue,
                    label="buffer_wait",
                )

    def _wait_for_asynchronous_data(self) -> None:
        runner = self.runner
        while not replay_buffer_ready_for_learning(
            int(self.replay_buffer.size[0]),
            batch_size=runner.batch_size,
            learning_starts=runner.learning_starts,
            num_envs=runner.num_envs,
        ):
            if not runner._check_collector_alive():
                runner._drain_metrics(
                    self.metrics_queue,
                    self.reward_history,
                    self.latest_reward_components,
                    self.logger,
                )
                self._fail_collector_died()
            self._log_buffer_fill(int(self.replay_buffer.size[0]))
            self.dependencies.time_module.sleep(0.1)
            runner._drain_metrics(
                self.metrics_queue,
                self.reward_history,
                self.latest_reward_components,
                self.logger,
                self.trace_recorder,
            )

    def _log_buffer_fill(self, current_size: int) -> None:
        threshold = self.runner.train_start_threshold
        if current_size - self.last_buf_log < self.runner.num_envs * 10:
            return
        self.last_buf_log = current_size
        self.logger.log_buffer_fill(current_size, threshold)

    def _sample_replay_batch(self) -> tuple[dict[str, torch.Tensor], float, bool]:
        runner = self.runner
        sample_ns = self.dependencies.time_module.perf_counter_ns()
        batch_ready = self.replay_pipeline.batch_ready(self.iteration, self.sample_count)
        wait_ns = self.dependencies.time_module.perf_counter_ns()
        if not batch_ready:
            batch_ready = runner._wait_for_replay_batch_ready(
                self.replay_pipeline,
                self.iteration,
                self.sample_count,
                self.metrics_queue,
                self.reward_history,
                self.latest_reward_components,
                self.logger,
                self.trace_recorder,
                self.replay_buffer,
                self.ckpt_path,
                self.train_start_wall,
            )
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/wait_for_replay_batch",
                category="learner",
                start_ns=wait_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
                args={"iteration": self.iteration, "batch_ready": batch_ready},
            )
        large_batch = self.replay_pipeline.sample_large_batch(
            tick_id=self.iteration,
            sample_count=self.sample_count,
        )
        h2d_time = float(getattr(self.replay_pipeline, "last_incremental_h2d_time_s", 0.0))
        collector_released = self._prefetch_next_iteration()
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/replay_sample",
                category="learner",
                start_ns=sample_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
                args={
                    "total_batch": self.sample_count,
                    "pipeline": "cpu_pinned_double_buffer",
                    "batch_ready": batch_ready,
                    "prefetch_mode": runner.replay_prefetch_mode,
                    "replay_pack_layout": runner.replay_pack_layout,
                    "replay_pack_executor": runner.replay_pack_executor,
                    "replay_h2d_submitter": runner.replay_h2d_submitter,
                    "replay_transfer_backend": runner.replay_transfer_backend,
                    "prepared_tick": self.prepared_tick,
                    "explicit_compute_stream": False,
                },
            )
        return large_batch, h2d_time, collector_released

    def _prefetch_next_iteration(self) -> bool:
        runner = self.runner
        if self.iteration >= self.options.max_iterations:
            return False
        minimum_ptr = int(self.replay_buffer.ptr[0]) + (runner.num_envs * runner.env_steps_per_sync)
        self.replay_pipeline.start_prepare(
            self.iteration + 1,
            self.sample_count,
            min_snapshot_ptr=minimum_ptr,
        )
        collector_released = False
        if runner.sync_collection and self.trainer_done_queue:
            runner._safe_put_trainer_done(self.trainer_done_queue, label="tick_release")
            collector_released = True
        self.prepared_tick = self.iteration + 1
        return collector_released

    def _update_learner(self, large_batch: dict[str, torch.Tensor]) -> dict[str, list[Any]]:
        runner = self.runner
        learner = runner.learner
        metrics: defaultdict[str, list[Any]] = defaultdict(list)
        for update_index in range(runner.updates_per_step):
            start = update_index * runner.batch_size
            end = start + runner.batch_size
            batch = {key: value[start:end] for key, value in large_batch.items()}
            critic_ns = self.dependencies.time_module.perf_counter_ns()
            for key, value in learner.update_critic(batch).items():
                metrics[key].append(value)
            if self.trace_recorder:
                self.trace_recorder.add_slice(
                    "learner/update_critic",
                    category="learner",
                    start_ns=critic_ns,
                    end_ns=self.dependencies.time_module.perf_counter_ns(),
                    args={"update_idx": update_index},
                )
            if update_index % runner.policy_frequency == 0:
                actor_ns = self.dependencies.time_module.perf_counter_ns()
                for key, value in learner.update_actor(batch).items():
                    metrics[key].append(value)
                if self.trace_recorder:
                    self.trace_recorder.add_slice(
                        "learner/update_actor",
                        category="learner",
                        start_ns=actor_ns,
                        end_ns=self.dependencies.time_module.perf_counter_ns(),
                        args={"update_idx": update_index},
                    )
            target_ns = self.dependencies.time_module.perf_counter_ns()
            learner.soft_update_target()
            if self.trace_recorder:
                self.trace_recorder.add_slice(
                    "learner/soft_update_target",
                    category="learner",
                    start_ns=target_ns,
                    end_ns=self.dependencies.time_module.perf_counter_ns(),
                    args={"update_idx": update_index},
                )
        self.replay_pipeline.after_tick()
        return metrics

    def _publish_normalizer(self) -> None:
        learner = self.runner.learner
        if not self.runner.obs_normalization or getattr(learner, "obs_normalizer", None) is None:
            return
        assert self.shared_obs_normalizer_stats is not None
        self.shared_obs_normalizer_stats.put(
            (
                learner.obs_normalizer.mean.cpu().numpy(),
                learner.obs_normalizer.std.cpu().numpy(),
            )
        )

    def _sync_weights(self) -> float:
        started_ns = self.dependencies.time_module.perf_counter_ns()
        started = self.dependencies.time_module.perf_counter()
        self.weight_sync.write_weights(self.runner.learner.actor.state_dict())
        elapsed = self.dependencies.time_module.perf_counter() - started
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/weight_sync_write",
                category="learner",
                start_ns=started_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
                args={"mode": "sync"},
            )
            self.trace_recorder.add_counter(
                "replay_size",
                int(self.replay_buffer.size[0]),
                category="replay",
            )
        return elapsed

    def _publish_iteration(
        self,
        *,
        metrics: dict[str, list[Any]],
        wait_time: float,
        train_time: float,
        h2d_time: float,
        weight_sync_time: float,
    ) -> None:
        runner = self.runner
        average_metrics = {
            key: statistics.mean(values) for key, values in metrics.items() if values
        }
        mean_reward = statistics.mean(self.reward_history) if self.reward_history else 0.0
        self.last_mean_reward = float(mean_reward)
        self.best_mean_reward = max(self.best_mean_reward, self.last_mean_reward)
        if (
            self.iteration == 1
            or self.iteration == self.options.max_iterations
            or self.iteration % runner.LEARNER_LOG_INTERVAL == 0
        ):
            runner._sync_logger_replay_counters(self.logger, self.replay_buffer)
            self.logger.log_step(
                iteration=self.iteration,
                metrics=average_metrics,
                reward=mean_reward,
                reward_metrics=build_reward_comparison_metrics(
                    self.reward_history,
                    mean_reward,
                ),
                reward_components=self.latest_reward_components,
                train_time=train_time,
                wait_time=wait_time,
                learner_incremental_h2d_time=h2d_time,
                weight_sync_time=weight_sync_time,
                extra_info={"throughput_steps": runner.num_envs * runner.env_steps_per_sync},
            )
        if self.options.save_interval > 0 and self.iteration % self.options.save_interval == 0:
            self.ckpt_path = runner._save_iteration_checkpoint(
                self.options.log_dir,
                iteration=self.iteration,
            )
            self.logger.log_save(self.ckpt_path)

    def _run_iteration(self) -> None:
        runner = self.runner
        wait_time = self._wait_for_data()
        ptr_before = int(self.replay_buffer.ptr[0])
        large_batch, h2d_time, collector_released = self._sample_replay_batch()
        train_started = self.dependencies.time_module.time()
        metrics = self._update_learner(large_batch)
        self._publish_normalizer()
        train_time = self.dependencies.time_module.time() - train_started
        runner.learner.update_count += 1
        weight_sync_time = self._sync_weights()
        if runner.sync_collection and self.trainer_done_queue and not collector_released:
            runner._safe_put_trainer_done(self.trainer_done_queue, label="weight_sync")
        write_delta = int(self.replay_buffer.ptr[0]) - ptr_before
        consumed = runner.batch_size * runner.updates_per_step
        self.write_read_ema = 0.9 * self.write_read_ema + 0.1 * (write_delta / max(consumed, 1))
        self.logger.update_buffer_utilization(self.write_read_ema)
        self._publish_iteration(
            metrics=metrics,
            wait_time=wait_time,
            train_time=train_time,
            h2d_time=h2d_time,
            weight_sync_time=weight_sync_time,
        )

    def _finalize_success(self) -> None:
        runner = self.runner
        if self.trace_recorder:
            self.trace_recorder.add_slice(
                "learner/training_e2e",
                category="learner",
                start_ns=self.training_e2e_start_ns,
                end_ns=self.dependencies.time_module.perf_counter_ns(),
                args={
                    "iterations": self.iteration,
                    "pipeline": "cpu_pinned_double_buffer",
                    "replay_h2d_submitter": runner.replay_h2d_submitter,
                    "replay_transfer_backend": runner.replay_transfer_backend,
                    "learner_log_interval": runner.LEARNER_LOG_INTERVAL,
                },
            )
        self.replay_pipeline.close()
        self.ckpt_path = runner._save_iteration_checkpoint(
            self.options.log_dir,
            iteration=self.options.max_iterations,
        )
        self.logger.log_save(self.ckpt_path)
        runner._sync_logger_replay_counters(self.logger, self.replay_buffer)
        self.logger.finish()
        if self.trace_recorder and self.trace_output_path:
            self.trace_recorder.write_json(self.trace_output_path)
            print(f"[DoubleBufferRunner] Perfetto trace written to {self.trace_output_path}")
        runner.last_run_summary = runner._make_summary(
            "completed",
            self.iteration,
            self.logger,
            self.last_mean_reward if self.reward_history else None,
            self.best_mean_reward if self.reward_history else None,
            self.ckpt_path,
            self.train_start_wall,
            str(self.trace_output_path) if self.trace_output_path else None,
        )
        runner._active_logger = None

    def _handle_collector_died(self) -> None:
        self._fail_collector_died()

    def run(self) -> None:
        if not 0 <= self.options.start_iteration < self.options.max_iterations:
            raise ValueError("start_iteration must be within [0, max_iterations)")
        self.prepare()
        try:
            for self.iteration in range(self.options.start_iteration + 1, self.options.max_iterations + 1):
                self._run_iteration()
            self._finalize_success()
        except self.dependencies.collector_died_error_cls:
            self._handle_collector_died()
            raise
