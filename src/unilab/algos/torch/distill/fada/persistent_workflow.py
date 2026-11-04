"""Official FADA persistent-async training orchestration (parent-side owner).

Runs the single formal official FADA loop in the parent process:

``prepare`` the resident official worker, collect the optimal offline source
(and the prepare-selected weak-checkpoint suboptimal sweep) unless the original
NPZ offline caches admit reuse, finalize the offline replay (original
trajectory train/val split + freeze), run ``warmup_train_steps`` offline-mode
updates under the original ``load_warmup_ckpt``/``force_warmup_train``
precedence, then run ``dagger_iters`` rounds of online raw collection,
parent-side trajectory subsample/split, mixed-mode updates and weight
activation. The parent alone owns the student, both optimizers, all counters
and the final normal close; the worker owns the native env/trainer/
strict-shadow and never trains.

Every collection goes over the bounded shared-memory transport into the
parent's own raw staging (``pending_raw_budget_bytes``), so no raw trajectory is
handed over as a file. Learning consumes that staging through the finite batch
pipeline, the learner's latest weights are published in memory, and one
single-flight committer persists each round's Replay snapshot plus checkpoint
pair while the next round's collection is already running.

Resume restores the learner (model, both optimizers, update counter) plus the
parent counters (``rollout_step_count``, ``completed_iterations``,
``samples_seen``) and the online replay through the original NPZ schema, then
runs warmup in cache-only form exactly like upstream resumed runs. Physical
environment/RNG replay is not claimed.
"""

from __future__ import annotations

import json
import math
import random
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from unilab.algos.torch.distill.fada.async_runtime import build_official_fada_runtime
from unilab.algos.torch.distill.fada.batch_pipeline import OfficialFADABatchPipeline
from unilab.algos.torch.distill.fada.checkpoint import (
    load_official_fada_checkpoint,
    load_official_fada_student_checkpoint,
    save_official_fada_checkpoint,
)
from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SCENARIO,
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADACollectRequest,
    OfficialFADACollectionSpec,
    OfficialFADARawTransitionBatch,
    official_raw_transition_layout,
)
from unilab.algos.torch.distill.fada.execution_contract import (
    FADAExecutionLimits,
    layout_nbytes,
    raw_chunk_rows,
)
from unilab.algos.torch.distill.fada.official_learning import (
    OfficialFADALearner,
    canonical_official_training_config,
)
from unilab.algos.torch.distill.fada.official_model import (
    build_official_fada_student,
    canonical_official_model_config,
)
from unilab.algos.torch.distill.fada.persistence import FADAIterationCommitter
from unilab.algos.torch.distill.fada.replay import OfficialFADAReplayBuffer
from unilab.algos.torch.distill.fada.shared_collection import (
    collect_shared_raw,
    raw_transport_layout,
)
from unilab.algos.torch.distill.fada.timing import FADAPerformanceRecorder, measure
from unilab.algos.torch.distill.runtime.async_runtime import DaggerPrepareRequest
from unilab.ipc.shared_tensor_channel import SharedTensorChannel

__all__ = ["run_fada_persistent_async"]

_PREPARE_METADATA_FIELDS = (
    "upstream_revision",
    "expert_checkpoint",
    "expert_checkpoint_sha256",
    "experiment_config_sha256",
    "preprocessing",
    "resolved_config",
    "model_config",
    "weak_checkpoints",
)
_RESUME_ONLY_FIELDS = (
    "resume_checkpoint",
    "resume_online_buffer",
    "load_warmup_ckpt",
    "force_warmup_train",
    "force_recollect_offline",
    "offline_cache_path",
    "offline_suboptimal_cache_path",
    "warmup_ckpt_path",
)


def _validate_prepared_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("official prepare metadata must be a mapping")
    missing = [name for name in _PREPARE_METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"official prepare metadata is incomplete: {missing}")
    if metadata["upstream_revision"] != OFFICIAL_FADA_SOURCE_REVISION:
        raise ValueError("official prepare upstream revision mismatch")
    for name in ("expert_checkpoint_sha256", "experiment_config_sha256"):
        value = metadata[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"official prepare {name} must be a SHA256 identity")
    if not isinstance(metadata["resolved_config"], Mapping):
        raise ValueError("official prepare resolved_config must be a mapping")
    if not isinstance(metadata["weak_checkpoints"], (list, tuple)) or not all(
        isinstance(item, str) for item in metadata["weak_checkpoints"]
    ):
        raise ValueError("official prepare weak_checkpoints must be a list of paths")
    return dict(metadata)


def _validate_resolved_config(resolved: Mapping[str, Any], spec: OfficialFADACollectionSpec) -> None:
    """Reject drift between the worker-resolved config and the declared spec."""

    mismatches = {
        name: (spec.config.get(name), resolved.get(name))
        for name in sorted(set(spec.config) - set(_RESUME_ONLY_FIELDS))
        if name in resolved and name != "wandb_mode" and resolved.get(name) != spec.config.get(name)
    }
    if mismatches:
        raise ValueError(f"official resolved config contradicts the collection spec: {mismatches}")
    if int(resolved.get("history_len", -1)) != spec.history_length or int(
        resolved.get("pred_horizon", -1)
    ) != spec.prediction_horizon:
        raise ValueError("official resolved config history/prediction horizon mismatch")


def _source_identity(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        "upstream_revision": str(metadata["upstream_revision"]),
        "expert_checkpoint": str(metadata["expert_checkpoint"]),
        "expert_checkpoint_sha256": str(metadata["expert_checkpoint_sha256"]),
        "experiment_config_sha256": str(metadata["experiment_config_sha256"]),
    }


def _set_global_seed(seed: int) -> None:
    """Mirror upstream ``set_seed``: global np.random/torch drive replay + dropout."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_loaded_source_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    context: str,
) -> None:
    """Reject a same-shaped checkpoint from a different expert/run.

    The checkpoint loader verifies structural configs/preprocessing but cannot
    know this run's prepared Oracle; provenance is compared here before any
    further collection consumes the loaded weights.
    """

    identity = payload.get("source_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"official {context} payload is missing source_identity")
    mismatches = {
        name: (expected[name], identity.get(name))
        for name in ("upstream_revision", "expert_checkpoint", "expert_checkpoint_sha256", "experiment_config_sha256")
        if identity.get(name) != expected[name]
    }
    if mismatches:
        raise ValueError(
            f"official {context} provenance mismatch against the prepared Oracle: {mismatches}"
        )


def run_fada_persistent_async(
    *,
    spec: OfficialFADACollectionSpec,
    model_config: Mapping[str, Any],
    device: str,
    output_dir: Path,
    execution_limits: FADAExecutionLimits,
    pending_raw_budget_bytes: int,
    request_timeout_seconds: float = 300,
    runtime_factory: Callable[..., Any] = build_official_fada_runtime,
    performance_enabled: bool = False,
) -> dict[str, Any]:
    """Run the official FADA warmup + DAgger loop with a persistent collector."""

    if not isinstance(spec, OfficialFADACollectionSpec):
        raise TypeError(
            f"official FADA training requires an OfficialFADACollectionSpec, got {type(spec)}"
        )
    canonical_model_config = canonical_official_model_config(model_config)
    target_device = torch.device(device)
    if not isinstance(performance_enabled, bool):
        raise ValueError("performance_enabled must be a boolean")
    if not isinstance(execution_limits, FADAExecutionLimits):
        raise TypeError(
            "official FADA training requires FADAExecutionLimits execution_limits, "
            f"got {type(execution_limits).__name__}"
        )
    if (
        isinstance(pending_raw_budget_bytes, bool)
        or not isinstance(pending_raw_budget_bytes, int)
        or pending_raw_budget_bytes <= 0
    ):
        raise ValueError(
            "official FADA training requires a positive integer pending_raw_budget_bytes, "
            f"got {pending_raw_budget_bytes!r}"
        )
    channel: SharedTensorChannel
    # Assigned once the DAgger phase builds its single-flight writer; the close
    # barrier below drains it before the runtime closes.
    committer: FADAIterationCommitter | None = None
    performance = FADAPerformanceRecorder(device=target_device) if performance_enabled else None
    outcome: dict[str, Any] | None = None
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (run_dir / "checkpoints" / "student_latest.pt").resolve()

    # The resident worker is built with the explicit initial student; its shared
    # weight layout is registered from this exact model before prepare runs.
    with measure(performance, "cold_start", group="parent_setup"):
        _set_global_seed(int(spec.config.get("seed", 42)))
        initial_student = build_official_fada_student(canonical_model_config, device="cpu")
        capacity = raw_chunk_rows(official_raw_transition_layout(spec, 1), execution_limits)
        channel = SharedTensorChannel.create(
            raw_transport_layout(spec, execution_limits, max_rows=capacity),
            slots=execution_limits.slots,
        )
        try:
            runtime = runtime_factory(
                spec=spec,
                student_loader=load_official_fada_student_checkpoint,
                initial_student=initial_student,
                request_timeout_seconds=float(request_timeout_seconds),
                **({"performance_enabled": True} if performance_enabled else {}),
                shared_endpoint=channel.endpoint,
                shared_timeout_seconds=float(request_timeout_seconds),
            )
        except BaseException:
            channel.cleanup()
            raise
    try:
        with measure(performance, "cold_start", group="prepare_rpc"):
            prepared = runtime.prepare(DaggerPrepareRequest(request_id="official-fada-prepare"))
        metadata = _validate_prepared_metadata(prepared.metadata)
        if performance is not None:
            performance.add_worker_metrics(
                prepared.metadata.get("performance_metrics", {}), source="prepare"
            )
        resolved_config = dict(metadata["resolved_config"])
        _validate_resolved_config(resolved_config, spec)
        preprocessing = metadata["preprocessing"]
        weak_checkpoints = tuple(str(path) for path in metadata["weak_checkpoints"])
        if canonical_official_model_config(metadata["model_config"]) != canonical_model_config:
            raise ValueError(
                "official model_config mismatch: the caller's model config does not equal "
                "the worker-resolved canonical config; refusing to silently rebuild "
                "another initial model"
            )
        training_config = canonical_official_training_config(
            {name: resolved_config[name] for name in canonical_official_training_config() if name in resolved_config}
        )
        expected_identity = _source_identity(metadata)
        seed = int(resolved_config["seed"])
        num_envs = int(resolved_config["num_envs"])
        if num_envs <= 0:
            raise ValueError("official resolved config num_envs must be positive")
        _set_global_seed(seed)

        # The learner exists only after the actual prepare preprocessing is bound.
        with measure(performance, "cold_start", group="learner_replay_init"):
            learner = OfficialFADALearner(
                initial_student.to(target_device),
                training_config=training_config,
                preprocessing=preprocessing,
                performance=performance,
            )
            replay = OfficialFADAReplayBuffer(
                spec,
                preprocessing=preprocessing,
                training_config=training_config,
                resolved_config=resolved_config,
                source_identity=expected_identity,
                performance=performance,
            )

        def _train(mode: str, count: int) -> dict[str, float]:
            """One bounded update stage: prefetched batches through the finite pipeline."""
            last: dict[str, float] = {}
            original_recorder = replay._performance
            sampling_recorder = (
                FADAPerformanceRecorder(device="cpu") if performance is not None else None
            )
            # Never mutate the learner's recorder from the sampler thread.
            replay._performance = sampling_recorder
            try:
                with OfficialFADABatchPipeline(
                    sample_cpu=replay.sample_update, spec=spec,
                    training_config=training_config, mode=mode,
                    planned_updates=count, device=target_device, limits=execution_limits,
                ) as pipeline:
                    for index in range(count):
                        # Entering the lease is what actually waits for a staged batch, so the
                        # batch_build window covers acquisition, not the learner update.
                        with ExitStack() as lease:
                            with measure(
                                performance, "learner_batch_staging", group="batch_build"
                            ):
                                idm_batches, planner_batch = lease.enter_context(
                                    pipeline.batch()
                                )
                            with measure(
                                performance, "total_elapsed", group="learner_update"
                            ):
                                last = learner.update(
                                    idm_batches, planner_batch,
                                    collect_metrics=index == count - 1,
                                )
            finally:
                replay._performance = original_recorder
            if sampling_recorder is not None:
                report = sampling_recorder.summary()
                performance.add_worker_metrics(
                    {f"{key}_seconds": value for key, value in report["wall_seconds"].items()},
                    source="cpu_sampler",
                )
                for name, value in report["counters"].items():
                    performance.increment(name, value)
            return last

        def _collect(
            request: OfficialFADACollectRequest,
        ) -> tuple[OfficialFADARawTransitionBatch, int]:
            steps = (
                math.ceil(request.collect_steps / num_envs)
                if request.collect_steps is not None else int(resolved_config[
                    "rollout_steps_per_iter" if request.source == "online"
                    else "warmup_max_steps_per_episode"
                ])
            )
            max_rows = steps * num_envs
            if layout_nbytes(official_raw_transition_layout(spec, max_rows)) > pending_raw_budget_bytes:
                raise ValueError("pending raw budget cannot retain this collection transaction")
            with measure(performance, "total_elapsed", group="collector_rpc"):
                batch, result = collect_shared_raw(
                    runtime, request, channel, spec=spec, max_rows=max_rows,
                    pending_budget_bytes=pending_raw_budget_bytes,
                    expected_metadata=metadata, timeout=float(request_timeout_seconds),
                )
            if performance is not None:
                performance.add_worker_metrics(result.metrics, source=request.source)
                performance.increment(f"raw_transitions.{request.source}", batch.num_samples)
                performance.increment(f"collector_requests.{request.source}")
            return batch, int(result.metadata["collected_steps"])

        def _admit(
            batch: OfficialFADARawTransitionBatch,
            source: str,
            required_length: int | None = None,
        ) -> int:
            return replay.add_raw_batch(
                batch, metadata={**metadata, "source": source},
                required_length=required_length,
            )

        def _activate() -> int:
            with measure(performance, "weight_sync", group="publication"):
                return runtime.activate_checkpoint(
                    checkpoint_path, state_dict=learner.policy.state_dict()
                )

        completed_iterations = 0
        samples_seen = 0
        rollout_step_count = 0
        start_iteration = 0
        online_target: Path | None = None
        resume_path_value = resolved_config.get("resume_checkpoint")
        if resume_path_value:
            resume_path = Path(str(resume_path_value)).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"official resume checkpoint not found: {resume_path}"
                )
            payload = load_official_fada_checkpoint(
                resume_path, learner, map_location=target_device
            )
            _check_loaded_source_identity(
                payload, expected_identity, context="resume checkpoint"
            )
            completed_iterations = int(payload["completed_iterations"])
            samples_seen = int(payload["samples_seen"])
            start_iteration = completed_iterations
            # Schema-2 guarantees a valid counter; it is the online seed base.
            rollout_step_count = int(payload["rollout_step_count"])
            # Restore the online replay exactly once: the checkpoint's own
            # counter-tagged reference wins, else the explicit config path; a
            # missing referenced snapshot is an error, never an empty start.
            online_ref = payload.get("online_buffer_path")
            explicit_online = resolved_config.get("resume_online_buffer")
            if explicit_online:
                online_target = Path(str(explicit_online)).expanduser().resolve()
            elif online_ref is not None:
                candidate = Path(str(online_ref))
                online_target = (
                    candidate
                    if candidate.is_absolute()
                    else (resume_path.parent / candidate).resolve()
                )
            if online_target is not None:
                if not online_target.is_file():
                    raise FileNotFoundError(
                        f"official online replay snapshot not found: {online_target}"
                    )
                loaded = replay.load_online_buffer(online_target)
                if loaded <= 0:
                    raise ValueError(
                        f"official online replay snapshot is empty: {online_target}"
                    )

        # The genuine initial/resumed checkpoint is published before any collect.
        with measure(performance, "checkpoint_save", group="initial_model"):
            save_official_fada_checkpoint(
                checkpoint_path,
                learner,
                completed_iterations=completed_iterations,
                samples_seen=samples_seen,
                source_identity=expected_identity,
                rollout_step_count=rollout_step_count,
                online_buffer_path=online_target,
            )
        weight_version = _activate()

        request_counter = 0
        collection_iteration = completed_iterations

        def _request(source: str, weak_checkpoint: str | None, collect_steps: int | None) -> OfficialFADACollectRequest:
            nonlocal request_counter
            request_counter += 1
            return OfficialFADACollectRequest(
                request_id=f"official-fada-{request_counter:04d}-v{weight_version}",
                scenario=OFFICIAL_FADA_SCENARIO,
                iteration=collection_iteration,
                checkpoint_path=str(checkpoint_path),
                output_path="",
                expected_weight_version=weight_version,
                source=source,  # type: ignore[arg-type]
                weak_checkpoint=weak_checkpoint,
                collect_steps=collect_steps,
                transport_id=channel.transport_id,
            )

        # ── Offline optimal collection (original warmup budget, cache reuse) ──
        # The configured target is computed BEFORE any collection, matching
        # upstream _target_optimal_offline_steps; the suboptimal budget below
        # derives from this configured target, never from actual retained rows.
        horizon = int(resolved_config["warmup_max_steps_per_episode"])
        if horizon <= 0:
            raise ValueError("official warmup_max_steps_per_episode must be positive")
        horizon_budget = horizon * num_envs
        warmup_collect_steps = resolved_config.get("warmup_collect_steps")
        optimal_target_steps = (
            int(warmup_collect_steps)
            if warmup_collect_steps is not None
            else int(resolved_config["warmup_episodes"]) * num_envs * horizon
        )
        optimal_cache_path = Path(str(resolved_config["offline_cache_path"]))
        suboptimal_cache_path = Path(str(resolved_config["offline_suboptimal_cache_path"]))
        cache_stats: dict[str, Any] = {"offline_cache_hit": False}
        if not bool(resolved_config.get("force_recollect_offline", False)):
            try:
                with measure(performance, "cumulative_aggregation", group="offline_cache_read"):
                    cache_stats = replay.load_offline_caches(
                        optimal_path=optimal_cache_path,
                        suboptimal_path=suboptimal_cache_path,
                    )
            except replay._cache_mismatch_errors as exc:  # type: ignore[attr-defined]
                raise RuntimeError(
                    "official offline cache compact-obs metadata mismatch; delete the "
                    "caches or set force_recollect_offline=true"
                ) from exc
        if not bool(cache_stats.get("optimal_offline_cache_hit")):
            # Upstream _collect_optimal_offline_data: fixed-horizon per-env
            # rollouts; each collected batch keeps only complete episodes
            # (len==horizon, last done, exactly one done). Episode mode runs exactly
            # warmup_episodes rollouts; step-budget mode loops until the target
            # with the original 10-rollout zero-progress failure threshold.
            kept_steps = 0
            stagnation_rollouts = 0
            rollouts = 0
            episode_mode = warmup_collect_steps is None
            target_rollouts = int(resolved_config["warmup_episodes"])
            while True:
                if episode_mode and rollouts >= target_rollouts:
                    break
                if not episode_mode and kept_steps >= optimal_target_steps:
                    break
                batch, _ = _collect(_request("optimal", None, horizon_budget))
                with measure(performance, "cumulative_aggregation", group="replay_admission"):
                    kept = _admit(batch, "optimal", required_length=horizon)
                kept_steps += kept
                samples_seen += kept
                rollouts += 1
                stagnation_rollouts = stagnation_rollouts + 1 if kept <= 0 else 0
                if not episode_mode and stagnation_rollouts >= 10:
                    raise RuntimeError(
                        "official optimal offline collection made no progress for 10 "
                        "consecutive rollouts; the expert checkpoint may be too unstable "
                        "to generate full-horizon trajectories"
                    )
            if episode_mode and kept_steps <= 0:
                raise RuntimeError(
                    "official optimal offline collection kept no complete episodes"
                )
            cache_stats["offline_collected_steps"] = kept_steps
        if replay.suboptimal_enabled and not bool(
            cache_stats.get("suboptimal_offline_cache_hit")
        ):
            if not weak_checkpoints:
                raise RuntimeError(
                    "official suboptimal source is enabled but prepare selected no weak "
                    "checkpoints; refusing to fabricate suboptimal data"
                )
            # Upstream _collect_suboptimal_offline_data: target is
            # ceil(optimal_target_steps * suboptimal_data_ratio) raw transitions;
            # each request collects a full warmup_max_steps_per_episode per-env
            # horizon while cycling through the prepare-selected weak
            # checkpoints, so the last request may overshoot the target.
            suboptimal_target = int(
                math.ceil(
                    optimal_target_steps * float(resolved_config["suboptimal_data_ratio"])
                )
            )
            suboptimal_kept = 0
            weak_index = 0
            while suboptimal_kept < suboptimal_target:
                weak = weak_checkpoints[weak_index % len(weak_checkpoints)]
                weak_index += 1
                batch, _ = _collect(_request("suboptimal", weak, horizon_budget))
                with measure(performance, "cumulative_aggregation", group="replay_admission"):
                    suboptimal_kept += _admit(batch, "suboptimal")
            samples_seen += suboptimal_kept
        if not bool(cache_stats.get("offline_cache_hit")):
            with measure(performance, "checkpoint_save", group="offline_cache"):
                replay.save_offline_caches(
                    optimal_path=optimal_cache_path,
                    suboptimal_path=suboptimal_cache_path,
                )
        with measure(performance, "cumulative_aggregation", group="offline_split"):
            split_stats = replay.finalize_offline(
                val_ratio=float(resolved_config["val_trajectory_ratio"]), seed=seed
            )

        # ── Warmup (mode="offline"; original load/force precedence) ──
        warmup_ckpt_path = Path(str(resolved_config["warmup_ckpt_path"]))
        warmup_loaded = False
        if (
            not resume_path_value and warmup_ckpt_path.is_file()
            and bool(resolved_config.get("load_warmup_ckpt", True))
            and not bool(resolved_config.get("force_warmup_train", True))
        ):
            warmup_payload = load_official_fada_checkpoint(
                warmup_ckpt_path, learner, map_location=target_device
            )
            _check_loaded_source_identity(
                warmup_payload, expected_identity, context="warmup checkpoint"
            )
            if int(warmup_payload["completed_iterations"]) != 0:
                raise ValueError(
                    "official warmup checkpoint must be a pre-DAgger warmup save"
                )
            warmup_loaded = True
        if not resume_path_value and not warmup_loaded and int(resolved_config["warmup_train_steps"]) > 0:
            with measure(performance, "total_elapsed", group="warmup_train"):
                _train("offline", int(resolved_config["warmup_train_steps"]))
            with measure(performance, "checkpoint_save", group="warmup_model"):
                save_official_fada_checkpoint(
                    warmup_ckpt_path,
                    learner,
                    completed_iterations=0,
                    samples_seen=samples_seen,
                    source_identity=expected_identity,
                    rollout_step_count=rollout_step_count,
                )

        # Offline collection may use random initial student weights; online control
        # must see the learned/resumed weights before its very first rollout.
        with measure(performance, "checkpoint_save", group="post_warmup_model"):
            save_official_fada_checkpoint(
                checkpoint_path, learner, completed_iterations=completed_iterations,
                samples_seen=samples_seen, source_identity=expected_identity,
                rollout_step_count=rollout_step_count, online_buffer_path=online_target,
            )
        weight_version = _activate()
        if performance is not None:
            print("[FADA PERF] " + json.dumps(
                {"phase": "warmup_complete", **performance.summary()}, sort_keys=True
            ), flush=True)

        # ── DAgger rounds (mode="mixed") ──
        dagger_iters = int(resolved_config["dagger_iters"])
        train_steps_per_iter = int(resolved_config["train_steps_per_iter"])
        save_every_iter = int(resolved_config["save_every_iter"])
        if start_iteration >= dagger_iters:
            raise ValueError(
                f"official resume iteration {start_iteration} is already at or past "
                f"dagger_iters={dagger_iters}"
            )
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        run_token = f"{request_counter:04d}-{uuid.uuid4().hex}"
        writer = FADAIterationCommitter(
            checkpoint_path=checkpoint_path, checkpoints_dir=checkpoints_dir,
            run_token=run_token, save_every_iter=save_every_iter, dagger_iters=dagger_iters,
            save_online_buffer=bool(resolved_config.get("save_online_buffer", True)),
        )
        # ``writer`` is the non-optional local handle the DAgger closures use; the outer
        # ``committer`` slot exists only so the close barrier can drain it.
        committer = writer
        last_metrics: dict[str, float] = {}

        def _wait_commit() -> None:
            nonlocal completed_iterations, online_target
            committed = writer.wait()
            if committed is None:
                return
            completed_iterations = committed.completed_iterations
            online_target = committed.online_buffer_path
            if performance is not None:
                performance.add_worker_metrics(
                    {f"{name}_seconds": value for name, value in committed.wall_seconds.items()},
                    source="persistence",
                )
                performance.increment("online_iterations")
                print("[FADA PERF] " + json.dumps(
                    {"phase": "dagger", "iteration": completed_iterations,
                     "weight_version": weight_version, **performance.summary()}, sort_keys=True,
                ), flush=True)

        for it in range(start_iteration + 1, dagger_iters + 1):
            batch, collected_steps = _collect(_request("online", None, None))
            # Collection only fills its own bounded staging area. The previous
            # learner/Replay state remains read-only until its paired save commits.
            _wait_commit()
            samples_seen += int(batch.num_samples)
            with measure(performance, "cumulative_aggregation", group="replay_admission"):
                replay.process_online_batch(
                    batch,
                    sample_ratio=float(resolved_config["online_rollout_sample_ratio"]),
                    val_ratio=float(resolved_config["val_trajectory_ratio"]),
                    seed=seed, rollout_step_count=rollout_step_count,
                )
            rollout_step_count += int(collected_steps)
            with measure(performance, "total_elapsed", group="dagger_train"):
                last_metrics = _train("mixed", train_steps_per_iter)
            collection_iteration = it
            # Publish this round's latest weights before the next collection sees
            # them; the commit of the same round is waited for at the top of the loop.
            if it < dagger_iters:
                weight_version = _activate()
            writer.submit(
                learner, replay, completed_iterations=it, samples_seen=samples_seen,
                source_identity=expected_identity, rollout_step_count=rollout_step_count,
            )
        _wait_commit()

        outcome = {
            "mode": "official_fada_training",
            "execution_mode": "persistent_async",
            "shared_execution": True,
            "checkpoint_path": str(checkpoint_path),
            "completed_iterations": completed_iterations,
            "samples_seen": samples_seen,
            "update_count": learner.update_count,
            "rollout_step_count": rollout_step_count,
            "online_snapshot_token": run_token,
            "warmup_loaded_from_checkpoint": warmup_loaded,
            "offline_cache": dict(cache_stats),
            "offline_split": split_stats,
            "weak_checkpoints": list(weak_checkpoints),
            "replay_sizes": {
                source: replay.size(source)
                for source in ("optimal", "suboptimal", "online")
            },
            "last_update_metrics": last_metrics,
        }
        return outcome
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        with measure(performance, "cleanup"):
            if primary_error is not None:
                channel.abort()
            close_actions = [runtime.close, channel.cleanup]
            if committer is not None:
                close_actions.insert(0, committer.close)
            for cleanup in close_actions:
                try:
                    cleanup()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if primary_error is None and cleanup_errors:
            raise cleanup_errors[0]
        if performance is not None and outcome is not None:
            report = performance.summary(finished=True)
            report["scope"] = "training_owner_initialization_through_close"
            outcome["performance"] = report
            print("[FADA PERF] " + json.dumps(
                {"phase": "complete", **report}, sort_keys=True
            ), flush=True)
