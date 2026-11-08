"""Native Oracle DAgger composition over the existing collection and learning owners."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.artifact_admission import (
    _fada_quality_batches,
    _require_fada_curriculum_artifact,
)
from unilab.algos.torch.distill.fada.async_collection import FADA_ASYNC_SCENARIO
from unilab.algos.torch.distill.fada.checkpoint import save_fada_checkpoint
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.oracle_runtime import build_oracle_fada_runtime
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer
from unilab.algos.torch.distill.fada.source_artifact import (
    open_fada_source_artifact,
    validate_fada_async_artifact_identity,
)
from unilab.algos.torch.distill.fada.source_evaluation import evaluate_fada_source_batch
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.workflow_setup import (
    _distill_device,
    _fada_execution_mode,
    _fada_path,
    _fada_v005_replay_settings,
    _paper_source_plan,
    assert_fada_source_route_contract,
    build_fada_architecture_config,
)
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerPrepareRequest,
)


def run_oracle_fada_training(
    cfg: DictConfig, *, teacher_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Collect, admit, update IDM then Planner, and save a self-contained student."""
    fada = cfg.training.fada
    architecture = build_fada_architecture_config(cfg)
    assert_fada_source_route_contract(cfg, architecture)
    if _fada_execution_mode(cfg) != "persistent_async":
        raise ValueError("Oracle FADA requires execution_mode=persistent_async")
    if str(fada.training_schedule) != "alternating_idm_then_planner":
        raise ValueError("Oracle FADA requires alternating_idm_then_planner")
    if any(OmegaConf.select(fada, name) not in (None, "") for name in ("resume_path", "initial_weights_path")):
        raise ValueError("Oracle FADA starts fresh; training resume is not supported")
    counts = {name: int(fada[name]) for name in (
        "iterations", "windows_per_iteration", "num_envs", "batch_size", "replay_capacity",
        "idm_updates", "planner_updates", "save_every_iter",
    )}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Oracle FADA iteration, collection, replay and update counts must be positive")
    teacher = _fada_path(
        teacher_checkpoint if teacher_checkpoint is not None else cfg.teacher.checkpoint_path,
        field_name="teacher.checkpoint_path", required=True,
    )
    checkpoint = _fada_path(fada.checkpoint_path, field_name="training.fada.checkpoint_path", required=True)
    artifact_dir = _fada_path(fada.async_artifact_dir, field_name="training.fada.async_artifact_dir", required=True)
    assert teacher is not None and checkpoint is not None and artifact_dir is not None
    if checkpoint.exists():
        raise FileExistsError(f"Oracle FADA requires a fresh checkpoint target: {checkpoint}")
    plan = _paper_source_plan(cfg)
    v005_enabled, sampling = _fada_v005_replay_settings(fada, batch_size=counts["batch_size"])
    seed = int(cfg.algo.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _distill_device(cfg)
    policy = FADAPlannerIDMPolicy(architecture).to(device)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(policy.idm.parameters(), lr=float(fada.idm_learning_rate)),
        planner_optimizer=torch.optim.Adam(policy.planner.parameters(), lr=float(fada.planner_learning_rate)),
        max_grad_norm=float(fada.max_grad_norm),
    )
    replay = FADAReplayBuffer(
        architecture, capacity=counts["replay_capacity"],
        suboptimal_retention_ratio=2 if plan.enabled and v005_enabled else None,
    )
    runtime = build_oracle_fada_runtime(
        cfg=cfg, architecture=architecture, final_teacher_checkpoint=teacher,
        initial_student=policy, source_allocations=plan.source_allocations,
        scene_model_file=OmegaConf.select(fada, "scene_model_file"),
        request_timeout_seconds=float(fada.async_request_timeout_seconds),
    )
    generator = torch.Generator().manual_seed(seed)
    samples_seen = 0
    history: list[dict[str, Any]] = []
    try:
        prepared = runtime.prepare(DaggerPrepareRequest(request_id="oracle-source-profile"))
        source_profile = prepared.metadata["source_profile"]
        runtime_config = OmegaConf.to_container(fada, resolve=True)
        if not isinstance(runtime_config, dict):
            raise ValueError("training.fada must be a mapping")
        runtime_config["source_profile"] = source_profile
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for iteration in range(counts["iterations"]):
            version = runtime.activate_checkpoint(checkpoint, state_dict=policy.state_dict())
            request = DaggerCollectRequest(
                request_id=f"oracle-{iteration:04d}", scenario=FADA_ASYNC_SCENARIO,
                iteration=iteration, checkpoint_path=str(checkpoint.resolve()),
                output_path=str(artifact_dir / f"source_iter_{iteration:04d}.pt"),
                expected_weight_version=version,
            )
            result = runtime.collect(request)
            artifact = open_fada_source_artifact(result.output_path, config=architecture)
            validate_fada_async_artifact_identity(
                artifact.metadata, expected={**asdict(request), "producer_pid": result.worker_pid},
            )
            if artifact.metadata.get("source_profile") != source_profile:
                raise ValueError("Oracle source profile changed during collection")
            identity = artifact.identity_fields()
            _require_fada_curriculum_artifact(cfg, artifact.metadata, identity=identity)
            replay.add_artifact(artifact)
            samples_seen += artifact.num_samples
            stats = trainer.update_from_replay(
                replay, batch_size=counts["batch_size"], idm_updates=counts["idm_updates"],
                planner_updates=counts["planner_updates"], device=device, generator=generator,
                replay_sampling_spec=sampling if v005_enabled else None,
            )
            metrics = asdict(stats)
            if v005_enabled:
                quality = _fada_quality_batches(
                    artifact.iter_batches(), config=architecture,
                    limit=int(fada.quality_eval_max_windows), sampling_spec=sampling,
                )
                policy.eval()
                metrics.update(evaluate_fada_source_batch(policy, quality, require_scenario_metrics=True))
            completed = iteration + 1
            save_options: dict[str, Any] = dict(
                completed_iterations=completed, samples_seen=samples_seen,
                runtime_config=runtime_config, quality_metrics=metrics,
            )
            save_fada_checkpoint(checkpoint, policy, trainer, **save_options)
            if completed % counts["save_every_iter"] == 0 or completed == counts["iterations"]:
                save_fada_checkpoint(
                    checkpoint.parent / f"student_iter_{completed:04d}.pt", policy, trainer, **save_options,
                )
            row = {"iteration": completed, "samples_seen": samples_seen, "replay_size": len(replay), **metrics}
            history.append(row)
            print(f"[oracle-fada] iteration={completed}/{counts['iterations']} "
                  f"samples={samples_seen} idm_loss={stats.idm_loss:.6g} planner_loss={stats.planner_loss:.6g}", flush=True)
    finally:
        runtime.close()
    return {
        "source_route": "unilab_oracle", "checkpoint_path": str(checkpoint),
        "completed_iterations": counts["iterations"], "samples_seen": samples_seen,
        "source_profile": source_profile, "history": history,
    }
