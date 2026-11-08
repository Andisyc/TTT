"""Single composition root for official FADA source training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.oracle_workflow import run_oracle_fada_training
from unilab.algos.torch.distill.fada.persistent_workflow import run_fada_persistent_async
from unilab.algos.torch.distill.fada.workflow_setup import (
    FADAWorkflowDependencies,
    resolve_official_fada_execution,
    resolve_official_fada_setup,
)


def run_fada_training_owner(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path | None = None,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Dispatch explicitly separated official and native Oracle source routes."""

    if OmegaConf.select(cfg, "training.fada.source_route") == "unilab_oracle":
        return run_oracle_fada_training(cfg, teacher_checkpoint=teacher_checkpoint)

    spec, model_config, device, output_dir, timeout = resolve_official_fada_setup(
        cfg, teacher_checkpoint=teacher_checkpoint
    )
    performance_enabled = OmegaConf.select(
        cfg, "training.fada.performance.enabled", default=False
    )
    if not isinstance(performance_enabled, bool):
        raise ValueError("training.fada.performance.enabled must be a boolean")
    execution_limits, pending_raw_budget_bytes = resolve_official_fada_execution(cfg, spec=spec)
    return run_fada_persistent_async(
        spec=spec,
        model_config=model_config,
        device=device,
        output_dir=output_dir,
        request_timeout_seconds=timeout,
        runtime_factory=dependencies.build_official_fada_runtime,
        performance_enabled=performance_enabled,
        execution_limits=execution_limits,
        pending_raw_budget_bytes=pending_raw_budget_bytes,
    )
