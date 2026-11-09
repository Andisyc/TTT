from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from unilab.algos.torch.distill.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
)
from unilab.algos.torch.distill.persistent_runtime import (
    PersistentDistillationRuntime,
)
from unilab.ipc import SharedWeightSync


def _save_linear_checkpoint(path: Path, values: list[float], *, in_features: int = 2) -> Path:
    model = torch.nn.Linear(in_features, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(values, dtype=torch.float32).reshape(1, in_features))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def _load_linear_checkpoint(path: Path) -> torch.nn.Module:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    in_features = int(state_dict["weight"].shape[1])
    model = torch.nn.Linear(in_features, 1, bias=False)
    model.load_state_dict(state_dict)
    return model


class _ResidentLinearWorker:
    def __init__(
        self,
        *,
        weight_sync_name: str,
        weight_sync_lock,
        weight_param_shapes: dict,
    ) -> None:
        in_features = int(weight_param_shapes["weight"][1])
        self.model = torch.nn.Linear(in_features, 1, bias=False)
        self.weight_sync = SharedWeightSync(
            weight_param_shapes,
            create=False,
            shm_name=weight_sync_name,
            lock=weight_sync_lock,
        )

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        observed_version = self.weight_sync.read_weights_into(self.model.state_dict())
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(request.output_path).write_bytes(b"resident-linear")
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=observed_version,
            num_samples=1,
            worker_pid=os.getpid(),
            metrics={"weight_sum": float(self.model.weight.detach().sum().item())},
        )

    def close(self) -> None:
        self.weight_sync.close()


def _build_resident_linear_worker(**kwargs) -> _ResidentLinearWorker:
    return _ResidentLinearWorker(**kwargs)


def _request(
    tmp_path: Path,
    *,
    checkpoint: Path,
    version: int,
    request_id: str,
) -> DaggerCollectRequest:
    return DaggerCollectRequest(
        request_id=request_id,
        scenario="walk_flat",
        iteration=version,
        checkpoint_path=str(checkpoint.resolve()),
        output_path=str((tmp_path / f"{request_id}.pt").resolve()),
        expected_weight_version=version,
    )


def test_persistent_distillation_runtime_updates_one_resident_worker_via_shared_weights(
    tmp_path: Path,
) -> None:
    first_checkpoint = _save_linear_checkpoint(tmp_path / "student_1.pt", [1.0, 2.0])
    second_checkpoint = _save_linear_checkpoint(tmp_path / "student_2.pt", [4.0, 5.0])
    runtime = PersistentDistillationRuntime(
        student_loader=_load_linear_checkpoint,
        worker_factory=_build_resident_linear_worker,
    )
    try:
        first_version = runtime.activate_checkpoint(first_checkpoint)
        first = runtime.collect(
            _request(
                tmp_path,
                checkpoint=first_checkpoint,
                version=first_version,
                request_id="first",
            )
        )
        second_version = runtime.activate_checkpoint(second_checkpoint)
        second = runtime.collect(
            _request(
                tmp_path,
                checkpoint=second_checkpoint,
                version=second_version,
                request_id="second",
            )
        )
    finally:
        runtime.close()

    assert (first_version, second_version) == (1, 2)
    assert first.worker_pid == second.worker_pid
    assert first.worker_pid != os.getpid()
    assert first.metrics["weight_sum"] == pytest.approx(3.0)
    assert second.metrics["weight_sum"] == pytest.approx(9.0)


def test_persistent_distillation_runtime_rejects_checkpoint_shape_drift_before_publish(
    tmp_path: Path,
) -> None:
    first_checkpoint = _save_linear_checkpoint(tmp_path / "student_1.pt", [1.0, 2.0])
    incompatible_checkpoint = _save_linear_checkpoint(
        tmp_path / "student_bad.pt",
        [1.0, 2.0, 3.0],
        in_features=3,
    )
    runtime = PersistentDistillationRuntime(
        student_loader=_load_linear_checkpoint,
        worker_factory=_build_resident_linear_worker,
    )
    try:
        assert runtime.activate_checkpoint(first_checkpoint) == 1
        with pytest.raises(ValueError, match="checkpoint state shape mismatch"):
            runtime.activate_checkpoint(incompatible_checkpoint)
        assert runtime.weight_version == 1
    finally:
        runtime.close()


def test_closed_runtime_is_released_without_cyclic_gc(tmp_path: Path) -> None:
    import weakref

    runtime = PersistentDistillationRuntime(
        student_loader=_load_linear_checkpoint,
        worker_factory=_build_resident_linear_worker,
    )
    runtime.activate_checkpoint(
        tmp_path / "unwritten.pt", state_dict={"weight": torch.ones(1, 2)},
    )
    reference = weakref.ref(runtime)
    runtime.close()
    del runtime
    assert reference() is None
