"""Request-scoped native Oracle collection using the existing FADA window pipeline."""

from __future__ import annotations

import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.async_collection import collect_fada_iteration
from unilab.algos.torch.distill.fada.async_config import fada_runtime_device
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.collection_contract import FADACollectionSpec
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import FADA_G1_STATE_OBSERVATION_CONTRACT
from unilab.algos.torch.distill.fada.oracle_source import build_oracle_fada_source
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    DaggerPrepareRequest,
    DaggerPrepareResult,
)
from unilab.algos.torch.distill.runtime.persistent_runtime import PersistentDistillationRuntime
from unilab.envs.locomotion.g1.motion_quality_config import build_motion_quality_reward_config
from unilab.ipc import SharedWeightSync
from unilab.training import create_env
from unilab.training.oracle.checkpoint import read_oracle_checkpoint, validate_environment_contract
from unilab.training.oracle.teacher import MotionQualityOracleTeacher


class OracleFADACollectorWorker:
    """Own native pools per source and keep frozen policies within one request."""

    def __init__(
        self,
        *,
        cfg_payload: Mapping[str, Any],
        architecture: Mapping[str, Any],
        final_teacher_checkpoint: str,
        source_allocations: Sequence[tuple[str, int]],
        device: str,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: Mapping[str, torch.Size],
        scene_model_file: str | None = None,
    ) -> None:
        self.cfg = OmegaConf.create(dict(cfg_payload))
        self.config = FADAArchitectureConfig(**dict(architecture))
        if self.config.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
            raise ValueError("Oracle FADA collection requires g1_fada_state_v2")
        if (self.cfg.training.task_name, self.cfg.training.sim_backend) != ("G1WalkFlat", "mujoco"):
            raise ValueError("Oracle FADA collection requires the G1WalkFlat/mujoco owner")
        if not self.cfg.training.fada.oracle_shadow_enabled:
            raise ValueError("Oracle FADA collection requires oracle_shadow_enabled=true")
        self.device = str(device)
        self.source_allocations = tuple((str(path), int(count)) for path, count in source_allocations)
        self.teacher_spec = None  # The shared collector passes this opaque argument to loaders.
        self.intermediate_teacher: torch.nn.Module | None = None
        self.intermediate_teacher_checkpoint: str | None = None
        self._intermediate_network_config: dict[str, Any] | None = None
        self.weight_sync: SharedWeightSync | None = None
        self.env: Any | None = None
        self.final_teacher: torch.nn.Module
        self._environment_in_use = False
        self._closed = False
        self._retired_physics_guard_trips = 0
        self._physics_guard_max_abs = float(
            OmegaConf.select(self.cfg, "training.fada.physics_guard_max_abs", default=1.0e4)
        )
        try:
            self.student = FADAPlannerIDMPolicy(self.config).to(self.device).eval()
            self.student.requires_grad_(False)
            self.weight_sync = SharedWeightSync(
                dict(weight_param_shapes), create=False, shm_name=weight_sync_name, lock=weight_sync_lock,
            )
            self.local_weight_version = self.weight_sync.read_weights_into(self.student.state_dict())
            self.env, self.final_teacher, self.source_profile = build_oracle_fada_source(
                final_teacher_checkpoint,
                num_envs=int(self.cfg.training.fada.num_envs),
                device=self.device,
                scene_model_file=scene_model_file,
            )
            if self.final_teacher.priv_info_dim != 205:
                raise ValueError("Oracle FADA collection requires 205 privileged features")
            self._prepare_environment(self.env)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _prepare_environment(self, env: Any) -> None:
        validate_environment_contract(
            self.source_profile["control_contract"], env.get_motion_quality_deployment_profile(),
        )
        env.set_physics_envelope_guard(self._physics_guard_max_abs)
        env.prepare_isolated_rollout_branch()

    def _materialize_collection_environment(self) -> Any:
        # Recreate a native pool, not its already-resident teacher or checkpoint payload.
        environment_config = deepcopy(self.source_profile["environment_config"])
        environment_config["reward_config"] = build_motion_quality_reward_config(
            environment_config["reward_config"]
        )
        env = create_env(
            self.cfg, num_envs=int(self.cfg.training.fada.num_envs),
            env_cfg_override=environment_config,
        )
        try:
            self._prepare_environment(env)
            return env
        except BaseException as error:
            try:
                env.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _retire_environment(self) -> None:
        env, self.env = self.env, None
        if env is not None:
            self._retired_physics_guard_trips += int(env.physics_guard_trip_count)
            env.close()

    @contextmanager
    def collection_environment(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("Oracle collector is closed")
        if self._environment_in_use:
            raise RuntimeError("Oracle collection environments cannot be nested")
        if self.env is None:
            self.env = self._materialize_collection_environment()
        self._environment_in_use = True
        try:
            yield self.env
        except BaseException as error:
            try:
                self._retire_environment()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        else:
            self._retire_environment()
        finally:
            self._environment_in_use = False

    @property
    def physics_guard_trip_count(self) -> int:
        active = 0 if self.env is None else int(self.env.physics_guard_trip_count)
        return self._retired_physics_guard_trips + active

    def _collection_spec(self) -> FADACollectionSpec:
        return FADACollectionSpec(
            observation_key="obs",
            teacher_projection="identity",
            student_projection=FADA_G1_STATE_OBSERVATION_CONTRACT,
            command_info_keys=("commands",),
            max_env_steps=OmegaConf.select(self.cfg, "training.fada.max_env_steps"),
            collect_oracle_shadow=True,
        )

    def _intermediate_teacher_loader(
        self, path: str, _spec: Any, *, device: str,
    ) -> MotionQualityOracleTeacher:
        if self.env is None:
            raise RuntimeError("Oracle source environment must exist before loading a teacher")
        payload = read_oracle_checkpoint(path)
        teacher = MotionQualityOracleTeacher(payload, self.env, device)
        self._intermediate_network_config = deepcopy(payload["metadata"]["network_config"])
        return teacher

    def _intermediate_teacher_reloader(
        self, teacher: torch.nn.Module, path: str, _spec: Any,
    ) -> None:
        if self.env is None:
            raise RuntimeError("Oracle source environment must exist before reloading a teacher")
        payload = read_oracle_checkpoint(path)
        metadata = payload["metadata"]
        validate_environment_contract(metadata, self.env.get_motion_quality_deployment_profile())
        if metadata["network_config"] != self._intermediate_network_config:
            raise ValueError("Intermediate Oracle network configuration differs")
        if not isinstance(teacher, MotionQualityOracleTeacher):
            raise TypeError("Intermediate teacher must be a native Oracle")
        teacher.actor.load_state_dict(payload["state"]["actor"], strict=True)
        teacher.normalizer.load_state_dict(payload["state"]["obs_normalizer"], strict=True)
        teacher.checkpoint_identity = {
            "format": payload["format"], "identity": metadata["identity"],
            "iteration": metadata["iteration"],
        }
        teacher.eval()
        teacher.requires_grad_(False)

    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult:
        if self._closed:
            raise RuntimeError("Oracle collector is closed")
        return DaggerPrepareResult(
            request_id=request.request_id, worker_pid=os.getpid(),
            metadata={"source_profile": deepcopy(self.source_profile)},
        )

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._closed:
            raise RuntimeError("Oracle collector is closed")
        return collect_fada_iteration(self, request, source_profile=self.source_profile)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        try:
            self._retire_environment()
        except BaseException as error:
            errors.append(error)
        self.intermediate_teacher = None
        if self.weight_sync is not None:
            try:
                self.weight_sync.close()
            except BaseException as error:
                errors.append(error)
            self.weight_sync = None
        if errors:
            raise errors[0] from (errors[1] if len(errors) > 1 else None)


class OracleFADARuntime(PersistentDistillationRuntime):
    """Use one short-lived preparation process and request-scoped collection."""

    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult:
        if self._closed:
            raise RuntimeError("cannot prepare a closed Oracle FADA runtime")
        preparation = PersistentDistillationRuntime(
            student_loader=self._student_loader,
            worker_factory=self._worker_factory,
            worker_kwargs=self._base_worker_kwargs,
            initial_student=self._initial_student,
            request_timeout_seconds=self._request_timeout_seconds,
            worker_lifecycle="persistent",
        )
        try:
            return preparation.prepare(request)
        finally:
            preparation.close()


def build_oracle_fada_runtime(
    *,
    cfg: DictConfig,
    architecture: FADAArchitectureConfig,
    final_teacher_checkpoint: str | Path,
    initial_student: FADAPlannerIDMPolicy,
    source_allocations: Sequence[tuple[str | Path, int]] = (),
    scene_model_file: str | Path | None = None,
    request_timeout_seconds: float = 300.0,
) -> PersistentDistillationRuntime:
    """Build the existing prepare/collect service without changing its IPC protocol.

    ``prepare()`` returns the source profile without initialization or stepping.
    Publish learner weights with ``activate_checkpoint(..., state_dict=...)``;
    the checkpoint path is the weight-barrier identity, not a required bootstrap file.
    """
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_payload, dict):
        raise ValueError("Oracle FADA config must resolve to a mapping")
    return OracleFADARuntime(
        student_loader=lambda path: load_fada_policy_checkpoint(path, device="cpu").policy,
        initial_student=initial_student,
        worker_factory=OracleFADACollectorWorker,
        worker_kwargs={
            "cfg_payload": cfg_payload,
            "architecture": asdict(architecture),
            "final_teacher_checkpoint": str(Path(final_teacher_checkpoint).expanduser().resolve()),
            "source_allocations": [
                (str(Path(path).expanduser().resolve()), int(count))
                for path, count in source_allocations
            ],
            "device": fada_runtime_device(cfg),
            "scene_model_file": (
                None if scene_model_file is None else str(Path(scene_model_file).expanduser().resolve())
            ),
        },
        request_timeout_seconds=float(request_timeout_seconds),
        worker_lifecycle="request",
    )
