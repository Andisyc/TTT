from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.algos.test_oracle_fada_source import recipe as recipe
from tests.algos.test_oracle_fada_source import source as source
from unilab.algos.torch.distill.fada import oracle_runtime, oracle_source
from unilab.algos.torch.distill.fada.async_collection import FADA_ASYNC_SCENARIO
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer
from unilab.algos.torch.distill.fada.source_artifact import open_fada_source_artifact
from unilab.algos.torch.distill.runtime.async_runtime import DaggerCollectRequest, DaggerPrepareRequest
from unilab.ipc import SharedWeightSync
from unilab.training.oracle.checkpoint import ENVIRONMENT_FIELDS


class _SourcePlant:
    """Deterministic terminal/reset and isolated-branch plant; inference stays real."""

    def __init__(self, control, num_envs):
        self.control = deepcopy(control)
        self.num_envs = num_envs
        self.action_space = SimpleNamespace(shape=(29,))
        self.obs_groups_spec = {"obs": 98, "critic": 303}
        self.state = None
        self.prepared = False
        self.closed = False
        self.fail_close = False
        self.physics_guard_trip_count = 0
        self.ticks = np.zeros(num_envs, dtype=np.int32)
        self.actions = np.zeros((num_envs, 29), dtype=np.float32)

    def get_motion_quality_deployment_profile(self):
        return deepcopy(self.control)

    def set_physics_envelope_guard(self, value):
        self.guard = value

    def prepare_isolated_rollout_branch(self):
        assert self.state is None
        self.prepared = True

    def _state(self):
        raw = np.zeros((self.num_envs, 98), dtype=np.float32)
        raw[:, 0] = self.ticks
        raw[:, 6:35] = self.actions
        raw[:, 64:93] = self.actions
        commands = np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (self.num_envs, 1))
        raw[:, 93:96] = commands
        raw[:, 96:] = self.ticks[:, None] * 0.01
        privilege = np.tile(np.linspace(0.1, 0.9, 205, dtype=np.float32), (self.num_envs, 1))
        return SimpleNamespace(
            obs={"obs": raw, "critic": np.concatenate((raw, privilege), axis=1)},
            info={"commands": commands, "critic_info": privilege},
            terminated=self.ticks >= 5, truncated=np.zeros(self.num_envs, dtype=np.bool_),
        )

    def init_state(self):
        assert self.prepared
        self.state = self._state()
        return self.state

    def reset(self, indices):
        self.ticks[indices] = 0
        self.actions[indices] = 0
        self.state = self._state()
        return ({key: value[indices] for key, value in self.state.obs.items()},
                {key: value[indices] for key, value in self.state.info.items()})

    def step(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32).copy()
        self.ticks += 1
        self.state = self._state()
        return self.state

    @contextmanager
    def isolated_rollout_branch(self):
        assert self.prepared
        ticks, actions, state = self.ticks.copy(), self.actions.copy(), self.state
        try:
            yield
        finally:
            self.ticks, self.actions, self.state = ticks, actions, state

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("cleanup failure")


@pytest.fixture
def collector(source, monkeypatch, tmp_path):
    path, payload, _ = source
    architecture = FADAArchitectureConfig(
        obs_dim=66, action_dim=29, command_dim=3, observation_contract="g1_fada_state_v2",
        history_length=2, prediction_horizon=2, hidden_dim=8, num_heads=2,
        planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=16,
    )
    with torch.random.fork_rng():
        torch.manual_seed(19)
        student = FADAPlannerIDMPolicy(architecture).eval()
    sync = SharedWeightSync({name: value.shape for name, value in student.state_dict().items()})
    sync.write_weights(student.state_dict())
    cfg = OmegaConf.create({"training": {
        "task_name": "G1WalkFlat", "sim_backend": "mujoco", "device": "cpu",
        "fada": {"num_envs": 2, "windows_per_iteration": 6,
                 "oracle_shadow_enabled": True, "max_env_steps": 40},
    }})
    envs = []
    control = {name: deepcopy(payload["metadata"][name]) for name in ENVIRONMENT_FIELDS}

    def create_env(cfg, *, num_envs, env_cfg_override):
        env = _SourcePlant(control, num_envs)
        envs.append(env)
        return env

    monkeypatch.setattr(oracle_source, "create_env", create_env)
    monkeypatch.setattr(oracle_runtime, "create_env", create_env)
    worker = None
    try:
        worker = oracle_runtime.OracleFADACollectorWorker(
            cfg_payload=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            architecture=asdict(architecture),
            final_teacher_checkpoint=str(path), source_allocations=(), device="cpu",
            weight_sync_name=sync.name, weight_sync_lock=sync._lock,
            weight_param_shapes={name: value.shape for name, value in student.state_dict().items()},
        )
        yield SimpleNamespace(worker=worker, sync=sync, student=student, config=architecture,
                              payload=payload, path=path, envs=envs, tmp_path=tmp_path)
    finally:
        if worker is not None:
            worker.close()
        sync.cleanup()


def _request(collector, name, iteration=0):
    return DaggerCollectRequest(
        request_id=name, scenario=FADA_ASYNC_SCENARIO, iteration=iteration,
        checkpoint_path=str(collector.tmp_path / "student.pt"),
        output_path=str(collector.tmp_path / f"{name}.pt"),
        expected_weight_version=collector.sync.version,
    )


def test_prepare_keeps_environment_uninitialized_and_worker_does_not_unlink_parent_weights(collector):
    result = collector.worker.prepare(DaggerPrepareRequest(request_id="prepare"))
    profile = json.loads(json.dumps(result.metadata["source_profile"]))
    assert profile["kind"] == "unilab_oracle"
    assert profile["control_contract"]["ctrl_dt"] == 0.02
    assert collector.envs[0].state is None
    np.testing.assert_array_equal(collector.envs[0].ticks, 0)
    collector.worker.close()
    assert collector.envs[0].closed
    assert collector.sync.read_weights_into(collector.student.state_dict()) == collector.sync.version


def test_oracle_windows_preserve_causal_futures_across_resets_and_reach_replay(collector):
    teacher = collector.worker.final_teacher
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    result = collector.worker.collect(_request(collector, "oracle"))
    artifact = open_fada_source_artifact(result.output_path, config=collector.config)
    assert artifact.metadata["source_profile"] == result.metadata["source_profile"]
    for batch in artifact.iter_batches():
        torch.testing.assert_close(batch.executed_action_chunk[:, 0], batch.oracle_first_action)
        expected = batch.observation_history[:, -1, 0, None] + torch.arange(1, 3)
        torch.testing.assert_close(batch.realized_future[:, :, 0], expected)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    replay = FADAReplayBuffer(collector.config, capacity=16)
    artifact.identity_fields()
    replay.add_artifact(artifact)
    sampled = replay.sample(4, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(sampled.executed_action_chunk[:, 0], sampled.oracle_first_action)
    assert all(env.closed for env in collector.envs)


def test_intermediate_reload_keeps_final_oracle_labels_and_real_privileged_inference(collector):
    zero_payload = deepcopy(collector.payload)
    zero_payload["metadata"]["iteration"] = 3
    zero_payload["state"]["update_count"] = 3
    for name, tensor in zero_payload["state"]["actor"].items():
        if "normalizer" not in name and name not in {"action_scale", "action_bias"}:
            zero_payload["state"]["actor"][name] = torch.zeros_like(tensor)
    zero_payload["state"]["actor"]["action_mean_head.bias"].fill_(0.25)
    zero_path = collector.tmp_path / "intermediate_constant.pt"
    torch.save(zero_payload, zero_path)
    collector.worker.source_allocations = ((str(collector.path), 2), (str(zero_path), 2))
    result = collector.worker.collect(_request(collector, "intermediates"))
    artifact = open_fada_source_artifact(result.output_path, config=collector.config)
    batches = list(artifact.iter_batches())
    last = batches[-1]
    torch.testing.assert_close(
        last.executed_action_chunk,
        torch.full_like(last.executed_action_chunk, float(np.tanh(0.25))),
    )
    assert not torch.allclose(last.oracle_first_action, last.executed_action_chunk[:, 0])
    assert bool(last.oracle_shadow_valid.all())
    assert collector.worker.intermediate_teacher.checkpoint_identity["iteration"] == 3
    assert all(env.closed for env in collector.envs)


def test_student_rollouts_consume_latest_published_weights(collector):
    first = collector.worker.collect(_request(collector, "student_a", iteration=1))
    first_batch = next(open_fada_source_artifact(first.output_path, config=collector.config).iter_batches())
    with torch.random.fork_rng():
        torch.manual_seed(31)
        changed = FADAPlannerIDMPolicy(collector.config).eval()
    collector.sync.write_weights(changed.state_dict())
    second = collector.worker.collect(_request(collector, "student_b", iteration=2))
    second_batch = next(open_fada_source_artifact(second.output_path, config=collector.config).iter_batches())
    assert second.observed_weight_version == second.expected_weight_version == collector.sync.version
    assert second.observed_weight_version > first.observed_weight_version
    assert not torch.allclose(first_batch.executed_action_chunk, second_batch.executed_action_chunk)
    assert not torch.allclose(second_batch.executed_action_chunk[:, 0], second_batch.oracle_first_action)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_invalid_intermediate_contract_closes_uninitialized_environment_without_masking_error(
    collector, cleanup_fails,
):
    bad = deepcopy(collector.payload)
    bad["metadata"]["ctrl_dt"] = 0.01
    bad_path = collector.tmp_path / "invalid.pt"
    torch.save(bad, bad_path)
    env = collector.envs[0]
    env.fail_close = cleanup_fails
    with pytest.raises(ValueError, match="ctrl_dt"):
        with collector.worker.collection_environment():
            collector.worker._intermediate_teacher_loader(str(bad_path), None, device="cpu")
    assert env.closed and env.state is None
    assert collector.worker.env is None
