"""Raw data from real NpEnv lifecycle and real LoRA updates, no simulator/GUI."""
from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.base.base import EnvCfg
from unilab.base.backend.base import SimBackend
from unilab.base.np_env import NpEnv
from unilab.algos.torch.distill.fada.checkpoint import LoadedFADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.adaptation import fada_adapter_parameters
from unilab.visualization.experiment_recording import (
    ExperimentRecorder, ExperimentRecordingConfig, experiment_recording_run,
)
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import PlaybackControls, RslRlPlaybackConfig


@dataclass
class _Config(EnvCfg):
    ctrl_dt: float = 0.02
    sim_dt: float = 0.01
    max_episode_seconds: float | None = 10.0
    control_config: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(action_scale=0.5))
    runtime_payload: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(
        enabled=True, body_name="test_torso", attachment_pos_m=[0.0, 0.2, 0.1],
        half_size_m=[0.1, 0.1, 0.1], added_mass_kg=0.7,
    ))


class _Backend:
    def __init__(self):
        self.count = 0
        self.velocity = np.full((1, 3), -9.0)

    def get_scene_artifacts(self):
        return SimpleNamespace(model_file=None)

    def step(self, actions, substeps):
        self.count += 1
        self.velocity[:] = [self.count, 2.0, 3.0]

    def get_base_ang_vel(self):
        return np.array([[0.0, 0.0, 0.1 * self.count]])

    def get_actuated_joint_names(self):
        return tuple(f"joint_{i}" for i in range(29))

    def close(self):
        pass


class _Env(NpEnv):
    def __init__(self):
        self.test_backend = _Backend()
        super().__init__(_Config(), cast(SimBackend, self.test_backend), 1)
        self.terminate_next = False
        self.closed = False

    @property
    def obs_groups_spec(self):
        return {"actor": 98}

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (29,), dtype=np.float32)

    def get_base_quat(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]])

    def get_base_lin_vel(self):
        return self.test_backend.velocity

    def apply_action(self, actions, state):
        state.info["current_actions"] = actions + 0.25
        state.info["runtime_payload_attached_this_step"][:] = self.step_counter == 2
        state.info["runtime_payload_added_mass_kg"][:] = 0.7 if self.step_counter >= 2 else 0.0
        return actions

    def update_state(self, state):
        # Publishing a next command here deliberately distinguishes it from
        # the command read by the action that just completed.
        state.info["commands"] = np.array([[0.8, 0.0, 0.1]], dtype=np.float32)
        terminal = self.terminate_next
        self.terminate_next = False
        return state.replace(
            obs={"actor": np.full((1, 98), self.test_backend.count * 0.1, dtype=np.float32)},
            reward=np.zeros(1), terminated=np.array([terminal]), truncated=np.zeros(1, dtype=bool),
        )

    def reset(self, env_indices):
        self.test_backend.velocity.fill(-9.0)
        return {"actor": np.zeros((len(env_indices), 98), dtype=np.float32)}, {
            "commands": np.array([[0.4, 0.0, 0.0]], dtype=np.float32),
            "current_actions": np.zeros((1, 29)),
            "runtime_payload_attached_this_step": np.zeros(1, dtype=bool),
            "runtime_payload_added_mass_kg": np.zeros(1),
        }

    def close(self):
        self.closed = True
        super().close()


class _Wrapper:
    def __init__(self, env):
        self.env = env
        self.num_obs = 98

    def get_observations(self):
        return {"actor": torch.from_numpy(self.env.state.obs["actor"])}

    def reset(self):
        self.env.reset_all()
        return self.get_observations(), {}

    def step(self, actions):
        state = self.env.step(actions.detach().cpu().numpy())
        done = torch.from_numpy((state.terminated | state.truncated).copy())
        return self.get_observations(), torch.zeros(1), done, state.info

    def close(self):
        self.env.close()


@pytest.fixture
def make_session(tmp_path):
    def make(name="run", *, recording=True, mode="trajectory_base", chunk_steps=3):
        with torch.random.fork_rng():
            torch.manual_seed(2)
            policy = FADAPlannerIDMPolicy(FADAArchitectureConfig(
                obs_dim=66, action_dim=29, command_dim=3, observation_contract="g1_fada_state_v2",
                history_length=3, prediction_horizon=2, hidden_dim=8, num_heads=2,
                planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=16, dropout=0.0,
            )).eval()
        env = _Env()
        wrapper = _Wrapper(env)
        checkpoint = tmp_path / "checkpoint.pt"
        torch.save(policy.state_dict(), checkpoint)
        cfg = OmegaConf.create({
            "training": {"task_name": "Test"}, "algo": {"seed": 2},
            "online_adaptation": {"enabled": True, "mode": mode, "sigma": 0.0,
                                  "window_size": 2, "warmup_steps": 0, "update_interval": 2,
                                  "lora_rank": 2, "lora_alpha": 4.0, "lora_dropout": 0.0},
            "experiment_recording": {"enabled": recording, "output_dir": str(tmp_path / name), "chunk_steps": chunk_steps},
        })
        with torch.random.fork_rng():
            torch.manual_seed(10)
            session, _, _ = create_fada_playback_session(
                playback_cfg=RslRlPlaybackConfig(
                    task="Test", load_run="-1", checkpoint=None, checkpoint_path=str(checkpoint),
                    action_mode="policy", policy_obs_mode="actor", algo_log_name="distill", log_root=None,
                ), cfg=cfg, root_dir=Path(__file__).resolve().parents[2], device="cpu", log=lambda _: None,
                deps={"resolve_checkpoint": lambda *a: checkpoint,
                      "load_fada_policy": lambda *a, **k: LoadedFADAPlannerIDMPolicy(policy, {"completed_iterations": 0}),
                      "create_env": lambda *a, **k: env, "wrapper_cls": lambda *a, **k: wrapper,
                      "build_env_cfg_override": lambda cfg: {}},
            )
        session.reset()
        return session, tmp_path / name
    return make


def _read(path):
    blocks = []
    for chunk in sorted((path / "measurements").glob("*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            blocks.append({name: data[name].copy() for name in data.files})
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    return json.loads((path / "run_config.json").read_text()), blocks, events


@pytest.mark.parametrize("mode", ["trajectory_base", "self_supervised_correction", "trajectory_lora", "fada_ttt"])
def test_recorder_real_updates_prereset_truth_and_payload_timestamp(make_session, mode):
    session, path = make_session(mode=mode)
    before = [value.detach().clone() for value in fada_adapter_parameters(session.controller.policy)]
    with experiment_recording_run(session):
        controls = PlaybackControls(paused=True)
        assert not session.advance(controls)
        for _ in range(8):
            session.step_once()
        session.env.terminate_next = True
        session.step_once()
        # The actual env has already reset, but the old terminal state must be on disk.
        np.testing.assert_array_equal(session.env.get_base_lin_vel(), [[-9.0, -9.0, -9.0]])
        session.step_once()
    metadata, blocks, events = _read(path)
    assert metadata["status"] == "completed" and metadata["recorded_control_steps"] == 10
    assert metadata["completed_updates"] >= 1
    assert len(metadata["checkpoint_sha256"]) == 64
    assert metadata["resolved_config"]["online_adaptation"]["mode"] == mode
    assert metadata["formal_protocol_approved"] is False
    assert any(not torch.equal(a, b) for a, b in zip(before, fada_adapter_parameters(session.controller.policy)))
    commands = np.concatenate([block["velocity_command_body"] for block in blocks])
    np.testing.assert_allclose(commands[0], [0.4, 0, 0])
    np.testing.assert_allclose(commands[1], [0.8, 0, 0.1])
    actual = np.concatenate([block["linear_velocity_world_after"] for block in blocks])
    np.testing.assert_allclose(actual[8], [9.0, 2.0, 3.0])
    episode = np.concatenate([block["episode_id"] for block in blocks])
    np.testing.assert_array_equal(episode, [0] * 9 + [1])
    terminal = np.concatenate([block["terminated"] for block in blocks])
    np.testing.assert_array_equal(terminal, [False] * 8 + [True, False])
    post_before = np.concatenate([block["linear_velocity_world_before"] for block in blocks])
    np.testing.assert_allclose(post_before[9], [-9, -9, -9])
    submitted = np.concatenate([block["submitted_environment_action"] for block in blocks])
    env_actions = np.concatenate([block["environment_action"] for block in blocks])
    np.testing.assert_allclose(env_actions, submitted + 0.25, atol=1e-6)
    payload, = [event for event in events if event["kind"] == "payload_applied"]
    assert payload["sim_time_s"] == pytest.approx(0.04)
    assert payload["phase"] == "before_physics" and payload["added_mass_kg"] == 0.7
    assert any(event["kind"] == "reset" and event["reason"] == "autoreset" for event in events)
    assert session.env._step_measurement_callback is None


@pytest.mark.parametrize("failure", ["update", "keyboard", "sigterm"])
def test_exception_and_interrupt_keep_completed_steps(make_session, monkeypatch, failure):
    session, path = make_session(chunk_steps=100)
    previous_handler = signal.getsignal(signal.SIGTERM)
    def fail_update(samples):
        raise RuntimeError("synthetic optimizer failure")
    error_type = RuntimeError if failure == "update" else KeyboardInterrupt
    with pytest.raises(error_type):
        with experiment_recording_run(session):
            if failure == "update":
                monkeypatch.setattr(session.controller.updater, "update", fail_update)
                for _ in range(8):
                    session.step_once()
            else:
                session.step_once()
                if failure == "keyboard":
                    raise KeyboardInterrupt("synthetic Ctrl-C")
                os.kill(os.getpid(), signal.SIGTERM)
    metadata, blocks, events = _read(path)
    expected_status = "failed" if failure == "update" else "interrupted"
    assert metadata["status"] == expected_status
    assert metadata["recorded_control_steps"] == sum(len(block["control_step"]) for block in blocks)
    assert metadata["recorded_control_steps"] >= 1
    assert events[-1]["kind"] == expected_status
    assert metadata["completed_updates"] == 0
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert session.env._step_measurement_callback is None


def test_disabled_recording_preserves_actions_and_learning(make_session):
    recorded, path = make_session("recorded")
    plain, unused = make_session("disabled", recording=False)
    with experiment_recording_run(recorded), experiment_recording_run(plain):
        for _ in range(8):
            recorded.step_once()
            plain.step_once()
            torch.testing.assert_close(recorded.actions, plain.actions, rtol=0, atol=0)
        for name, parameter in recorded.controller.policy.named_parameters():
            torch.testing.assert_close(parameter, dict(plain.controller.policy.named_parameters())[name], rtol=0, atol=0)
    assert not unused.exists()
    assert _read(path)[0]["status"] == "completed"


def test_existing_output_is_not_overwritten(make_session):
    session, path = make_session()
    with experiment_recording_run(session):
        session.step_once()
    original = (path / "run_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        ExperimentRecorder(session.env, ExperimentRecordingConfig(True, str(path)), {
            "policy_joint_names": tuple(f"joint_{i}" for i in range(29)),
            "environment_joint_names": tuple(f"joint_{i}" for i in range(29)),
        })
    assert (path / "run_config.json").read_bytes() == original


def test_storage_failure_is_exposed_and_keeps_committed_chunk(make_session, monkeypatch):
    session, path = make_session(chunk_steps=1)
    with pytest.raises(OSError, match="synthetic storage full"):
        with experiment_recording_run(session):
            session.step_once()
            def fail(*args, **kwargs):
                raise OSError("synthetic storage full")
            monkeypatch.setattr(np, "savez", fail)
            session.step_once()
    with np.load(path / "measurements" / "part_000000.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["linear_velocity_world_after"], [[1.0, 2.0, 3.0]])
    assert json.loads((path / "run_config.json").read_text())["status"] != "completed"
    assert session.env._step_measurement_callback is None
