"""Small real-model trials through the shared Session, with controlled physics."""
from __future__ import annotations

import json
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.adaptation import fada_adapter_parameters
from unilab.algos.torch.distill.fada.checkpoint import LoadedFADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.base.backend.base import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv
from unilab.visualization import experiment_execution as execution

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Config(EnvCfg):
    ctrl_dt: float = 0.02
    sim_dt: float = 0.01
    max_episode_seconds: float | None = 10.0
    commands: SimpleNamespace = field(default_factory=SimpleNamespace)
    runtime_payload: SimpleNamespace = field(default_factory=SimpleNamespace)


class _Backend:
    def __init__(self):
        self.count = 0
        self.velocity = np.zeros((1, 3))

    def get_scene_artifacts(self):
        return SimpleNamespace(model_file=None)

    def get_actuated_joint_names(self):
        return tuple(f"joint_{i}" for i in range(29))

    def get_base_ang_vel(self):
        return np.zeros((1, 3))

    def step(self, actions, substeps):
        self.count += 1
        self.velocity[:] = [0.3 + float(actions.mean()) * 0.01, 0.01, 0.0]

    def close(self):
        pass


class _Env(NpEnv):
    def __init__(self, cfg, *, end=None, interrupt=None):
        self.physics = _Backend()
        self.end = end
        self.interrupt = interrupt
        self.reset_count = 0
        self.closed = False
        self.payload_active = False
        commands = OmegaConf.to_container(cfg.env.commands, resolve=True)
        payload = OmegaConf.to_container(cfg.env.runtime_payload, resolve=True)
        assert isinstance(commands, dict) and isinstance(payload, dict)
        self.test_cfg = _Config(
            commands=SimpleNamespace(**{str(key): value for key, value in commands.items()}),
            runtime_payload=SimpleNamespace(**{str(key): value for key, value in payload.items()}),
        )
        super().__init__(self.test_cfg, cast(SimBackend, self.physics), 1)

    @property
    def obs_groups_spec(self):
        return {"actor": 98}

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (29,), dtype=np.float32)

    def get_base_quat(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]])

    def get_base_lin_vel(self):
        return self.physics.velocity

    def _obs(self, command):
        value = np.full((1, 98), self.physics.count * 0.03, dtype=np.float32)
        value[:, :3] = command
        return {"actor": value}

    def refresh_state(self):
        state = self.state
        assert state is not None
        self._state = state.replace(obs=self._obs(state.info["commands"]))
        return self._state

    def apply_action(self, actions, state):
        if self.interrupt == self.step_counter:
            raise KeyboardInterrupt("controlled interruption")
        state.info["current_actions"] = actions.copy()
        payload = self.test_cfg.runtime_payload
        attached = bool(payload.enabled and not self.payload_active and
                        self.step_counter * self.cfg.ctrl_dt >= payload.attach_time_s)
        self.payload_active |= attached
        state.info["runtime_payload_attached_this_step"][:] = attached
        state.info["runtime_payload_added_mass_kg"][:] = payload.added_mass_kg if self.payload_active else 0.0
        return actions

    def update_state(self, state):
        ended = self.physics.count == 5
        return state.replace(obs=self._obs(state.info["commands"]), reward=np.zeros(1),
                             terminated=np.array([ended and self.end == "terminated"]),
                             truncated=np.array([ended and self.end == "truncated"]))

    def reset(self, env_indices):
        self.reset_count += 1
        command = np.array([self.test_cfg.commands.vel_limit[0]], dtype=np.float32)
        return self._obs(command), {"commands": command, "current_actions": np.zeros((1, 29)),
                                    "runtime_payload_attached_this_step": np.zeros(1, dtype=bool),
                                    "runtime_payload_added_mass_kg": np.zeros(1)}

    def close(self):
        self.closed = True
        super().close()


class _Wrapper:
    def __init__(self, env, **kwargs):
        self.env = env
        self.num_obs = 98

    def get_observations(self):
        return {"actor": torch.from_numpy(self.env.state.obs["actor"])}

    def reset(self):
        self.env.reset_all()
        return self.get_observations(), {}

    def step(self, actions):
        state = self.env.step(actions.detach().cpu().numpy())
        return self.get_observations(), torch.zeros(1), torch.from_numpy((state.terminated | state.truncated).copy()), state.info

    def close(self):
        self.env.close()


@pytest.fixture
def trial(tmp_path, monkeypatch):
    from unilab.algos.torch.distill.fada import online_controller
    original = online_controller.OnlineAdaptationController
    holder = {}

    class ObservedController(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["controller"] = self
            holder["initial_lora"] = [p.detach().clone() for p in fada_adapter_parameters(self.policy)]

    monkeypatch.setattr(online_controller, "OnlineAdaptationController", ObservedController)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        policy = FADAPlannerIDMPolicy(FADAArchitectureConfig(
            obs_dim=66, action_dim=29, command_dim=3, observation_contract="g1_fada_state_v2",
            history_length=3, prediction_horizon=2, hidden_dim=8, num_heads=2,
            planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=16, dropout=0.0,
        )).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(policy.state_dict(), checkpoint)
    options = {
        "group": "full_oat", "task_owner": "g1_walk_flat/mujoco", "checkpoint_path": str(checkpoint),
        "output_dir": str(tmp_path / "trial"), "seed": 7, "command": [0.3, 0.0, 0.0],
        "duration_s": 0.2, "env_overrides": {}, "device": "cpu", "chunk_steps": 3,
        "payload": {"enabled": True, "body_name": "torso_link", "added_mass_kg": 0.7,
                    "attachment_pos_m": [0.0, 0.2, 0.1], "half_size_m": [0.1] * 3, "attach_time_s": 0.04},
        "online": {"window_size": 2, "warmup_steps": 0, "update_interval": 2, "gamma": 0.05,
                   "sigma": 0.0001, "learning_rate": 0.0001, "max_grad_norm": 1.0,
                   "lora_rank": 2, "lora_alpha": 4.0, "lora_dropout": 0.0},
        "evaluation": {"enabled": True, "linear_frame": "body_xy", "sample_phase": "completed",
                       "relative_to": "run_start", "start_s": 0.0, "end_s": 0.2},
    }

    def create_env(cfg, **kwargs):
        env = _Env(cfg, end=holder.get("end"), interrupt=holder.get("interrupt"))
        holder["env"] = env
        return env

    deps = {"resolve_checkpoint": lambda *args: checkpoint,
            "load_fada_policy": lambda *args, **kwargs: LoadedFADAPlannerIDMPolicy(policy, {"completed_iterations": 0}),
            "create_env": create_env, "wrapper_cls": _Wrapper,
            "build_env_cfg_override": lambda cfg: OmegaConf.to_container(cfg.env)}
    return options, deps, holder


def _read(output):
    raw = Path(output) / "raw"
    metadata = json.loads((raw / "run_config.json").read_text())
    events = [json.loads(line) for line in (raw / "events.jsonl").read_text().splitlines()]
    return metadata, events


@pytest.mark.parametrize(("group", "mode", "anchored"), [
    ("fada_ttt", "fada_ttt", False), ("online_reference", "trajectory_lora", True),
    ("self_supervised_correction", "self_supervised_correction", True), ("no_anchor", "trajectory_base", False),
    ("full_oat", "trajectory_base", True),
])
def test_five_real_updates_record_and_evaluate(trial, group, mode, anchored):
    options, deps, holder = trial
    options["group"] = group
    messages = []
    result = execution.run_oat_experiment(options, root_dir=ROOT, deps=deps, log=messages.append)
    assert result["executed_steps"] == 10
    assert result["stop_reason"] == "duration_reached"
    assert result["completed_updates"] >= 1
    assert holder["controller"].mode == mode
    assert any(not torch.equal(before, after) for before, after in
               zip(holder["initial_lora"], fada_adapter_parameters(holder["controller"].policy)))
    metadata, events = _read(options["output_dir"])
    actual = metadata["resolved_config"]["online_adaptation"]
    assert actual["sigma"] == (0.0001 if anchored else 0.0)
    assert metadata["recorded_control_steps"] == 10
    assert len([e for e in events if e["kind"] == "reset"]) == 1
    assert [e["sim_time_s"] for e in events if e["kind"] == "payload_applied"] == [0.04]
    summary = json.loads((Path(options["output_dir"]) / "evaluation" / "summary.json").read_text())
    assert summary["episodes"][0]["full_requested_window"] is True
    assert summary["episodes"][0]["velocity_observed_segment"]["E_lin_m_s"] > 0
    assert holder["env"].closed and holder["env"].reset_count == 1
    assert any("step=10/10" in message for message in messages)


@pytest.mark.parametrize("reason", ["terminated", "truncated"])
def test_stops_on_first_done_without_reset_or_filling_horizon(trial, reason):
    options, deps, holder = trial
    holder["end"] = reason
    result = execution.run_oat_experiment(options, root_dir=ROOT, deps=deps, log=lambda message: None)
    assert result["stop_reason"] == reason
    assert result["executed_steps"] == 5
    assert holder["env"].reset_count == 1
    metadata, events = _read(options["output_dir"])
    assert metadata["recorded_control_steps"] == 5
    assert [e["kind"] for e in events].count(reason) == 1
    summary = json.loads((Path(options["output_dir"]) / "evaluation" / "summary.json").read_text())
    assert summary["episodes"][0]["full_requested_window"] is False
    assert summary["survival"]["status"] == "not_evaluable"


def test_interrupt_flushes_committed_physics_and_closes(trial):
    options, deps, holder = trial
    holder["interrupt"] = 5
    with pytest.raises(KeyboardInterrupt):
        execution.run_oat_experiment(options, root_dir=ROOT, deps=deps, log=lambda message: None)
    metadata, events = _read(options["output_dir"])
    assert metadata["status"] == "interrupted"
    assert metadata["recorded_control_steps"] == 5
    assert events[-1]["kind"] == "interrupted"
    assert holder["env"].closed
    assert not (Path(options["output_dir"]) / "evaluation").exists()


def test_missing_protocol_cannot_start_an_environment(trial):
    options, deps, holder = trial
    del options["online"]["gamma"]
    with pytest.raises(ValueError, match="online.gamma"):
        execution.run_oat_experiment(options, root_dir=ROOT, deps=deps)
    assert "env" not in holder
    assert not Path(options["output_dir"]).exists()


def test_actual_hydra_entrypoint_has_no_renderer_and_collects_only(trial, monkeypatch):
    options, deps, holder = trial
    options["payload"] = {"enabled": False}
    options["evaluation"] = {"enabled": False}
    original = execution.create_fada_playback_session
    def offline_session(**kwargs):
        kwargs["deps"] = deps
        return original(**kwargs)

    monkeypatch.setattr(execution, "create_fada_playback_session", offline_session)

    def overrides(value, prefix=""):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict) and item:
                yield from overrides(item, name)
            else:
                yield f"{name}={json.dumps(item, separators=(',', ':'))}"

    monkeypatch.setattr(sys, "argv", ["run_oat_experiment.py", *overrides(options)])
    # Any attempted use of the old rendering owner is a real failure in this headless path.
    monkeypatch.setitem(sys.modules, "unilab.visualization.fada_rendering", None)
    runpy.run_path(str(ROOT / "scripts" / "run_oat_experiment.py"), run_name="__main__")
    metadata, events = _read(options["output_dir"])
    assert metadata["recorded_control_steps"] == 10
    assert metadata["resolved_config"]["env"]["runtime_payload"]["enabled"] is False
    assert not any(e["kind"] == "payload_applied" for e in events)
    assert holder["env"].closed
    assert not (Path(options["output_dir"]) / "evaluation").exists()
