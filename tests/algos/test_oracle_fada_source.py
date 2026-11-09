"""Source compatibility, frozen teacher inference, and failed-construction cleanup."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.distill.fada import oracle_source
from unilab.algos.torch.distill.fada.observation import project_fada_g1_state
from unilab.training.oracle.checkpoint import STATE_FIELDS, validate_environment_contract
from unilab.training.oracle.config import OracleInterface
from unilab.training.oracle.playback import OracleInferencePolicy


class _SourceEnv:
    obs_groups_spec = {"obs": 98, "critic": 303}
    action_space = SimpleNamespace(shape=(29,))

    def __init__(self, profile):
        self.profile = deepcopy(profile)
        self.closed = False
        self.fail_close = False

    def get_motion_quality_deployment_profile(self):
        return deepcopy(self.profile)

    def init_state(self):
        pytest.fail("source construction must leave initialization to the collector")

    def reset(self):
        pytest.fail("source construction must not reset the environment")

    def step(self, actions):
        pytest.fail("source construction must not advance physics")

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("cleanup failure")


@pytest.fixture(scope="module")
def recipe():
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "offpolicy"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco_motion_quality_oracle"])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture
def source(tmp_path, monkeypatch, recipe):
    interface = OracleInterface()
    control = {
        "observation_layout": [list(term) for term in interface.actor_obs_layout],
        "privileged_field_slices": [["privileged", 0, 205]],
        "body_names": ["world", "pelvis"],
        "actuated_joint_names": [f"joint_{i}" for i in range(29)],
        "asset_sha256": "a" * 64,
        "ctrl_dt": 0.02,
        "sim_dt": 1 / 150,
        "action_scale": 1.0,
        "default_joint_pos": [0.0] * 29,
        "kp": [40.0] * 29,
        "kd": [2.0] * 29,
    }
    actor_options = {
        "priv_info_embed_dim": 2,
        "priv_mlp_hidden_dims": [4, 2],
        "priv_info_normalization": True,
    }
    with torch.random.fork_rng():
        torch.manual_seed(7)
        actor = build_actor("hora_sac", 98, 29, 16, False, "cpu", priv_info_dim=205, **actor_options)
    normalizer = EmpiricalNormalization(shape=98, device="cpu")
    state: dict[str, Any] = {name: None for name in STATE_FIELDS}
    state.update(actor=actor.state_dict(), obs_normalizer=normalizer.state_dict(), update_count=2)
    payload = {
        "format": interface.checkpoint_format,
        "metadata": {
            **deepcopy(control),
            "identity": interface.identity,
            "algorithm": "sac",
            "iteration": 2,
            "config": deepcopy(recipe),
            "environment_config": {
                "scene": {"model_file": "unused-scene.xml"},
                "reward_config": deepcopy(recipe["reward"]),
                "sim_dt": control["sim_dt"],
                "ctrl_dt": control["ctrl_dt"],
            },
            "obs_normalization": True,
            "priv_info_normalization": True,
            "network_config": {
                "obs_dim": 98, "critic_obs_dim": 303, "action_dim": 29,
                "actor_hidden_dim": 16, "use_layer_norm": False, "actor": actor_options,
            },
        },
        "state": state,
    }
    path = tmp_path / "oracle.pt"
    torch.save(payload, path)
    envs = []

    def create_env(cfg, *, num_envs, env_cfg_override):
        env = _SourceEnv(control)
        envs.append(env)
        return env

    monkeypatch.setattr(oracle_source, "create_env", create_env)
    monkeypatch.setattr(oracle_source, "ensure_registries", lambda **kwargs: None)
    return path, payload, envs


def test_source_inference_matches_oracle_and_keeps_both_normalizers_frozen(source):
    path, payload, envs = source
    env, teacher, profile = oracle_source.build_oracle_fada_source(path, num_envs=1)
    try:
        reference = OracleInferencePolicy(payload, env, "cpu")
        obs = np.arange(98, dtype=np.float32)[None, :] / 100
        privilege = np.arange(205, dtype=np.float32)[None, :] / 200
        mapping = {"obs": obs, "critic": np.concatenate((obs, privilege), axis=1)}
        before = {key: value.clone() for key, value in teacher.state_dict().items()}
        with torch.inference_mode():
            expected = reference(torch.from_numpy(obs), torch.from_numpy(privilege)).numpy()
        for _ in range(2):
            actual = teacher.actions_from_env_observation(mapping, {"critic_info": privilege})
            np.testing.assert_allclose(actual, expected, atol=1e-7)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert not teacher.training
        assert all(not parameter.requires_grad for parameter in teacher.parameters())
        restored = json.loads(json.dumps(profile))
        validate_environment_contract(restored["control_contract"], env.get_motion_quality_deployment_profile())
        assert restored["kind"] == "unilab_oracle"
        assert restored["student_observation_contract"] == "g1_fada_state_v2"
        projected = project_fada_g1_state(obs)
        changed = obs.copy()
        changed[:, 64:96] += 100
        np.testing.assert_array_equal(project_fada_g1_state(changed), projected)
        with pytest.raises(ValueError, match="critic observations"):
            teacher.actions_from_env_observation({"obs": obs}, None)
        assert not env.closed
    finally:
        env.close()


@pytest.mark.parametrize("fault", ["layout", "action_dim"])
def test_incompatible_source_is_rejected_before_environment_allocation(source, fault):
    path, payload, envs = source
    if fault == "layout":
        layout = payload["metadata"]["observation_layout"]
        layout[0], layout[1] = layout[1], layout[0]
    else:
        payload["metadata"]["network_config"]["action_dim"] = 28
    torch.save(payload, path)
    with pytest.raises(ValueError):
        oracle_source.build_oracle_fada_source(path, num_envs=1)
    assert not envs


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_environment_contract_failure_closes_env_and_preserves_original_error(source, monkeypatch, cleanup_fails):
    path, payload, envs = source
    factory = oracle_source.create_env

    def incompatible_env(*args, **kwargs):
        env = cast(_SourceEnv, factory(*args, **kwargs))
        env.profile["ctrl_dt"] = 0.01
        env.fail_close = cleanup_fails
        return env

    monkeypatch.setattr(oracle_source, "create_env", incompatible_env)
    with pytest.raises(ValueError, match="ctrl_dt"):
        oracle_source.build_oracle_fada_source(path, num_envs=1)
    assert envs[0].closed


def test_damaged_actor_closes_environment_after_checkpoint_metadata_was_accepted(source):
    path, payload, envs = source
    payload["state"]["actor"] = {
        key: value for key, value in payload["state"]["actor"].items()
        if key.startswith("priv_info_normalizer.")
    }
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        oracle_source.build_oracle_fada_source(path, num_envs=1)
    assert envs[0].closed
