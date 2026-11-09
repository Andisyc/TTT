"""Behavioral coverage for the independent, frozen Torch teacher boundary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from fada_oracle.oracle import OracleAlgorithm, OraclePolicy, read_checkpoint
from fada_oracle.oracle.checkpoint import OBSERVATION_LAYOUT
from fada_oracle.oracle.policy import EmpiricalNormalization, HoraSACActor


def _statistics(width: int) -> dict[str, torch.Tensor]:
    return {
        "_mean": torch.linspace(-0.4, 0.6, width).unsqueeze(0),
        "_var": torch.linspace(0.8, 1.8, width).unsqueeze(0),
        "_std": torch.linspace(0.05, 0.9, width).unsqueeze(0),
        "count": torch.tensor(1234, dtype=torch.long),
    }


@pytest.fixture
def payload() -> dict:
    network = {
        "obs_dim": 98, "critic_obs_dim": 303, "action_dim": 29,
        "actor_hidden_dim": 16, "critic_hidden_dim": 24, "use_layer_norm": True,
        "num_atoms": 101, "v_min": -30.0, "v_max": 30.0,
        "actor": {
            "fixed_privileged_input": False, "priv_info_embed_dim": 3,
            "priv_mlp_hidden_dims": [12, 6, 3], "priv_info_normalization": True,
        },
    }
    with torch.random.fork_rng():
        torch.manual_seed(42)
        actor = HoraSACActor(network)
    actor_state = actor.state_dict()
    for key, value in _statistics(205).items():
        actor_state[f"priv_info_normalizer.{key}"] = value
    return {
        "format": "unilab_motion_quality_oracle_v1",
        "metadata": {
            "identity": "motion_quality_oracle_v1", "algorithm": "sac", "iteration": 7,
            "config": {"algo": {"algo": "sac"}}, "network_config": network,
            "obs_normalization": True, "priv_info_normalization": True,
            "observation_layout": [list(item) for item in OBSERVATION_LAYOUT],
            "privileged_field_slices": [["privileged", 0, 205]],
            "body_names": ["pelvis"], "actuated_joint_names": [f"joint_{i}" for i in range(29)],
            "asset_sha256": "a" * 64, "ctrl_dt": 0.02, "sim_dt": 0.002,
            "action_scale": 1.0, "default_joint_pos": [0.0] * 29,
            "kp": [30.0] * 29, "kd": [1.0] * 29, "environment_config": {},
        },
        "state": {
            "actor": actor_state, "obs_normalizer": _statistics(98), "update_count": 7,
            "qnet": {}, "qnet_target": {}, "log_alpha": torch.tensor(0.1),
            "actor_optimizer": {}, "q_optimizer": {}, "alpha_optimizer": {}, "amp_scaler": {},
        },
    }


def _save(tmp_path: Path, payload: dict, name: str = "teacher.pt") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _observations() -> dict[str, torch.Tensor]:
    return {
        "actor_obs": torch.linspace(-1.0, 1.0, 3 * 98).reshape(3, 98),
        "priv_info": torch.linspace(-2.0, 2.0, 3 * 205).reshape(3, 205),
    }


def test_read_checkpoint_preserves_full_artifact(tmp_path: Path, payload: dict) -> None:
    loaded = read_checkpoint(_save(tmp_path, payload))
    assert loaded.keys() == payload.keys()
    assert loaded["metadata"] == payload["metadata"]
    assert loaded["state"].keys() == payload["state"].keys()
    torch.testing.assert_close(loaded["state"]["log_alpha"], payload["state"]["log_alpha"])
    for key, value in payload["state"]["actor"].items():
        torch.testing.assert_close(loaded["state"]["actor"][key], value, rtol=0, atol=0)


def test_empirical_normalization_uses_saved_std_and_external_epsilon() -> None:
    normalizer = EmpiricalNormalization(3)
    state = _statistics(3)
    normalizer.load_state_dict(state, strict=True)
    normalizer.train(True)
    observation = torch.tensor([[0.0, 1.0, -1.0], [4.0, -3.0, 2.0]])
    torch.testing.assert_close(
        normalizer(observation, update=True),
        (observation - state["_mean"]) / (state["_std"] + 0.01),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        normalizer(observation, center=False, update=True),
        observation / (state["_std"] + 0.01), rtol=0, atol=0,
    )
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_policy_is_frozen_deterministic_and_never_updates_statistics(payload: dict) -> None:
    teacher = OraclePolicy(payload)
    inputs = _observations()
    before = {key: value.clone() for key, value in teacher.state_dict().items()}
    teacher.train(True)
    assert all(not module.training for module in teacher.modules())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    first = teacher(inputs["actor_obs"].requires_grad_(), inputs["priv_info"])
    assert first.shape == (3, 29)
    assert not first.requires_grad
    assert torch.isfinite(first).all()
    for _ in range(4):
        torch.testing.assert_close(teacher(inputs["actor_obs"], inputs["priv_info"]), first, rtol=0, atol=0)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_both_saved_normalizers_influence_actual_actions(payload: dict) -> None:
    observations = _observations()
    baseline = OraclePolicy(payload)(observations["actor_obs"], observations["priv_info"])
    for prefix in ("observation", "privileged"):
        changed = deepcopy(payload)
        stats = changed["state"]["obs_normalizer"] if prefix == "observation" else changed["state"]["actor"]
        key = "_mean" if prefix == "observation" else "priv_info_normalizer._mean"
        stats[key] = stats[key] + torch.linspace(0.0, 5.0, stats[key].numel()).reshape_as(stats[key])
        actual = OraclePolicy(changed)(observations["actor_obs"], observations["priv_info"])
        assert not torch.allclose(actual, baseline)


def test_mean_sanitization_tanh_scale_and_bias(payload: dict) -> None:
    state = payload["state"]["actor"]
    state["action_mean_head.weight"].zero_()
    state["action_mean_head.bias"].zero_()
    state["action_mean_head.bias"][:5] = torch.tensor([20.0, -20.0, float("nan"), float("inf"), -float("inf")])
    state["action_scale"] = torch.linspace(0.2, 1.2, 29)
    state["action_bias"] = torch.linspace(-0.3, 0.3, 29)
    inputs = _observations()
    actions = OraclePolicy(payload)(inputs["actor_obs"], inputs["priv_info"])
    sanitized_mean = torch.zeros(29)
    sanitized_mean[:5] = torch.tensor([10.0, -10.0, 0.0, 10.0, -10.0])
    expected = torch.tanh(sanitized_mean) * state["action_scale"] + state["action_bias"]
    torch.testing.assert_close(actions, expected.expand_as(actions), rtol=0, atol=0)


def test_projection_and_no_layer_norm_checkpoint_are_loaded_strictly(payload: dict) -> None:
    network = payload["metadata"]["network_config"]
    network["use_layer_norm"] = False
    network["actor"]["priv_info_embed_dim"] = 5
    payload["state"]["actor"] = HoraSACActor(network).state_dict()
    teacher = OraclePolicy(payload)
    assert isinstance(teacher.actor.priv_projection, nn.Linear)
    assert not any(isinstance(module, nn.LayerNorm) for module in teacher.modules())
    inputs = _observations()
    actions = teacher(inputs["actor_obs"], inputs["priv_info"])
    assert actions.shape == (3, 29)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize("failure", ["missing_weight", "wrong_shape", "unexpected_weight", "missing_stats", "network", "family"])
def test_failed_replacement_retains_teacher_path_and_actions(tmp_path: Path, payload: dict, failure: str) -> None:
    original_path = _save(tmp_path, payload)
    algorithm = OracleAlgorithm(original_path)
    teacher = algorithm.teacher
    metadata = deepcopy(algorithm.metadata)
    inference = algorithm.get_inference_policy()
    observations = _observations()
    expected = inference(observations)
    replacement = deepcopy(payload)
    if failure == "missing_weight":
        del replacement["state"]["actor"]["action_mean_head.weight"]
    elif failure == "wrong_shape":
        replacement["state"]["actor"]["action_mean_head.weight"] = torch.zeros(28, 4)
    elif failure == "unexpected_weight":
        replacement["state"]["actor"]["unexpected.weight"] = torch.zeros(1)
    elif failure == "missing_stats":
        del replacement["state"]["obs_normalizer"]["_std"]
    elif failure == "network":
        # Even a critic-only change must fail the exact resident network contract.
        replacement["metadata"]["network_config"]["critic_hidden_dim"] += 1
    else:
        replacement["format"] = "unsupported_checkpoint"
    with pytest.raises((ValueError, RuntimeError)):
        algorithm.load(_save(tmp_path, replacement, "replacement.pt"))
    assert algorithm.teacher is teacher
    assert algorithm.checkpoint == str(original_path.resolve())
    assert algorithm.metadata == metadata
    torch.testing.assert_close(inference(observations), expected, rtol=0, atol=0)


def test_unreadable_replacement_retains_resident_teacher(tmp_path: Path, payload: dict) -> None:
    algorithm = OracleAlgorithm(_save(tmp_path, payload))
    teacher, path = algorithm.teacher, algorithm.checkpoint
    with pytest.raises(FileNotFoundError):
        algorithm.load(tmp_path / "absent.pt")
    assert algorithm.teacher is teacher
    assert algorithm.checkpoint == path


def test_successful_replacement_changes_retained_inference_callable(tmp_path: Path, payload: dict) -> None:
    algorithm = OracleAlgorithm(_save(tmp_path, payload))
    teacher = algorithm.teacher
    inference = algorithm.get_inference_policy()
    observations = _observations()
    before = inference(observations)
    replacement = deepcopy(payload)
    replacement["state"]["actor"]["action_mean_head.weight"].zero_()
    replacement["state"]["actor"]["action_mean_head.bias"].fill_(0.75)
    replacement_path = _save(tmp_path, replacement, "replacement.pt")
    algorithm.load(replacement_path)
    after = inference(observations)
    assert algorithm.teacher is not teacher
    assert algorithm.checkpoint == str(replacement_path.resolve())
    assert not torch.allclose(after, before)
    torch.testing.assert_close(after, torch.full_like(after, torch.tanh(torch.tensor(0.75)).item()))
    assert not after.requires_grad
    assert all(not module.training for module in algorithm.teacher.modules())
    assert all(not parameter.requires_grad for parameter in algorithm.teacher.parameters())


@pytest.mark.parametrize("section,key,value", [
    ("network_config", "obs_dim", 97),
    ("network_config", "critic_obs_dim", 302),
    ("network_config", "action_dim", 28),
    ("metadata", "identity", "other_teacher"),
    ("metadata", "obs_normalization", False),
])
def test_unsupported_metadata_fails_before_inference(payload: dict, section: str, key: str, value: object) -> None:
    owner = payload["metadata"] if section == "metadata" else payload["metadata"][section]
    owner[key] = value
    with pytest.raises(ValueError):
        OraclePolicy(payload)


def test_input_shape_mismatch_is_rejected(payload: dict) -> None:
    teacher = OraclePolicy(payload)
    with pytest.raises(ValueError, match="actor observations"):
        teacher(torch.zeros(2, 97), torch.zeros(2, 205))
    with pytest.raises(ValueError, match="privileged observations"):
        teacher(torch.zeros(2, 98), torch.zeros(3, 205))
