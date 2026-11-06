"""Focused proof of the G1 config/reward/observation boundary; no simulator."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.privileged_oracle import (
    FADAOracleCheckpointContract,
    canonical_fada_config_sha256,
    seal_fada_oracle_checkpoint,
    validate_fada_oracle_checkpoint_payload,
    validate_fada_oracle_lineage,
)
from unilab.algos.torch.distill.fada.privileged_oracle_sac import (
    resolve_privileged_locomotion_sac_runtime,
)
from unilab.base.curriculum import PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.registry import apply_cfg_overrides
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.joystick import G1WalkEnv
from unilab.envs.locomotion.g1.walk_commands import sample_g1_walk_commands
from unilab.envs.locomotion.g1.walk_config import G1RewardConfig, G1WalkFlatCfg
from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
from unilab.envs.locomotion.g1.walk_runtime_bindings import G1WalkRuntimeBindings
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.offpolicy.factory import build_offpolicy_env_cfg_override


@pytest.mark.parametrize("num_samples", [1, 4])
@pytest.mark.parametrize("straight_prob", [0.0, 0.12])
def test_straight_command_is_independent_of_batch_size(monkeypatch, num_samples, straight_prob):
    original = np.array([
        [-0.5, 0.3, -0.7],
        [-0.5, -0.2, 0.8],
        [0.6, 0.2, 0.75],
        [-0.6, -0.3, -0.75],
    ], dtype=np.float32)[:num_samples]
    draw = np.array([0.26]) if num_samples == 1 else np.array([0.1, 0.26, 0.7, 0.9])

    def uniform(*, low=0.0, high=1.0, size=None):
        if size == (num_samples, 3):
            return original.copy()
        if size == (num_samples,):
            return draw.copy()
        return np.full(size, (low + high) / 2)

    monkeypatch.setattr(np.random, "uniform", uniform)
    commands = SimpleNamespace(
        vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        small_xy_threshold=0.0,
        rel_standing_envs=0.2,
        rel_transition_envs=0.0,
        rel_straight_envs=straight_prob,
    )
    sampled = sample_g1_walk_commands(
        SimpleNamespace(cfg=SimpleNamespace(commands=commands)), num_samples
    )
    expected = original.copy()
    if num_samples == 4:
        expected[0] = 0.0
    if straight_prob > 0:
        expected[0 if num_samples == 1 else 1] = [0.5, 0.0, 0.0]
    np.testing.assert_allclose(sampled, expected)


def test_straight_bucket_does_not_replace_transition_commands(monkeypatch):
    original = np.array([
        [-0.5, 0.3, -0.7], [-0.5, -0.2, 0.8],
        [0.6, 0.2, 0.75], [-0.6, -0.3, -0.75],
    ], dtype=np.float32)

    def uniform(*, low=0.0, high=1.0, size=None):
        if size == (4, 3):
            return original.copy()
        if size == (4,):
            return np.array([0.1, 0.25, 0.35, 0.9])
        return np.broadcast_to((np.asarray(low) + high) / 2, size).copy()

    monkeypatch.setattr(np.random, "uniform", uniform)
    commands = SimpleNamespace(
        vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        transition_vel_limit=[[0.05, -0.1, -0.2], [0.25, 0.1, 0.2]],
        small_xy_threshold=0.0,
        rel_standing_envs=0.2,
        rel_transition_envs=0.1,
        rel_straight_envs=0.12,
    )
    sampled = sample_g1_walk_commands(
        SimpleNamespace(cfg=SimpleNamespace(commands=commands)), 4
    )
    np.testing.assert_allclose(sampled, [
        [0.0, 0.0, 0.0], [0.15, 0.0, 0.0],
        [0.5, 0.0, 0.0], original[3],
    ])


def test_yaml_parameters_reach_typed_config_without_weakening_checkpoint_identity(monkeypatch, capsys):
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "config-boundary")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=[
            "algo=sac", "task=sac/g1_walk_flat/mujoco_fada_fixed_contact",
            "env.commands.rel_transition_envs=0.2", "env.commands.resampling_time=3.0",
            "env.command_gated_phase_contact.duty_factor=0.6",
            "env.command_gated_phase_contact.contact_force_threshold=2.0",
            "reward.scales.phase_contact=-0.7", "reward.gait_frequency=1.2",
        ])
    assert cfg.algo.actor.oracle_behavior_profile == "configured_gait"
    raw = OmegaConf.to_container(cfg.reward, resolve=True)
    runtime = resolve_privileged_locomotion_sac_runtime(OmegaConf.to_container(cfg.algo, resolve=True))
    runtime.validate_training_config(cfg)
    typed = G1WalkFlatCfg()
    apply_cfg_overrides(typed, build_offpolicy_env_cfg_override("sac", cfg))
    typed.validate()
    assert typed.reward_config.feet_phase_mode == "fixed_contact"
    assert typed.reward_config.feet_orientation_stance_only
    assert typed.reward_config.scales["phase_contact"] == -0.7
    assert typed.reward_config.gait_frequency == 1.2
    assert typed.commands.resampling_time == 3.0
    assert OmegaConf.to_container(cfg.reward, resolve=True) == raw
    typed.command_gated_phase_contact.duty_factor = 1.1
    with pytest.raises(ValueError, match="duty_factor"):
        typed.validate()

    # The same master config can disable Reward, phase, both curricula and noise.
    switches = deepcopy(cfg)
    switches.reward.scales.phase_contact = 0.0
    switches.reward.feet_orientation_stance_only = False
    switches.env.gait_phase_enabled = False
    switches.env.curriculum.enabled = False
    switches.env.noise_config.level = 0.0
    switches.env.domain_rand.randomize_kp = False
    switches.env.domain_rand.actuator_strength.group_curriculum_enabled = False
    switches.algo.max_iterations = 1200
    switches.algo.save_interval = 400
    runtime.validate_training_config(switches)
    adapter = BackendAdapter(switches, root_dir=root, algo_name="sac")
    train = adapter.build_task_env_cfg_override()
    assert train["reward_config"]["scales"]["phase_contact"] == 0.0
    assert train["domain_rand"]["randomize_kp"] is False
    switches.reward.scales.phase_contact = -1.0
    with pytest.raises(ValueError, match="gait_phase_enabled"):
        runtime.validate_training_config(switches)
    switches.reward.scales.phase_contact = 0.0
    switches.training.play_only = True
    # Partial playback overrides preserve the checkpoint's ranges and commands.
    restored = deepcopy(train)
    restored["commands"]["dead_zone_xy"] = 0.07
    restored["domain_rand"]["kp_multiplier_range"] = [0.95, 1.05]
    played = adapter.build_play_env_cfg_override(restored)
    assert played["commands"]["dead_zone_xy"] == 0.07
    assert played["domain_rand"]["kp_multiplier_range"] == [0.95, 1.05]
    assert played["commands"]["rel_standing_envs"] == 1.0
    assert restored["commands"]["rel_standing_envs"] == 0.3
    adapter.report_effective_settings(played, mode="回放")
    assert "[G1 配置表 · 回放]" in capsys.readouterr().out

    contract = FADAOracleCheckpointContract(
        oracle_lineage_id="config-boundary", privileged_schema="g1_fada_privileged_v1",
        task_name="G1WalkFlat", backend="mujoco", action_scale=(1.0,), seed=1,
        obs_dim=3, critic_obs_dim=5, action_dim=2, body_names=("world", "pelvis"),
        actuated_joint_names=("left", "right"), privileged_field_slices=(("privileged", 0, 2),),
        asset_sha256="a" * 64,
        config_hashes=(("reward", canonical_fada_config_sha256(raw)),),
        behavior_profile="fixed_phase_contact_v2",
    )
    payload = seal_fada_oracle_checkpoint({}, contract, iteration=240)
    validate_fada_oracle_checkpoint_payload(payload, contract, expected_iteration=240)
    altered = replace(contract, config_hashes=(("reward", "b" * 64),))
    with pytest.raises(ValueError):
        validate_fada_oracle_checkpoint_payload(payload, altered, expected_iteration=240)
    configured_contract = replace(contract, behavior_profile="configured_gait",
                                  final_iteration=1200, save_interval=400)
    records = []
    for iteration in (400, 800, 1200):
        sealed = seal_fada_oracle_checkpoint({}, configured_contract, iteration=iteration)
        validate_fada_oracle_checkpoint_payload(sealed, configured_contract, expected_iteration=iteration)
        records.append(sealed["fada_privileged_oracle"])
    admitted = validate_fada_oracle_lineage(records)
    assert admitted.intermediate_iterations == (400, 800)
    assert admitted.final_iteration == 1200


def test_historical_reward_weights_and_curriculum_are_preserved(default_g1_reward_config):
    # Independent expected weighted contributions for a unit raw cost.
    cases = [
        ("command_height_v1", "feet_phase", 2.0, -2.0),
        ("command_height_v2", "feet_phase", 2.0, -1.0),
        ("phase_contact_v1", "phase_contact", 2.0, -2.0),
        ("fixed_phase_contact_v1", "phase_contact", 2.0, -2.0),
        ("fixed_phase_contact_v2", "phase_contact", -2.0, -1.0),
    ]
    for mode, term, old_weight, expected in cases:
        raw = dict(default_g1_reward_config, feet_phase_mode=mode,
                   scales={term: old_weight, "pose": -4.0})
        saved = deepcopy(raw)
        reward = G1RewardConfig(**raw)
        env = SimpleNamespace(cfg=SimpleNamespace(reward_config=reward))
        curriculum = PenaltyCurriculum(env, initial_scale=0.5,
                                      penalty_names=reward.penalty_curriculum_terms)
        assert reward.scales[term] == expected
        assert reward.scales["pose"] == -2.0
        assert (term in curriculum.penalty_names) == (old_weight < 0)
        assert asdict(G1RewardConfig(**asdict(reward))) == asdict(reward)
        assert raw == saved

        # Both feet in stance, both airborne: mismatch cost is exactly one.
        if term == "phase_contact":
            owner = object.__new__(G1WalkEnv)
            owner._reward_cfg = reward
            owner._cfg = G1WalkFlatCfg(reward_config=reward)
            owner._net_foot_force_z = lambda: (np.zeros(1), np.zeros(1))
            ctx = SimpleNamespace(info={"gait_phase": np.zeros((1, 2)),
                                        "commands": np.array([[0.2, 0.0, 0.0]])})
            np.testing.assert_allclose(owner._reward_phase_contact(ctx) * reward.scales[term], expected)


class _LifecycleOwner(G1WalkControlBindings, G1WalkRuntimeBindings):
    """Keep real command and runtime bindings; replace only physics and scoring."""

    def __init__(self, gated):
        self._cfg = G1WalkFlatCfg()
        self._cfg.commands.resampling_time = 4.0
        self._cfg.command_gated_phase_contact.enabled = gated
        self._cfg.domain_rand.actuator_strength.curriculum_progress_mode = "iterations"
        self._reward_cfg = SimpleNamespace(max_tilt_deg=45.0, min_base_height=0.5)
        self._num_envs = 1
        self.step_counter = 199
        self._episode_tracker = self._penalty_curriculum = None
        self.events = []
        self._backend = SimpleNamespace(get_sensor_data=lambda _: np.array([[0.0, 0.0, 1.0]]))
        self._fada_dr_provider = SimpleNamespace(
            update_iteration_curriculum=lambda *_: self.events.append("curriculum")
        )
        self._state = NpEnvState({}, np.array([7.0]), np.array([False]), np.array([False]), {
            "commands": np.array([[0.2, 0.0, 0.0]]), "steps": np.array([199]),
            "gait_phase": np.array([[0.0, np.pi]], dtype=get_global_dtype()),
            "command_is_null": np.array([False]),
            "command_gated_actuator_target_valid": np.array([True]),
        })

    def get_local_linvel(self):
        return np.zeros((1, 3))

    get_gyro = get_dof_pos = get_dof_vel = get_local_linvel

    def _gait_constraint_cfg(self):
        return SimpleNamespace(enabled=False)

    def _compute_obs(self, info, *_args):
        return {"obs": info["commands"].copy()}

    def _compute_reward(self, info, *_args):
        self.events.append("reward")
        return info["commands"][:, 0].copy()

    def _terrain_relative_base_height(self):
        return np.ones(1)

    def _forward_progress_failure(self, _info):
        return np.zeros(1, dtype=bool)

    def _refresh_command_gated_torque(self, *_args):
        pass

    def _debug_action_trace(self, *_args, **_kwargs):
        pass

    def _write_curriculum_log(self, _info):
        pass


def test_refresh_and_step_have_separate_command_transactions(monkeypatch):
    for gated in (False, True):
        owner = _LifecycleOwner(gated)

        def sample(_env, _count):
            owner.events.append("sample")
            return np.array([[0.8, 0.0, 0.0]])

        monkeypatch.setattr("unilab.envs.locomotion.g1.walk_control_bindings.sample_g1_walk_commands", sample)
        phase = owner._state.info["gait_phase"].copy()
        owner.refresh_state()
        owner.refresh_state()
        assert owner.events == []
        np.testing.assert_allclose(owner._state.reward, [7.0])
        np.testing.assert_array_equal(owner._state.info["gait_phase"], phase)
        owner._state = owner.update_state(owner._state)
        assert owner.events == ["reward", "sample", "curriculum"]
        np.testing.assert_allclose(owner._state.reward, [0.2])
        np.testing.assert_allclose(owner._state.obs["obs"], [[0.8, 0.0, 0.0]])
        after_step = deepcopy(owner._state.info)
        owner.refresh_state()
        assert owner.events == ["reward", "sample", "curriculum"]
        np.testing.assert_array_equal(owner._state.info["gait_phase"], after_step["gait_phase"])
        np.testing.assert_array_equal(owner._state.info["steps"], [199])
