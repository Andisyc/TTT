"""Behavior at the teacher, portable physics, and original-checkpoint boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.algos.test_oracle_fada_source import _SourceEnv, recipe as _recipe, source as _source
from unilab.algos.torch.distill.fada.upstream_oracle import (
    UpstreamOracleAlgorithm, upstream_oracle_observation,
)

recipe = _recipe
source = _source


def test_teacher_input_and_atomic_checkpoint_reload(source, tmp_path):
    path, payload, _ = source
    env = _SourceEnv(payload["metadata"])
    algorithm = UpstreamOracleAlgorithm(path, env, "cpu")
    obs = np.arange(98, dtype=np.float32)[None] / 100
    privilege = np.arange(205, dtype=np.float32)[None] / 200
    state = SimpleNamespace(obs={"obs": obs, "critic": np.concatenate((obs, privilege), axis=1)},
                            info={"critic_info": privilege})
    tensors = upstream_oracle_observation(state, device="cpu")
    inputs = {"actor_obs": tensors["obs"], "priv_info": tensors["priv_info"]}
    expected = algorithm.teacher.actions_from_env_observation(state.obs, state.info)
    before = {key: value.clone() for key, value in algorithm.teacher.state_dict().items()}
    torch.testing.assert_close(algorithm.inference(inputs), torch.from_numpy(expected))
    for key, value in algorithm.teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    broken = deepcopy(payload)
    broken["metadata"]["network_config"]["action_dim"] = 28
    bad = tmp_path / "incompatible.pt"
    torch.save(broken, bad)
    with pytest.raises(ValueError, match="network differs"):
        algorithm.load(str(bad))
    torch.testing.assert_close(algorithm.inference(inputs), torch.from_numpy(expected))
    algorithm.load(str(path))
    torch.testing.assert_close(algorithm.get_inference_policy()(inputs), torch.from_numpy(expected))


def test_portable_snapshot_carries_different_environment_masses():
    mujoco = pytest.importorskip("mujoco")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.base.scene import SceneCfg

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="pelvis">
      <joint name="hinge"/><geom type="sphere" size="0.1" mass="1"/>
      </body></worldbody><actuator><position joint="hinge" kp="10"/></actuator></mujoco>''')
    backends = [MuJoCoBackend(SceneCfg(model_file=model), num_envs=2, sim_dt=0.002,
                             base_name="pelvis", num_threads=1) for _ in range(2)]
    first, second = backends
    try:
        for backend in backends:
            backend.materialize()
        first.set_body_mass_and_com(np.array([0, 1]), np.array([1]), mass=np.array([[2.0], [3.0]]))
        first.step(np.array([[0.4], [-0.3]]), nsteps=2)
        snapshot = first.capture_transfer_rollout_state()
        second.restore_transfer_rollout_state(snapshot)
        restored = second.capture_transfer_rollout_state()
        np.testing.assert_array_equal(restored["model_fields"]["body_mass"], snapshot["model_fields"]["body_mass"])
        before = first.capture_rollout_state()
        control = np.array([[0.2], [-0.1]])
        first.step(control, nsteps=3)
        second.step(control, nsteps=3)
        np.testing.assert_allclose(first.get_physics_state(), second.get_physics_state(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(snapshot["state"]["physics_state"], before["physics_state"])
    finally:
        for backend in backends:
            backend.close()


def test_original_checkpoint_preserves_inference_and_source_semantics(source, tmp_path):
    from unilab.algos.torch.distill.fada.adaptation_checkpoint import load_fada_deployable_policy_checkpoint
    from unilab.algos.torch.distill.fada.collection_contract import OFFICIAL_FADA_SOURCE_REVISION
    from unilab.algos.torch.distill.fada.official_model import build_official_fada_student
    from unilab.algos.torch.distill.fada.oracle_deployment import oracle_student_source_profile
    from unilab.algos.torch.distill.fada.runtime_policy import OfficialFADAPlaybackProfile
    from unilab.algos.torch.distill.fada.upstream_checkpoint import load_upstream_fada_checkpoint

    oracle_path, oracle_payload, _ = source
    config = dict(history_len=3, pred_horizon=2, planner_d_model=16, planner_nhead=2,
                  planner_num_layers=1, planner_dim_feedforward=32, planner_dropout=0.0,
                  idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1,
                  idm_decoder_num_layers=1, idm_dim_feedforward=32, idm_dropout=0.0)
    reference = build_official_fada_student(config).eval()
    cfg = reference.official_model_config
    terms = ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    identity = dict(upstream_revision=OFFICIAL_FADA_SOURCE_REVISION,
                    expert_checkpoint=str(oracle_path), expert_checkpoint_sha256="a" * 64,
                    experiment_config_sha256="b" * 64)
    meta = oracle_payload["metadata"]
    profile = dict(kind="unilab_oracle", version=1, student_observation_contract="official_compact_64_v1",
                   control_contract={key: meta[key] for key in ("observation_layout", "privileged_field_slices", "body_names", "actuated_joint_names", "asset_sha256", "ctrl_dt", "sim_dt", "action_scale", "default_joint_pos", "kp", "kd")},
                   environment_config=meta["environment_config"],
                   teacher_identity=dict(format=oracle_payload["format"], identity=meta["identity"], checkpoint_sha256="a" * 64))
    control = dict(source_identity=identity, joint_names=meta["actuated_joint_names"],
                   default_joint_pos=meta["default_joint_pos"], action_scale=[1.0]*29,
                   stiffness=meta["kp"], damping=meta["kd"], clip_actions=None,
                   sim_dt=meta["sim_dt"], ctrl_dt=meta["ctrl_dt"], gait_frequency=1.0,
                   gait_clock_mode="continuous", phase_init=None)
    payload = dict(model_state_dict=reference.state_dict(), cfg=cfg,
                   obs_dim=64, act_dim=29, cmd_dim=7,
                   compact_obs=dict(term_order=terms, term_scale=dict.fromkeys(terms, 1.0),
                                    term_noise=dict.fromkeys(terms, 0.0), noise_enabled=False),
                   unilab_upstream_source=dict(version=1, source_identity=identity,
                                              source_profile=profile, control=control))
    path = tmp_path / "model_final.pt"
    torch.save(payload, path)
    loaded = load_fada_deployable_policy_checkpoint(path)
    args = (torch.randn(2, 3, 64), torch.randn(2, 3, 29), torch.randn(2, 7))
    mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.inference_mode():
        expected = reference(*args, history_valid_mask=mask)
        actual = loaded.policy(*args, history_valid_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = oracle_student_source_profile(loaded.checkpoint, observation_contract="official_compact_64_v1")
    assert restored == profile
    controls = OfficialFADAPlaybackProfile.from_mapping(control, source_identity=identity)
    assert controls.clip_actions is None and controls.phase_init is None
    from dataclasses import replace
    from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
    with torch.no_grad():
        loaded.policy.idm.action_head.weight.zero_()
        loaded.policy.idm.action_head.bias.fill_(9.0)
    unclipped = OfficialFADAPlaybackController(
        loaded.policy, profile=controls, target_joint_names=controls.joint_names, device="cpu",
    )
    clipped = OfficialFADAPlaybackController(
        loaded.policy, profile=replace(controls, clip_actions=0.5),
        target_joint_names=controls.joint_names, device="cpu",
    )
    torch.testing.assert_close(unclipped.act_projected(torch.zeros(1, 64), torch.zeros(1, 7)), torch.full((1, 29), 9.0))
    torch.testing.assert_close(clipped.act_projected(torch.zeros(1, 64), torch.zeros(1, 7)), torch.full((1, 29), 0.5))
    payload.pop("unilab_upstream_source")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="explicit source"):
        load_upstream_fada_checkpoint(path)
