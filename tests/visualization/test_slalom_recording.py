"""Real G1 scene integration; no fake checkpoint or policy-quality claims."""
from pathlib import Path
from typing import Any

import mujoco as _mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base.scene import ScenePoleCfg
from unilab.envs.locomotion.g1.joystick import G1WalkEnv
from unilab.envs.locomotion.g1.motion_quality_config import MOTION_QUALITY_TERMS, MotionQualityRewardConfig
from unilab.envs.locomotion.g1.walk_config import G1WalkFlatCfg
from unilab.training.oracle.checkpoint import validate_environment_contract
from unilab.visualization.slalom import SlalomCommandProgram, SlalomConfig, SlalomEvaluator

ROOT = Path(__file__).resolve().parents[2]
mujoco: Any = _mujoco


def _env_cfg(poles):
    reward = MotionQualityRewardConfig(
        scales={name: 1.0 if name == "alive" else 0.0 for name in MOTION_QUALITY_TERMS},
        tracking_sigma=0.25, gait_frequency=1.5, feet_phase_swing_height=0.05,
        feet_phase_tracking_sigma=0.05, base_height_target=0.75, min_base_height=0.35,
        max_tilt_deg=65, gait_period_randomization_width=0,
    )
    cfg = G1WalkFlatCfg(reward_config=reward, mujoco_num_threads=1, post_step_forward_sensor=True)
    cfg.scene.task_poles = poles
    cfg.fada_privileged_observation.enabled = True
    cfg.reset_base_qvel_limit = cfg.standing_reset_base_qvel_limit = 0
    cfg.noise_config.level = 0
    for name in vars(cfg.domain_rand):
        if name.startswith("randomize_") or name == "push_robots":
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.actuator_strength.enabled = False
    cfg.domain_rand.actuator_strength.group_curriculum_enabled = False
    cfg.domain_rand.torque_rfi_fraction = 0
    cfg.commands.resampling_time = 0
    cfg.commands.heading_command = False
    cfg.commands.dead_zone_enabled = False
    cfg.commands.small_xy_threshold = 0
    cfg.commands.rel_standing_envs = 1
    cfg.max_episode_seconds = 120
    return cfg


def _poles():
    return [ScenePoleCfg(f"slalom_pole_{i+1}", (1.5+i, 0.5, 0.4)) for i in range(5)]


@pytest.fixture
def physical_env():
    np.random.seed(0)
    env = G1WalkEnv(_env_cfg(_poles()), num_envs=1, backend_type="mujoco")
    try:
        env.reset_all()
        env.set_autoreset(False)
        yield env
    finally:
        env.close()


def test_real_g1_robot_contract_survives_free_task_poles(physical_env):
    env = physical_env
    original = G1WalkEnv(_env_cfg([]), num_envs=1, backend_type="mujoco")
    try:
        original.reset_all()
        validate_environment_contract(original.get_motion_quality_deployment_profile(),
                                      env.get_motion_quality_deployment_profile())
        assert env.obs_groups_spec == original.obs_groups_spec
        np.testing.assert_allclose(env.default_angles, original.default_angles, atol=0, rtol=0)
        assert env.get_dof_pos().shape == env.get_dof_vel().shape == (1, 29)
        assert env.state is not None and original.state is not None
        assert env.state.info["fada_body_mass_scale"].shape == original.state.info["fada_body_mass_scale"].shape
        pos, quat, contacts = env.get_task_pole_state()
        np.testing.assert_allclose(pos[0], [p.position for p in _poles()], atol=1e-7)
        np.testing.assert_allclose(quat[0], np.tile([1, 0, 0, 0], (5, 1)))
        assert not contacts.any()
        before = {key: value.shape for key, value in env.state.obs.items()}
        env.step(np.zeros((1, 29)))
        assert {key: value.shape for key, value in env.state.obs.items()} == before
        assert all(np.isfinite(value).all() for value in env.state.obs.values())
    finally:
        original.close()


def test_actual_collision_toppling_and_reset_are_not_visual_only(physical_env):
    env = physical_env
    backend = env._backend
    qpos, qvel = env._init_qpos.copy()[None, :], env._init_qvel.copy()[None, :]
    joint = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "slalom_pole_1_free")
    start = int(backend.model.jnt_qposadr[joint])
    qpos[0, start:start+3] = [0, 0, 0.4]
    backend.set_state(np.array([0]), qpos, qvel)
    assert env.get_task_pole_state()[2][0, 0]
    env.reset_all()
    pole = np.array([backend.get_body_id("slalom_pole_1")], dtype=np.int32)
    backend.apply_body_force(pole, np.array([[[0., 100., 0.]]]))
    ctrl = env.default_angles[None, :].copy()
    backend.step(ctrl, nsteps=20)
    backend.step(ctrl, nsteps=100)
    pos, quat, _ = env.get_task_pole_state()
    assert np.linalg.norm(pos[0, 0, :2]-[1.5, 0.5]) > 0.1
    assert 1-2*(quat[0, 0, 1]**2+quat[0, 0, 2]**2) < np.cos(np.deg2rad(30))
    env.reset_all()
    np.testing.assert_allclose(env.get_task_pole_state()[0][0], [p.position for p in _poles()], atol=1e-7)
    np.testing.assert_allclose(env.get_task_pole_state()[1][0], np.tile([1, 0, 0, 0], (5, 1)))


def test_integrated_command_and_measured_gate_success_require_real_motion():
    config = SlalomConfig(pole_count=5)
    program = SlalomCommandProgram.from_config(config, ctrl_dt=0.02, prediction_horizon=6)
    evaluator = SlalomEvaluator(config, program.pole_centers)
    pose = np.zeros(3)
    poles = np.column_stack((program.pole_centers, np.full(5, 0.4)))
    quats = np.tile([1., 0., 0., 0.], (5, 1))
    for step in range(program.total_control_steps):
        vx, vy, wz = program.command_at_control_step(step)
        turn = wz*program.ctrl_dt
        mid = pose[2]+turn/2
        pose[:2] += np.array([np.cos(mid)*vx-np.sin(mid)*vy,
                             np.sin(mid)*vx+np.cos(mid)*vy])*program.ctrl_dt*np.sinc(turn/(2*np.pi))
        pose[2] += turn
        evaluator.update(pose[:2], poles, quats, np.zeros(5, bool), False, 0.02, False)
    # Integrator roundoff is not a physical retreat at the analytical finish.
    assert abs(pose[0]-program.finish_x_m) < 1e-10
    pose[0] = program.finish_x_m
    for _ in range(100):
        evaluator.update(pose[:2], poles, quats, np.zeros(5, bool), False, 0.02, True)
    assert evaluator.report()["success"]
    np.testing.assert_array_equal(program.command_at_control_step(program.total_control_steps*2), 0)
    still = SlalomEvaluator(config, program.pole_centers)
    still.update(np.zeros(2), poles, quats, np.zeros(5, bool), False, program.duration_s+2, True)
    assert not still.report()["success"]


def test_real_factory_uses_slalom_profile_without_replacing_gerono():
    from unilab.training import ensure_registries
    from unilab.visualization.playback_cli import _build_play_args
    from unilab.visualization.playback_controls import _build_playback_config
    from unilab.visualization.playback_distill_sessions import create_fada_playback_session

    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT/"conf/distill"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "task=g1_walk_flat/mujoco", "recording_profile=pole_slalom", "training.device=cpu",
            "interactive.action_mode=zero", "video_recording.enabled=false",
        ])
    assert not cfg.demo.figure_eight.enabled and cfg.demo.slalom.enabled
    args = _build_play_args(cfg, algo="fada")
    session, _, _ = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1), cfg=cfg, root_dir=ROOT,
        device="cpu", log=lambda _: None,
    )
    private = Path(session.env.get_scene_artifacts().artifacts_dir)
    try:
        assert isinstance(session.command_program, SlalomCommandProgram)
        session.set_autoreset(False)
        session.reset()
        pos, _, _ = session.env.get_task_pole_state()
        np.testing.assert_allclose(pos[0, :, :2], session.command_program.pole_centers, atol=1e-7)
        session.step_once()
        assert session.actions is not None
        assert session.actions.shape == (1, 29)
    finally:
        session.wrapped_env.close()
    assert not private.exists()
