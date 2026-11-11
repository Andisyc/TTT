"""Figure-eight turn direction and closed-path command integration."""

import numpy as np
import pytest

from unilab.visualization.demo_commands import (
    DoubleCircleCommandConfig,
    DoubleCircleCommandProgram,
    FigureEightCommandConfig,
    FigureEightCommandProgram,
)


@pytest.mark.parametrize("width,height", [(4.0, 2.0), (6.0, 3.0), (4.0, 1.0), (0.4, 0.2)])
def test_figure_eight_turns_without_reversals_and_closes(width, height):
    program = FigureEightCommandProgram.from_config(
        FigureEightCommandConfig(width_m=width, height_m=height), ctrl_dt=0.02,
    )
    commands = program.commands
    half = len(commands) // 2
    assert np.all(commands[:half, 2] < 0.0)
    assert np.all(commands[half:, 2] > 0.0)
    assert np.all(commands[:, 0] > 0.0)
    np.testing.assert_array_equal(commands[:, 1], 0.0)

    turns = commands[:, 2] * program.segment_duration_s
    headings = np.concatenate(([0.0], np.cumsum(turns)))
    mean_headings = (headings[:-1] + headings[1:]) / 2.0
    distances = commands[:, 0] * program.segment_duration_s * np.sinc(turns / (2.0 * np.pi))
    positions = np.vstack((np.zeros(2), np.cumsum(
        distances[:, None] * np.column_stack((np.cos(mean_headings), np.sin(mean_headings))),
        axis=0,
    )))
    np.testing.assert_allclose(positions, program.reference_points, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(positions[half], 0.0, atol=1e-12)
    np.testing.assert_allclose(positions[-1], 0.0, atol=1e-12)
    assert abs(headings[-1]) < 1e-12
    assert program.segment_steps == 40
    assert program.segment_index_at_control_step(program.total_control_steps) == 0


def test_double_circle_holds_unequal_durations_and_repeats_at_cycle_boundary():
    program = DoubleCircleCommandProgram.from_config(
        DoubleCircleCommandConfig(0.3, 0.3, 1.01, 1.07), ctrl_dt=0.02,
    )
    assert program.phase_steps == (51, 54)
    assert program.total_control_steps == 105
    assert program.duration_s == pytest.approx(2.1)
    for step in (0, 50, 105):
        np.testing.assert_allclose(program.command_at_control_step(step), [0.3, 0.0, -0.3])
    for step in (51, 104):
        np.testing.assert_allclose(program.command_at_control_step(step), [0.3, 0.0, 0.3])
    np.testing.assert_allclose(
        np.linalg.norm(program.reference_points[:129] - [0.0, -1.0], axis=1), 1.0,
    )
    np.testing.assert_allclose(
        np.linalg.norm(program.reference_points[128:] - [0.0, 1.0], axis=1), 1.0,
    )
    np.testing.assert_array_equal(program.reference_points[[0, 128, -1]], 0.0)


_VELOCITY_LIMITS = np.array([[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]])


def _advance_planar_pose(pose, command, dt, *, speed_scale=1.0, yaw_bias=0.0):
    """Independent exact integration of a held body-frame planar twist."""
    turn = (command[2] + yaw_bias) * dt
    middle = pose[2] + turn / 2.0
    rotation = np.array([[np.cos(middle), -np.sin(middle)],
                         [np.sin(middle), np.cos(middle)]])
    pose[:2] += rotation @ command[:2] * speed_scale * dt * np.sinc(turn / (2.0 * np.pi))
    pose[2] += turn


def test_figure_eight_open_loop_repeats_closed_pose_and_holds_commands():
    program = FigureEightCommandProgram.from_config(
        FigureEightCommandConfig(width_m=4.0, height_m=2.0), ctrl_dt=0.02,
    )
    origin = np.array([2.0, -1.0, 0.7])
    pose = origin.copy()
    previous_command = None
    for step in range(3 * program.total_control_steps):
        command = program.command_at_control_step(step)
        if step % program.segment_steps:
            np.testing.assert_array_equal(command, previous_command)
        previous_command = command
        _advance_planar_pose(pose, command, program.ctrl_dt)
        if (step + 1) % program.total_control_steps == 0:
            np.testing.assert_allclose(pose, origin, atol=1e-10)




def test_figure_eight_session_repeats_without_pose_access_or_timeout_phase_reset():
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from unilab.visualization.fada_recording import (
        FADAVideoRecordingConfig, resolve_fada_recording_timing,
    )
    from unilab.visualization.playback_distill_sessions import create_fada_playback_session
    from unilab.visualization.playback_sessions import RslRlPlaybackConfig

    class PlanarEnv:
        def __init__(self, overrides):
            self.pose = np.zeros(3)
            self.steps = 0
            self.cfg = SimpleNamespace(
                ctrl_dt=0.02, max_episode_seconds=overrides["max_episode_seconds"],
                commands=SimpleNamespace(
                    vel_limit=_VELOCITY_LIMITS, resampling_time=0.0,
                    heading_command=False, dead_zone_enabled=False, small_xy_threshold=0.0,
                ),
            )
            self.state = SimpleNamespace(info={"commands": np.zeros((1, 3))})
            self.action_space = SimpleNamespace(shape=(3,))

        def refresh_state(self):
            self.observation = np.concatenate((self.pose, self.state.info["commands"][0]))[None, :]


    class PlanarWrapper:
        def __init__(self, env, **kwargs):
            self.env = env

        def get_observations(self):
            self.env.refresh_state()
            return torch.from_numpy(self.env.observation.copy())

        def reset(self):
            self.env.pose[:] = 0.0
            self.env.steps = 0
            return self.get_observations(), {}

        def step(self, actions):
            env = self.env
            _advance_planar_pose(
                env.pose, env.state.info["commands"][0], env.cfg.ctrl_dt,
                speed_scale=0.85, yaw_bias=0.03,
            )
            env.steps += 1
            done = env.steps * env.cfg.ctrl_dt >= env.cfg.max_episode_seconds
            return self.get_observations(), torch.zeros(1), np.array([done]), {}

    cfg = OmegaConf.create({
        "training": {"task_name": "test"},
        "demo": {"figure_eight": {
            "enabled": True, "width_m": 4.0, "height_m": 2.0, "speed_m_s": 0.3,
            "segment_duration_s": 0.8,
        }},
        "video_recording": {"enabled": True, "duration_s": 220.0},
    })
    session, _, _ = create_fada_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="test", load_run="-1", checkpoint=None, action_mode="zero",
            policy_obs_mode="actor", algo_log_name="test", log_root=None,
        ),
        cfg=cfg, root_dir=".", device="cpu",
        deps={
            "resolve_checkpoint": lambda *args: None,
            "build_env_cfg_override": lambda cfg: {"max_episode_seconds": 180.0},
            "create_env": lambda cfg, **kwargs: PlanarEnv(kwargs["env_cfg_override"]),
            "wrapper_cls": PlanarWrapper,
        },
        log=lambda message: None,
    )
    timing = resolve_fada_recording_timing(
        session, FADAVideoRecordingConfig(duration_s=220.0),
    )
    assert timing.control_steps == 11000
    session.reset()
    steps = 5 * session.command_program.total_control_steps
    for _ in range(steps):
        session.step_once()
    assert session.command_control_steps[0] == steps
    np.testing.assert_allclose(
        session.env.state.info["commands"][0],
        session.command_program.command_at_control_step(steps - 1),
    )
