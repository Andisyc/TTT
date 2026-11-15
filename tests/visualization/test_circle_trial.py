"""Circle adaptation follows the measured motion without consuming its lap budget."""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from unilab.visualization.circle_trial import run_circle_trial

if TYPE_CHECKING:
    from unilab.visualization.playback_sessions import FADAPlaybackSession


class _PlanarSession:
    """Integrate commanded body-frame motion without a policy or physics backend."""

    def __init__(self):
        self.pose = np.zeros(3)
        self.command = np.zeros(3)
        self.command_program = None
        self.command_control_steps = np.zeros(1, dtype=np.int64)
        self.controller = None
        self.footprint_recorder = None
        self.experiment_recorder = None
        self.env = SimpleNamespace(
            cfg=SimpleNamespace(
                ctrl_dt=0.05,
                commands=SimpleNamespace(
                    resampling_time=0.0, heading_command=False,
                    vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                    dead_zone_enabled=False, small_xy_threshold=0.0,
                ),
                runtime_payload=SimpleNamespace(enabled=False),
            ),
            state=SimpleNamespace(terminated=[False], truncated=[False]),
            get_base_pos=lambda: np.array([[self.pose[0], self.pose[1], 0.7]]),
            get_base_quat=lambda: np.array([
                [np.cos(self.pose[2] / 2), 0.0, 0.0, np.sin(self.pose[2] / 2)],
            ]),
        )

    def set_autoreset(self, enabled):
        self.autoreset = enabled

    def reset(self):
        self.pose.fill(0)
        self.command_control_steps.fill(0)

    def set_external_command(self, command):
        self.command = np.asarray(command).copy()

    def step_once(self):
        command = self.command
        if self.command_program is not None:
            command = self.command_program.command_at_control_step(int(self.command_control_steps[0]))
        dt = self.env.cfg.ctrl_dt
        turn = command[2] * dt
        heading = self.pose[2] + turn / 2
        self.pose[:2] += command[0] * dt * np.sinc(turn / (2 * np.pi)) * np.array([
            np.cos(heading), np.sin(heading),
        ])
        self.pose[2] += turn
        self.command_control_steps += 1


@pytest.mark.parametrize("yaw_rate", [-0.6, 0.6])
@pytest.mark.parametrize("approach_s", [0.0, 11.0])
@pytest.mark.parametrize("test_speed", [None, 0.6])
def test_circle_switches_to_test_speed_without_consuming_its_lap_budget(tmp_path, yaw_rate, approach_s, test_speed):
    session = _PlanarSession()
    captured_steps = []

    @contextmanager
    def capture_trial():
        yield lambda step, **kwargs: captured_steps.append(step)

    trajectory = tmp_path / "trajectory.npz"
    result = run_circle_trial(
        cast("FADAPlaybackSession", session), forward_m_s=0.4, yaw_rate_rad_s=yaw_rate,
        approach_s=approach_s, duration_s=15.0, test_forward_m_s=test_speed,
        trajectory_path=trajectory, capture_trial=capture_trial,
    )
    dt = session.env.cfg.ctrl_dt
    entry = round(approach_s / dt)
    lap_steps = math.ceil(math.tau / (abs(yaw_rate) * dt))
    assert result["completed_lap"]
    assert result["circle_start_step"] == entry
    assert result["duration_s"] == pytest.approx((entry + lap_steps) * dt)
    assert result["directed_yaw_rad"] == pytest.approx(lap_steps * abs(yaw_rate) * dt)
    measured_speed = 0.4 if test_speed is None else test_speed
    assert result["shape"]["radius_m"] == pytest.approx(measured_speed / abs(yaw_rate), abs=1e-6)
    assert captured_steps == list(range(lap_steps + 1))
    with np.load(trajectory) as recorded:
        np.testing.assert_array_equal(recorded["command"], [measured_speed, 0.0, yaw_rate])
        np.testing.assert_array_equal(recorded["approach_command"], [0.4, 0.0, yaw_rate] if entry else [])
        quats = recorded["base_quats"]
        yaw = np.unwrap(2 * np.arctan2(quats[:, 3], quats[:, 0]))
        assert yaw[entry] == pytest.approx(yaw_rate * approach_s)
        radius = 0.4 / yaw_rate
        expected_entry = [radius * np.sin(yaw_rate * approach_s), radius * (1 - np.cos(yaw_rate * approach_s))]
        np.testing.assert_allclose(recorded["base_positions"][entry, :2], expected_entry, atol=1e-12)
