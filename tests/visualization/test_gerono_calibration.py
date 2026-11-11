"""Offline measured-response, bounded inversion, and frozen playback behavior."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.demo_commands import (
    CalibratedFigureEightCommandProgram, FigureEightCommandConfig, FigureEightCommandProgram,
)
from unilab.visualization.gerono_calibration import (
    ResponseModel, ResponseTrace, body_twists, build_calibrated_program,
    load_measurements, save_measurements, save_program,
)
from unilab.visualization.gerono_measurement import TrialTerminatedError, collect_response_trial

DT = 0.02
GEOMETRY = dict(width_m=0.4, height_m=0.2, speed_m_s=0.08, segment_duration_s=0.8)
IDENTITY = {"ctrl_dt": DT, "geometry": GEOMETRY, "checkpoint_sha256": "synthetic-only"}


def _actual_twists(commands):
    velocity, rows = np.zeros(3), []
    for v, _, w in commands:
        steady = np.array([0.82*v + 0.02*v*abs(w), 0.005*w,
                           (0.78 if w >= 0 else 0.69)*w + 0.04*v*w])
        decay = np.array([0.7, 0.6, 0.8])
        velocity = decay*velocity + (1-decay)*steady
        rows.append(velocity.copy())
    return np.array(rows)


def _poses(twists, origin=(0.0, 0.0, 0.0)):
    pose, rows = np.array(origin, dtype=float), [np.array(origin, dtype=float)]
    for vx, vy, wz in twists:
        turn = wz*DT
        middle = pose[2] + turn/2
        pose[:2] += DT*np.sinc(turn/(2*np.pi))*np.array([
            np.cos(middle)*vx - np.sin(middle)*vy,
            np.sin(middle)*vx + np.cos(middle)*vy,
        ])
        pose[2] += turn
        rows.append(pose.copy())
    return np.array(rows)


def _trial(commands):
    return ResponseTrace(commands, _poses(_actual_twists(commands)), DT)


@pytest.fixture
def response_trials():
    trials = [_trial(np.tile([v, 0, w], (80, 1)))
              for v in (0.01, 0.12, 0.3) for w in (-0.8, -0.4, 0, 0.4, 0.8)]
    holdout = _trial(np.repeat([[0.12, 0, w] for w in (0, 0.4, 0.8, -0.8, -0.4)], 35, axis=0))
    return trials, holdout


def test_body_twists_recover_lateral_motion_and_wrapped_heading():
    actual = np.tile([0.21, -0.04, 0.7], (100, 1))
    poses = _poses(actual, origin=(2, -3, 2.8))
    poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))
    recovered = body_twists(ResponseTrace(np.tile([0.3, 0, 0.8], (100, 1)), poses, DT))
    np.testing.assert_allclose(recovered, actual, atol=2e-13)


def test_asymmetric_dynamic_response_predicts_independent_transitions(response_trials):
    trials, holdout = response_trials
    model = ResponseModel.fit(trials)
    np.testing.assert_allclose(model.decay, [0.7, 0.6, 0.8], atol=1e-9)
    np.testing.assert_allclose(model.predict(holdout.commands), _actual_twists(holdout.commands), atol=1e-10)
    np.testing.assert_allclose(model.validation_rmse([holdout]), 0, atol=1e-10)


def test_response_fit_refuses_one_direction_only(response_trials):
    trials, _ = response_trials
    with pytest.raises(ValueError, match="Unidentifiable"):
        ResponseModel.fit([trial for trial in trials if np.all(trial.commands[:, 2] >= 0)])


def test_measurement_roundtrip_preserves_trial_reset_boundaries(tmp_path, response_trials):
    trials, _ = response_trials
    path = tmp_path / "response.npz"
    save_measurements(path, trials[:2], {"complete": True})
    loaded, metadata = load_measurements(path)
    assert metadata["complete"] is True
    for original, restored in zip(trials[:2], loaded, strict=True):
        np.testing.assert_allclose(body_twists(restored), _actual_twists(original.commands), atol=1e-10)
    with pytest.raises(FileExistsError):
        save_measurements(path, trials[:2], {})


@pytest.fixture
def calibrated_file(tmp_path, response_trials):
    trials, holdout = response_trials
    source = tmp_path / "measurements.npz"
    output = tmp_path / "calibrated.npz"
    save_measurements(source, [*trials, holdout], {
        "complete": True, "identity": IDENTITY, "train_trial_count": len(trials),
        "min_segment_steps": 1, "velocity_limits": [[0.0, -0.4, -0.8], [0.3, 0.4, 0.8]],
    })
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        "uv", "run", "--frozen", "--no-sync", "--offline", "python",
        str(root / "scripts/calibrate_fada_gerono.py"), "recording_profile=figure_eight",
        "gerono_calibration.stage=fit", f"gerono_calibration.measurements_path={source}",
        f"gerono_calibration.program_path={output}", "gerono_calibration.time_scales=[1.5]",
        "gerono_calibration.max_nfev=12", "gerono_calibration.position_tolerance_m=0.04",
        "gerono_calibration.heading_tolerance_deg=4.0",
    ], cwd=root, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_offline_cli_corrects_known_response_and_frozen_playback_needs_no_pose(calibrated_file):
    import torch
    from unilab.visualization.playback_sessions import FADAPlaybackSession

    program = CalibratedFigureEightCommandProgram.from_file(calibrated_file, identity=IDENTITY, min_segment_steps=1)
    reference = FigureEightCommandProgram.from_config(FigureEightCommandConfig(**GEOMETRY), ctrl_dt=DT)

    class Plant:
        def __init__(self):
            self.cfg = SimpleNamespace(ctrl_dt=DT)
            self.action_space = SimpleNamespace(shape=(1,))
            self.state = SimpleNamespace(info={"commands": np.zeros((1, 3))})
            self.pose, self.velocity = np.zeros(3), np.zeros(3)

        def refresh_state(self):
            pass

    class Wrapper:
        def __init__(self, plant):
            self.plant = plant

        def get_observations(self):
            return torch.zeros((1, 1))

        def reset(self):
            self.plant.pose[:] = 0
            self.plant.velocity[:] = 0
            return self.get_observations(), {}

        def step(self, actions):
            v, _, w = self.plant.state.info["commands"][0]
            target = np.array([0.82*v+0.02*v*abs(w), 0.005*w,
                               (0.78 if w >= 0 else 0.69)*w+0.04*v*w])
            decay = np.array([0.7, 0.6, 0.8])
            self.plant.velocity = decay*self.plant.velocity + (1-decay)*target
            self.plant.pose = _poses(self.plant.velocity[None, :], self.plant.pose)[-1]
            return self.get_observations(), torch.zeros(1), np.array([False]), {}

    plant = Plant()  # Deliberately provides no position/quaternion getter to playback.
    session = FADAPlaybackSession(env=plant, wrapped_env=Wrapper(plant), device="cpu",
                                  action_mode="zero", num_envs=1, controller=None, command_program=program)
    session.reset()
    boundaries = set(np.cumsum(np.tile(program.hold_steps, 3)))
    observed = []
    for step in range(3 * program.total_control_steps):
        session.step_once()
        if step+1 in boundaries:
            observed.append(plant.pose.copy())
    desired_yaw = np.cumsum(reference.commands[:, 2]*reference.segment_duration_s)
    desired = np.tile(np.column_stack((reference.reference_points[1:], desired_yaw)), (3, 1))
    error = np.array(observed)-desired
    assert np.max(np.linalg.norm(error[:, :2], axis=1)) < 0.04
    assert np.max(np.abs(error[:, 2])) < np.deg2rad(4)
    raw = np.repeat(np.tile(reference.commands, (3, 1)), reference.segment_steps, axis=0)
    assert np.linalg.norm(_poses(_actual_twists(raw))[-1, :2]) > 0.1
    assert session.command_control_steps[0] == 3 * program.total_control_steps


@pytest.mark.parametrize("fault", ["identity", "timestep", "window"])
def test_frozen_table_rejects_incompatible_identity_time_and_windows(tmp_path, fault):
    reference = FigureEightCommandProgram.from_config(FigureEightCommandConfig(**GEOMETRY), ctrl_dt=DT)
    path = tmp_path / "program.npz"
    save_program(path, reference.commands, np.full(len(reference.commands), 40), reference.reference_points,
                 DT*2 if fault == "timestep" else DT, {"identity": IDENTITY})
    expected = dict(IDENTITY, checkpoint_sha256="wrong") if fault == "identity" else IDENTITY
    with pytest.raises(ValueError):
        CalibratedFigureEightCommandProgram.from_file(path, identity=expected,
                                                     min_segment_steps=41 if fault == "window" else 1)


def test_bounded_calibration_refuses_unreachable_turning_domain(response_trials):
    model = ResponseModel.fit(response_trials[0])
    reference = FigureEightCommandProgram.from_config(FigureEightCommandConfig(**GEOMETRY), ctrl_dt=DT)
    with pytest.raises(ValueError, match="both turn directions"):
        build_calibrated_program(model, reference, [[0, 0, 0.1], [0.3, 0, 0.8]], min_segment_steps=1,
                                 time_scales=[1], max_nfev=2, position_tolerance_m=0.04,
                                 heading_tolerance_rad=0.1)


def test_measurement_preserves_terminal_step_without_autoreset():
    command_cfg = SimpleNamespace(resampling_time=0, heading_command=False,
                                  vel_limit=np.array([[0, -1, -1], [1, 1, 1]]),
                                  dead_zone_enabled=False, small_xy_threshold=0)

    class Session:
        def __init__(self):
            self.env = self
            self.cfg = SimpleNamespace(ctrl_dt=DT, commands=command_cfg)
            self.info = {"commands": np.zeros((1, 3))}
            self.state = SimpleNamespace(terminated=np.array([False]), truncated=np.array([False]))
            self.steps = self.resets = 0

        def reset(self):
            self.resets += 1
            self.steps = 0

        def set_external_command(self, command):
            self.info["commands"][0] = command

        def step_once(self):
            self.steps += 1
            self.state.terminated[:] = self.steps == 2

        def get_base_pos(self):
            return np.array([[self.steps*0.01, 0, 0.8]])

        def get_base_quat(self):
            return np.array([[1, 0, 0, 0]])

    session = Session()
    with pytest.raises(TrialTerminatedError) as caught:
        collect_response_trial(session, np.tile([0.3, 0, 0.2], (10, 1)))
    assert session.steps == 2 and session.resets == 1
    np.testing.assert_allclose(caught.value.trace.poses[:, 0], [0, 0.01, 0.02])
    assert caught.value.trace.commands.shape == (2, 3)
