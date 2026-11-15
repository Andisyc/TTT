"""Finalized circle outputs retain failures, metric windows and numeric precision."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.circle_evaluation import evaluate_circle_shape
from unilab.visualization.circle_trial import finalize_circle_trial
from unilab.visualization.experiment_recording import ExperimentRecorder, ExperimentRecordingConfig


def _record_measurements(output: Path, linear_error: float, angular_error: float, *, terminated=False):
    physical = {"linear": np.zeros((1, 3)), "angular": np.zeros((1, 3))}
    env = SimpleNamespace(
        num_envs=1, cfg=SimpleNamespace(ctrl_dt=1.0),
        set_step_measurement_callback=lambda callback: None,
        get_base_quat=lambda: np.array([[1.0, 0.0, 0.0, 0.0]]),
        get_base_lin_vel=lambda: physical["linear"],
        get_base_ang_vel=lambda: physical["angular"],
    )
    recorder = ExperimentRecorder(
        env, ExperimentRecordingConfig(True, str(output / "raw"), 3),
        {"ctrl_dt": 1.0, "policy_joint_names": ["j0"], "environment_joint_names": ["j0"],
         "checkpoint_sha256": "base", "resolved_config": {"algo": {"seed": 0},
         "online_adaptation": {"enabled": True}}},
    )
    recorder.on_reset("explicit")
    action = np.zeros((1, 1))
    for index in range(6):
        recorder.begin_query(np.array([[0.4, 0.0, 0.2]]))
        recorder.submit_actions(action, action)
        # Large adaptation errors must not enter the measured segment [2, 6].
        physical["linear"] = np.array([[0.4 + (10.0 if index < 2 else linear_error), 0.0, 0.0]])
        physical["angular"] = np.array([[0.0, 0.0, 0.2 + (8.0 if index < 2 else angular_error)]])
        recorder.on_step_completed(SimpleNamespace(
            terminated=[terminated and index == 5], truncated=[False], info={"current_actions": action},
        ), None)
    recorder.close("completed")


@pytest.mark.parametrize("linear, angular, radial, radius, expected_suffix", [
    (0.012345, 1.23456, 0.000123456, 1.23456,
     "_E_shape_NA_E_lin_0.0123_E_ang_1.23_E_rad_0.000123_R_1.23"),
    (0.000012345, 0.0, 0.0, 2.0,
     "_E_shape_NA_E_lin_1.23e-05_E_ang_0.00_E_rad_0.00_R_2.00"),
])
def test_full_lap_directory_uses_three_significant_digits_but_json_keeps_precision(
    tmp_path, linear, angular, radial, radius, expected_suffix,
):
    _record_measurements(tmp_path, linear, angular)
    video = tmp_path / "trial.mp4"
    video.write_bytes(b"closed-video-fixture")
    result = finalize_circle_trial({
        "circle_start_step": 2, "ctrl_dt": 1.0, "duration_s": 6.0,
        "completed_lap": True, "stop_reason": "lap_completed",
        "shape": {"geometry_status": "measured", "radial_rmse_m": radial, "radius_m": radius},
    }, tmp_path)

    destination = tmp_path.with_name(tmp_path.name + expected_suffix)
    assert (destination / "trial.mp4").read_bytes() == b"closed-video-fixture"
    assert not video.exists()
    assert result["velocity_scope"] == "full_lap"
    assert result["output_dir"] == str(destination)
    assert result["video_path"] == str(destination / "trial.mp4")
    assert not tmp_path.exists()
    saved = json.loads((destination / "summary.json").read_text())
    assert saved["shape_comparison"]["status"] == "reference_required"
    assert saved["shape_comparison"]["E_shape_m"] is None
    assert Path(saved["video_path"]).is_file()
    assert saved["velocity"] == pytest.approx({"E_lin_m_s": linear, "E_ang_rad_s": angular})
    assert saved["shape"]["radial_rmse_m"] == radial
    assert saved["shape"]["radius_m"] == radius
    evaluation = json.loads((destination / "velocity/summary.json").read_text())
    assert evaluation["run_dir"] == str(destination / "raw")
    assert evaluation["options"]["run_dir"] == str(destination / "raw")
    assert evaluation["options"]["output_dir"] == str(destination / "velocity")
    assert (destination / "raw/run_config.json").is_file()
    assert (destination / "velocity/errors.csv").is_file()
    assert evaluation["requested_window_s"] == [2.0, 6.0]
    assert evaluation["episodes"][0]["samples"] == 4


def test_terminated_lap_keeps_terminal_velocity_but_does_not_invent_circle_geometry(tmp_path):
    _record_measurements(tmp_path, 0.123456, 0.00123456, terminated=True)
    (tmp_path / "trial.mp4").write_bytes(b"terminated-video-fixture")
    result = finalize_circle_trial({
        "circle_start_step": 2, "ctrl_dt": 1.0, "duration_s": 6.0,
        "completed_lap": False, "stop_reason": "terminated",
        "shape": {"geometry_status": "incomplete_lap"},
    }, tmp_path)

    destination = Path(result["output_dir"])
    assert destination.name == (
        tmp_path.name + "_incomplete_terminated_E_shape_NA_E_lin_0.123_E_ang_0.00123_E_rad_NA_R_NA"
    )
    assert Path(result["video_path"]).name == "trial.mp4"
    assert Path(result["video_path"]).read_bytes() == b"terminated-video-fixture"
    assert result["velocity_scope"] == "incomplete_lap"
    assert result["shape"] == {"geometry_status": "incomplete_lap"}
    evaluation = json.loads((destination / "velocity/summary.json").read_text())
    assert evaluation["episodes"][0]["terminated"] is True
    assert evaluation["episodes"][0]["samples"] == 4


def test_failure_before_measurement_saves_summary_without_a_fake_video_or_metrics(tmp_path):
    result = finalize_circle_trial({
        "circle_start_step": -1, "ctrl_dt": 1.0, "duration_s": 1.0,
        "completed_lap": False, "stop_reason": "terminated",
        "shape": {"geometry_status": "not_entered"},
    }, tmp_path)

    assert result["velocity"] is None
    assert result["velocity_scope"] == "not_entered"
    assert result["video_path"] is None
    destination = Path(result["output_dir"])
    assert destination.name.endswith("_incomplete_terminated_E_shape_NA_E_lin_NA_E_ang_NA_E_rad_NA_R_NA")
    assert json.loads((destination / "summary.json").read_text())["stop_reason"] == "terminated"
    assert not (destination / "velocity").exists()
    assert not list(destination.glob("*.mp4"))


def test_existing_metric_directory_is_not_overwritten(tmp_path):
    destination = tmp_path.with_name(
        tmp_path.name + "_incomplete_terminated_E_shape_NA_E_lin_NA_E_ang_NA_E_rad_NA_R_NA"
    )
    destination.mkdir()
    retained = destination / "trial.mp4"
    retained.write_bytes(b"previous-run")
    with pytest.raises(FileExistsError):
        finalize_circle_trial({
            "circle_start_step": -1, "ctrl_dt": 1.0, "duration_s": 1.0,
            "completed_lap": False, "stop_reason": "terminated",
            "shape": {"geometry_status": "not_entered"},
        }, tmp_path)
    assert retained.read_bytes() == b"previous-run"
    assert tmp_path.is_dir()
    assert not (tmp_path / "summary.json").exists()


def _circle_trajectory(
    directory, *, radius=1.0, yaw=0.0, shift=(0.0, 0.0), repeats=1,
    completed=True, payload=0.0, online=False, seed=0, checkpoint="base",
    approach_command=(0.4, 0.0, 0.2), dt=0.02, approach_steps=2, samples=2001,
    command=(0.4, 0.0, 0.2),
):
    directory.mkdir(parents=True, exist_ok=True)
    angle = np.linspace(0, 2 * np.pi, samples)
    xy = radius * np.column_stack((np.sin(angle), 1 - np.cos(angle)))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = np.repeat(xy @ rotation.T + shift, repeats, axis=0)
    xy = np.vstack((np.full((approach_steps, 2), 100.0), xy))
    heading = np.concatenate((np.zeros(approach_steps), np.repeat(angle + yaw, repeats)))
    quats = np.column_stack((np.cos(heading / 2), np.zeros((len(heading), 2)), np.sin(heading / 2)))
    path = directory / "trajectory.npz"
    np.savez_compressed(
        path, time_s=np.arange(len(xy)) * dt,
        base_positions=np.column_stack((xy, np.full(len(xy), 0.7))), base_quats=quats,
        command=np.array(command), ctrl_dt=dt,
        stop_reason="lap_completed" if completed else "terminated",
        circle_start_step=approach_steps, approach_steps=approach_steps, startup_steps=0,
        approach_command=np.array(approach_command), payload_mass_kg=np.full(len(xy), payload),
    )
    raw = directory / "raw"
    raw.mkdir(exist_ok=True)
    metadata = raw / "run_config.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({
            "checkpoint_sha256": checkpoint,
            "resolved_config": {"algo": {"seed": seed}, "online_adaptation": {"enabled": online}},
        }))
    return path


def test_shape_ignores_entry_pose_and_dwell_but_not_the_actual_outline(tmp_path):
    reference = _circle_trajectory(tmp_path / "reference")
    actual = _circle_trajectory(tmp_path / "actual", yaw=1.3, shift=(5.0, -2.0), repeats=3)
    comparison = evaluate_circle_shape(actual, reference)
    assert comparison["status"] == "scored"
    assert comparison["E_shape_m"] == pytest.approx(0.0, abs=1e-12)


def test_shape_preserves_radius_difference_instead_of_fitting_it_away(tmp_path):
    reference = _circle_trajectory(tmp_path / "reference", radius=1.0)
    actual = _circle_trajectory(tmp_path / "actual", radius=1.5)
    comparison = evaluate_circle_shape(actual, reference)
    # At equal angles, distance is 0.5 * sqrt(2 - 2*cos(angle)).
    expected = 0.5 * np.sqrt(2000 / 1001)
    assert comparison["E_shape_m"] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("changes, message", [
    ({"payload": 2.0}, "unloaded"),
    ({"online": True}, "frozen"),
    ({"seed": 1}, "seed"),
    ({"checkpoint": "other"}, "checkpoint"),
    ({"command": (0.4, 0.0, 0.3)}, "command"),
    ({"dt": 0.01}, "ctrl_dt"),
])
def test_shape_rejects_nonmatching_or_nonbaseline_reference(tmp_path, changes, message):
    actual = _circle_trajectory(tmp_path / "actual")
    reference = _circle_trajectory(tmp_path / "reference", **changes)
    with pytest.raises(ValueError, match=message):
        evaluate_circle_shape(actual, reference)


def test_shape_records_prefix_differences_without_rejecting_an_explicit_reference(tmp_path):
    actual = _circle_trajectory(tmp_path / "actual", approach_steps=3)
    reference = _circle_trajectory(tmp_path / "reference", approach_command=(0.4, 0.0, 0.0))
    comparison = evaluate_circle_shape(actual, reference)
    assert comparison["status"] == "scored"
    assert comparison["E_shape_m"] == pytest.approx(0.0, abs=1e-12)
    assert comparison["adaptation_protocol_differences"] == {
        "approach_steps": {"trial": 3, "reference": 2},
        "approach_command": {"trial": [0.4, 0.0, 0.2], "reference": [0.4, 0.0, 0.0]},
    }


def test_shape_does_not_resample_a_failed_trial_as_a_complete_lap(tmp_path):
    actual = _circle_trajectory(tmp_path / "actual", completed=False)
    reference = _circle_trajectory(tmp_path / "reference")
    comparison = evaluate_circle_shape(actual, reference)
    assert comparison["status"] == "DNF"
    assert comparison["E_shape_m"] is None


def test_shape_is_first_in_final_directory_and_keeps_existing_metrics(tmp_path):
    output = tmp_path / "run"
    _record_measurements(output, 0.012345, 1.23456)
    _circle_trajectory(output, dt=1.0, samples=5)
    reference = _circle_trajectory(tmp_path / "reference", dt=1.0, samples=5)
    (output / "trial.mp4").write_bytes(b"closed-video-fixture")
    result = finalize_circle_trial({
        "circle_start_step": 2, "ctrl_dt": 1.0, "duration_s": 6.0,
        "completed_lap": True, "stop_reason": "lap_completed",
        "shape": {"radial_rmse_m": 0.000123456, "radius_m": 1.23456},
    }, output, reference=reference)
    assert Path(result["output_dir"]).name == (
        "run_E_shape_0.00_E_lin_0.0123_E_ang_1.23_E_rad_0.000123_R_1.23"
    )
    saved = json.loads((Path(result["output_dir"]) / "summary.json").read_text())
    assert saved["shape_comparison"]["E_shape_m"] == pytest.approx(0.0)
    assert saved["shape_comparison"]["reference"] == str(reference.resolve())
    assert Path(saved["video_path"]).is_file()


def test_shape_does_not_give_stationary_spinning_a_perfect_score(tmp_path):
    actual = _circle_trajectory(tmp_path / "actual", radius=0.0)
    reference = _circle_trajectory(tmp_path / "reference")
    comparison = evaluate_circle_shape(actual, reference)
    assert comparison["status"] == "not_evaluable"
    assert comparison["E_shape_m"] is None
