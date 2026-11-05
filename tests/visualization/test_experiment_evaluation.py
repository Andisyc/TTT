"""Consumer-visible offline metrics and an actual evaluation CLI smoke run."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.common.rotation import np_quat_apply, np_quat_from_euler_xyz
from unilab.visualization.experiment_evaluation import evaluate_experiment, read_recorded_run
from unilab.visualization.experiment_recording import ExperimentRecorder, ExperimentRecordingConfig


@pytest.fixture
def recorded(tmp_path):
    def make(name="run", *, velocity=(1.0, 2.0, 0.0), angular=0.1, terminate=False, initial_rest=False):
        quat = np_quat_from_euler_xyz(*[np.array([np.deg2rad(value)]) for value in (20.0, -10.0, 90.0)]).reshape(1, 4)
        linear = np_quat_apply(quat, np.array([velocity]))
        omega = np_quat_apply(quat, np.array([[0.0, 0.0, angular]]))
        current = {"linear": np.zeros((1, 3)) if initial_rest else linear.copy()}
        payload = SimpleNamespace(body_name="shoulder", attachment_pos_m=[0, 0.2, 0], half_size_m=[0.1] * 3)
        env = SimpleNamespace(num_envs=1, cfg=SimpleNamespace(ctrl_dt=1.0, runtime_payload=payload),
                              set_step_measurement_callback=lambda callback: None,
                              get_base_quat=lambda: quat, get_base_lin_vel=lambda: current["linear"],
                              get_base_ang_vel=lambda: omega)
        metadata = {"ctrl_dt": 1.0, "policy_joint_names": ["j0", "j1"], "environment_joint_names": ["j0", "j1"],
                    "effective_env_config": {"runtime_payload": vars(payload)}, "policy_family": "legacy",
                    "resolved_config": {"algo": {"seed": 7}}, "checkpoint_path": "fixture.pt", "checkpoint_sha256": "fixture"}
        path = tmp_path / name
        recorder = ExperimentRecorder(env, ExperimentRecordingConfig(True, str(path), 2), metadata)
        recorder.on_reset("explicit")
        for index in range(4):
            if terminate and index == 2:
                recorder.on_reset("autoreset")
            recorder.begin_query(np.array([[4.0, 6.0, 0.6]]))
            action = np.array([[float(index), float(index * 2)]])
            recorder.submit_actions(action, action + 0.2)
            current["linear"] = linear.copy()
            state = SimpleNamespace(terminated=np.array([terminate and index == 1]), truncated=np.array([False]),
                                    info={"current_actions": action + 0.4,
                                          "runtime_payload_attached_this_step": np.array([index == 1]),
                                          "runtime_payload_added_mass_kg": np.array([3.0 if index >= 1 else 0.0])})
            recorder.on_step_completed(state, None)
            if index == 2:
                recorder.on_update({"loss": 0.123, "version": 1})
        recorder.close("completed")
        return path
    return make


def options(tmp_path, run, name="evaluation", **changes):
    return {"run_dir": str(run), "output_dir": str(tmp_path / name), "linear_frame": "body_xy",
            "sample_phase": "completed", "relative_to": "run_start", "start_s": 0.0, "end_s": 4.0, **changes}


def test_actual_cli_reads_chunks_and_writes_known_metrics(recorded, tmp_path):
    run = recorded()
    output = tmp_path / "cli"
    command = [sys.executable, "scripts/evaluate_oat_experiment.py", f"run_dir={run}", f"output_dir={output}",
               "linear_frame=body_xy", "sample_phase=completed", "relative_to=run_start", "start_s=0", "end_s=4",
               "attitude_reference=[0,0]", "attitude_unit=deg", "attitude_aggregation=rms",
               "action_source=policy_action", "action_joint_reduction=mean", "curve_window_s=2"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    episode = summary["episodes"][0]
    assert episode["velocity_observed_segment"] == pytest.approx({"E_lin_m_s": 5.0, "E_ang_rad_s": 0.5})
    assert episode["attitude"]["roll"] == pytest.approx(20.0)
    assert episode["attitude"]["pitch"] == pytest.approx(10.0)
    assert episode["action_change"]["mean_squared_change"] == pytest.approx(2.5)
    assert episode["full_requested_window"] is True
    assert [item["E_lin_m_s"] for item in episode["time_windows"]] == pytest.approx([5.0, 5.0])
    assert summary["selected_update_count"] == 1
    assert summary["survival"]["status"] == "not_evaluable"
    assert summary["normalization"]["status"] == "not_evaluated"
    with (output / "errors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["linear_squared_error_m2_s2"]) for row in rows] == pytest.approx([25.0] * 4)
    assert [float(row["time_s"]) for row in rows] == [1, 2, 3, 4]


def test_sampling_phase_and_actual_payload_origin(recorded, tmp_path):
    run = recorded(initial_rest=True)
    query = evaluate_experiment(options(tmp_path, run, "query", sample_phase="query"))
    completed = evaluate_experiment(options(tmp_path, run, "completed"))
    assert query["episodes"][0]["velocity_observed_segment"]["E_lin_m_s"] == pytest.approx(np.sqrt(127 / 4))
    assert completed["episodes"][0]["velocity_observed_segment"]["E_lin_m_s"] == pytest.approx(5.0)
    payload = evaluate_experiment(options(tmp_path, run, "payload", relative_to="payload", payload_event_index=0,
                                          start_s=0, end_s=2, curve_window_s=1))
    assert payload["requested_window_s"] == [1.0, 3.0]
    assert payload["episodes"][0]["samples"] == 2
    assert payload["selected_update_count"] == 1
    yaw = evaluate_experiment(options(tmp_path, run, "yaw", linear_frame="yaw_frame_xy"))
    assert yaw["episodes"][0]["velocity_observed_segment"]["E_lin_m_s"] != pytest.approx(5.0)


def test_resets_and_early_end_remain_visible(recorded, tmp_path):
    run = recorded(terminate=True)
    summary = evaluate_experiment(options(tmp_path, run, curve_window_s=2,
                                          action_source="policy_action", action_joint_reduction="mean"))
    assert [episode["episode_id"] for episode in summary["episodes"]] == [0, 1]
    assert [episode["terminated"] for episode in summary["episodes"]] == [True, False]
    assert not any(episode["full_requested_window"] for episode in summary["episodes"])
    assert [episode["action_change"]["mean_squared_change"] for episode in summary["episodes"]] == [2.5, 2.5]
    # Retains the terminal sample rather than reporting a deceptively shorter trajectory.
    with (tmp_path / "evaluation" / "errors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["terminated"] == "True"
    short = evaluate_experiment(options(tmp_path, run, "short", episode_id=0, end_s=6))
    assert short["episodes"][0]["covered_duration_s"] == 2.0
    assert short["episodes"][0]["full_requested_window"] is False


def test_explicit_common_denominator_and_incompatible_reference(recorded, tmp_path):
    run = recorded()
    reference = recorded("reference", velocity=(0.0, 0.0, 0.0), angular=0.0)
    summary = evaluate_experiment(options(tmp_path, run, reference_run_dir=str(reference)))
    assert summary["episodes"][0]["normalized_E_v"] == pytest.approx(5.5 / (np.sqrt(52) + 0.6))
    metadata_path = reference / "run_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["effective_env_config"]["runtime_payload"]["body_name"] = "different_shoulder"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible effective_env_config"):
        evaluate_experiment(options(tmp_path, run, "bad_reference", reference_run_dir=str(reference)))
    assert not (tmp_path / "bad_reference").exists()


def test_missing_chunks_and_unselected_conventions_are_errors(recorded, tmp_path):
    run = recorded()
    with pytest.raises(ValueError, match="linear_frame"):
        evaluate_experiment(options(tmp_path, run, linear_frame=None))
    with pytest.raises(ValueError, match="attitude requires"):
        evaluate_experiment(options(tmp_path, run, attitude_reference=[0, 0]))
    (run / "measurements" / "part_000000.npz").unlink()
    with pytest.raises(ValueError, match="missing or misordered"):
        read_recorded_run(run)
