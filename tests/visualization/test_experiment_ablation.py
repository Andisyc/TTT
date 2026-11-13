"""Offline regression checks for batch identity, resume and fall accounting."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from unilab.visualization import experiment_ablation as batch
from unilab.visualization.experiment_common import GROUP_MODES

ROOT = Path(__file__).resolve().parents[2]


def _config() -> DictConfig:
    config = OmegaConf.load(ROOT / "conf/distill/oat_component_ablation.yaml")
    assert isinstance(config, DictConfig)
    return config


def _finish(trial: dict, *, fall: bool = False, wrong_seed: bool = False) -> None:
    path = Path(trial["output_dir"])
    raw, evaluation = path / "raw", path / "evaluation"
    raw.mkdir(parents=True)
    evaluation.mkdir()
    run_id = str(path)
    actual = dict(trial)
    if wrong_seed:
        actual["seed"] += 1
    metadata = {"status": "completed", "run_id": run_id, "resolved_config": {
        "experiment_execution": actual,
        "online_adaptation": {**trial["online"], "mode": GROUP_MODES[trial["group"]]},
    }}
    (raw / "run_config.json").write_text(json.dumps(metadata))
    events = [{"kind": "terminated", "run_id": run_id, "fall_terminated": True}] if fall else []
    (raw / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    summary = {"run_id": run_id, "episodes": [{
        "full_requested_window": not fall,
        "velocity_observed_segment": {"E_lin_m_s": 0.4 if fall else 0.1, "E_ang_rad_s": 0.2},
        "attitude": {"roll": 4.0 if fall else 1.0, "pitch": 0.5, "aggregation": "rms", "unit": "deg"},
        "action_change": {"mean_squared_change": 0.01},
    }]}
    (evaluation / "summary.json").write_text(json.dumps(summary))
    (path / "trial.mp4").touch()
    row = {"group": trial["group"], "seed": trial["seed"], "raw_dir": str(raw),
           "evaluation": "completed", "evaluation_dir": str(evaluation),
           "stop_reason": "terminated" if fall else "duration_reached",
           "actual_duration_s": 15.0 if fall else trial["duration_s"]}
    (path / "trial.json").write_text(json.dumps(row))


def test_resume_skips_finalized_trials_and_counts_falls_without_zero_filling(tmp_path, monkeypatch):
    config = _config()
    config.methods = ["fada_ttt", "full_oat"]
    config.payloads_kg = [2]
    config.repeats = 2
    output = tmp_path / "batch"
    trials = batch.plan_ablation_trials(config, output_dir=output)
    calls = []

    def child(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        calls.append(request)
        _finish(request, fall=request["group"] == "fada_ttt" and request["seed"] == 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(batch.subprocess, "run", child)
    batch.run_ablation_batch(trials, output_dir=output, root_dir=ROOT)
    assert len(calls) == 4
    summary = json.loads((output / "summary.json").read_text())
    baseline, oat = summary["points"]
    assert baseline["survival"]["percent"] == 50.0
    assert baseline["incomplete_windows"] == 1
    assert baseline["metrics"]["roll_rms_deg"]["n"] == 1
    assert baseline["metrics"]["roll_rms_deg"]["mean"] == 1.0
    assert oat["survival"]["percent"] == 100.0
    assert oat["normalization"]["status"] == "not_configured"
    assert "normalized_E_v" not in oat["metrics"]
    batch.run_ablation_batch(trials, output_dir=output, root_dir=ROOT, resume=True)
    assert len(calls) == 4
    changed = json.loads(json.dumps(trials))
    changed[0]["online"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="identical saved batch"):
        batch.run_ablation_batch(changed, output_dir=output, root_dir=ROOT, resume=True)
    assert len(calls) == 4


def test_resume_preflights_partial_outputs_before_starting_missing_trials(tmp_path, monkeypatch):
    config = _config()
    config.methods = ["full_oat"]
    config.payloads_kg = [2]
    config.repeats = 2
    output = tmp_path / "batch"
    trials = batch.plan_ablation_trials(config, output_dir=output)
    output.mkdir()
    (output / "batch_config.json").write_text(json.dumps({"trials": trials}))
    Path(trials[1]["output_dir"]).mkdir(parents=True)

    def unexpected(*args, **kwargs):
        pytest.fail("A partial output must block the batch before launching a worker")

    monkeypatch.setattr(batch.subprocess, "run", unexpected)
    with pytest.raises(FileExistsError, match="Incomplete trial"):
        batch.run_ablation_batch(trials, output_dir=output, root_dir=ROOT, resume=True)


def test_resume_rejects_renamed_seed_identity(tmp_path):
    config = _config()
    config.methods = ["full_oat"]
    config.payloads_kg = [2]
    config.repeats = 1
    trial, = batch.plan_ablation_trials(config, output_dir=tmp_path / "batch")
    _finish(trial, wrong_seed=True)
    with pytest.raises(ValueError, match="protocol differs"):
        batch._load_completed_trial(trial)


@pytest.mark.parametrize("events,stop,elapsed,expected", [
    ([], "duration_reached", 30, True),
    ([], "truncated", 30, True),
    ([{"kind": "terminated", "fall_terminated": True}], "terminated", 15, False),
    ([{"kind": "terminated", "fall_terminated": False, "forward_progress_terminated": True}], "terminated", 15, None),
    ([{"kind": "terminated"}], "terminated", 15, None),
    ([], "truncated", 15, None),
])
def test_survival_distinguishes_falls_time_limits_and_unknown_causes(events, stop, elapsed, expected):
    result = batch.classify_survival(events, {"stop_reason": stop, "actual_duration_s": elapsed}, 30)
    assert result["survived"] is expected
    assert result["status"] == ("not_evaluable" if expected is None else "computed")
