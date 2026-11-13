"""Table 3 batches: existing trial semantics, one fresh process per trial."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.visualization.experiment_common import GROUP_MODES, _read_trial_metrics, _summarize, _write_json

METHODS = ("fada_ttt", "online_reference", "self_supervised_correction", "full_oat")


def plan_ablation_trials(config: DictConfig, *, output_dir: Path) -> list[dict[str, Any]]:
    """Materialize explicit method/mass/seed identities without creating files."""
    methods = list(config.methods)
    masses = [float(value) for value in config.payloads_kg]
    repeats, first_seed = config.repeats, config.first_seed
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError(f"methods must be distinct members of {METHODS}")
    if not masses or len(set(masses)) != len(masses) or any(not math.isfinite(m) or m <= 0 for m in masses):
        raise ValueError("payloads_kg must contain distinct finite positive masses")
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1
            or isinstance(first_seed, bool) or not isinstance(first_seed, int)
            or first_seed < 0 or first_seed + repeats > 2**32):
        raise ValueError("repeats and first_seed must define a nonempty valid seed range")
    protocol = OmegaConf.to_container(config.protocol, resolve=True, throw_on_missing=True)
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an explicit experiment mapping")
    payload = protocol["payload"]
    if payload.get("enabled") is not True or payload.get("additional_mass_events"):
        raise ValueError("component ablations require one enabled payload attachment")
    evaluation = protocol["evaluation"]
    if (evaluation.get("enabled") is not True or evaluation.get("relative_to") != "payload"
            or evaluation.get("attitude_reference") != [0.0, 0.0]
            or evaluation.get("attitude_unit") != "deg" or evaluation.get("attitude_aggregation") != "rms"):
        raise ValueError("ablations require payload-relative evaluation and upright roll RMS in degrees")
    dt = float(protocol["timing_ctrl_dt_s"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("timing_ctrl_dt_s must be finite and positive")
    duration, attach = float(protocol["duration_s"]), float(payload["attach_time_s"])
    start, end = float(evaluation["start_s"]), float(evaluation["end_s"])
    if not all(math.isfinite(x) for x in (duration, attach, start, end)) or not (0 <= attach < duration and 0 <= start < end <= duration - attach + 1e-9):
        raise ValueError("evaluation must fit entirely after attachment and within the run")
    if int(protocol["online"]["warmup_steps"]) <= math.ceil(attach / dt):
        raise ValueError("warmup_steps must prevent online updates before attachment")
    trials = []
    for method in methods:
        for mass in masses:
            for seed in range(first_seed, first_seed + repeats):
                path = output_dir.resolve() / method / f"payload={mass:g}" / f"seed_{seed}"
                trial = OmegaConf.merge(protocol, {
                    "group": method, "seed": seed, "output_dir": str(path),
                    "payload": {"added_mass_kg": mass},
                    "video": {"enabled": True, "label": f"{method} | {mass:g}kg"},
                    "evaluation": {"enabled": True, "allow_incomplete_window": True},
                })
                if method == "fada_ttt":
                    trial.online.sigma = 0.0
                reference = evaluation.get("reference_run_dir")
                if reference:
                    trial.evaluation.reference_run_dir = str(Path(str(reference).format(
                        seed=seed, payload_kg=f"{mass:g}",
                    )).expanduser().resolve())
                trials.append(OmegaConf.to_container(trial, resolve=True, throw_on_missing=True))
    return trials


def classify_survival(events: list[dict[str, Any]], result: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Use explicit environment fall provenance, never generic done flags."""
    terminal = [event for event in events if event["kind"] == "terminated"]
    if any(event.get("fall_terminated") is True for event in terminal):
        return {"status": "computed", "survived": False, "reason": "environment_fall_predicate"}
    if terminal:
        return {"status": "not_evaluable", "survived": None, "reason": "nonfall_or_unclassified_termination"}
    if (result["stop_reason"] in ("duration_reached", "truncated")
            and math.isclose(float(result["actual_duration_s"]), duration_s, rel_tol=0, abs_tol=1e-8)):
        return {"status": "computed", "survived": True, "reason": "full_horizon_without_termination"}
    return {"status": "not_evaluable", "survived": None, "reason": "incomplete_horizon"}


def run_ablation_trial(trial: dict[str, Any], *, root_dir: Path) -> dict[str, Any]:
    """Child-process boundary; keep simulation and optimizer ownership unchanged."""
    from unilab.visualization.experiment_execution import run_oat_experiment

    result = run_oat_experiment(trial, root_dir=root_dir)
    full, metrics = _read_trial_metrics(result)
    raw = Path(result["raw_dir"])
    events = [json.loads(line) for line in (raw / "events.jsonl").read_text().splitlines() if line.strip()]
    result.update(full_evaluation_window=full, metrics=metrics,
                  survival=classify_survival(events, result, float(trial["duration_s"])))
    _write_json(Path(trial["output_dir"]) / "trial.json", result)
    return result


def _load_completed_trial(trial: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(trial["output_dir"])
    if not path.exists():
        return None
    required = (path / "trial.json", path / "raw" / "run_config.json", path / "raw" / "events.jsonl")
    if any(not file.is_file() for file in required):
        raise FileExistsError(f"Incomplete trial, refusing overwrite: {path}. Preserve it and use a new batch directory.")
    row = json.loads(required[0].read_text())
    metadata = json.loads(required[1].read_text())
    actual = metadata["resolved_config"]["experiment_execution"]
    if any(actual.get(key) != value for key, value in trial.items()):
        raise ValueError(f"Saved trial protocol differs from requested protocol: {path}")
    online = metadata["resolved_config"]["online_adaptation"]
    if (metadata.get("status") != "completed" or row.get("group") != trial["group"]
            or row.get("seed") != trial["seed"] or online.get("mode") != GROUP_MODES[trial["group"]]
            or online.get("sigma") != trial["online"]["sigma"]):
        raise ValueError(f"Saved trial is unfinished or has incorrect method/seed identity: {path}")
    if not (path / "trial.mp4").is_file():
        raise FileNotFoundError(f"Completed trial has no requested video: {path}")
    if row["evaluation"] == "completed":
        summary_path = path / "evaluation" / "summary.json"
        evaluation = json.loads(summary_path.read_text())
        if evaluation["run_id"] != metadata["run_id"]:
            raise ValueError(f"Evaluation/raw run identity mismatch: {path}")
        # Resolve from the actual directory, allowing the batch to be inspected locally.
        row["evaluation_dir"] = str(summary_path.parent)
    elif row["evaluation"] != "unavailable":
        raise ValueError(f"Unfinished evaluation: {path}")
    full, metrics = _read_trial_metrics(row)
    events = [json.loads(line) for line in required[2].read_text().splitlines() if line.strip()]
    if any(event.get("run_id") != metadata["run_id"] for event in events):
        raise ValueError(f"Event/raw run identity mismatch: {path}")
    row.update(full_evaluation_window=full, metrics=metrics,
               survival=classify_survival(events, row, float(trial["duration_s"])))
    return row


def _save_batch_summary(trials: list[dict[str, Any]], rows: dict[str, dict[str, Any]], output_dir: Path) -> None:
    requested = Counter((t["group"], t["payload"]["added_mass_kg"]) for t in trials)
    points = []
    for (method, mass), count in requested.items():
        planned = [t for t in trials if (t["group"], t["payload"]["added_mass_kg"]) == (method, mass)]
        found = [rows[t["output_dir"]] for t in planned if t["output_dir"] in rows]
        summary = _summarize(found, count, planned[0]["evaluation"])
        known = [row["survival"]["survived"] for row in found if row["survival"]["status"] == "computed"]
        summary["survival"] = {
            "status": "computed" if found and len(known) == len(found) else "not_evaluable",
            "n": len(known), "unknown": len(found) - len(known),
            "survived": sum(known),
            "percent": 100 * sum(known) / len(known) if known and len(known) == len(found) else None,
        }
        summary["normalization"] = {
            "status": "reference_configured" if planned[0]["evaluation"].get("reference_run_dir") else "not_configured",
            "reason": "Uses the existing evaluator and its reference compatibility checks; no inferred denominator.",
        }
        points.append({"method": method, "payload_kg": mass, **summary})
    _write_json(output_dir / "summary.json", {"planned_trials": len(trials), "recorded_trials": len(rows), "points": points})
    temporary = output_dir / "table3.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "payload_kg", "recorded_trials", "full_windows", "normalized_E_v_mean",
                         "normalized_E_v_sd", "E_lin_m_s_mean", "E_ang_rad_s_mean", "roll_rms_deg_mean",
                         "roll_rms_deg_sd", "survival_pct", "survival_unknown"])
        for point in points:
            metrics = point["metrics"]
            def metric(name: str, key: str = "mean") -> Any:
                return metrics.get(name, {}).get(key)
            writer.writerow([point["method"], point["payload_kg"], point["recorded_trials"],
                             point["recorded_trials"] - point["incomplete_windows"],
                             metric("normalized_E_v"), metric("normalized_E_v", "sample_std"),
                             metric("E_lin_m_s"), metric("E_ang_rad_s"), metric("roll_rms_deg"),
                             metric("roll_rms_deg", "sample_std"), point["survival"]["percent"], point["survival"]["unknown"]])
    temporary.replace(output_dir / "table3.csv")


def run_ablation_batch(trials: list[dict[str, Any]], *, output_dir: Path, root_dir: Path, resume: bool = False) -> None:
    """Resume only verified finalized trials; stop on the first child failure."""
    output_dir, root_dir = output_dir.resolve(), root_dir.resolve()
    snapshot = output_dir / "batch_config.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"Batch exists; use --resume or a new output directory: {output_dir}")
        if not snapshot.is_file() or json.loads(snapshot.read_text()) != {"trials": trials}:
            raise ValueError("Resume requires the identical saved batch protocol and seed plan")
    else:
        output_dir.mkdir(parents=True)
        _write_json(snapshot, {"trials": trials})
    # Preflight every existing directory before starting any new simulation.
    rows = {}
    for trial in trials:
        row = _load_completed_trial(trial)
        if row is not None:
            rows[trial["output_dir"]] = row
        elif (output_dir / "logs" / f"{trial['group']}_kg{trial['payload']['added_mass_kg']:g}_seed{trial['seed']}.log").exists():
            raise FileExistsError(f"Unfinished worker log for {trial['output_dir']}; preserve it and use a new batch directory")
    requests, logs = output_dir / "requests", output_dir / "logs"
    requests.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    _save_batch_summary(trials, rows, output_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "linux":
        env.setdefault("MUJOCO_GL", "egl")
    for index, trial in enumerate(trials):
        if trial["output_dir"] in rows:
            print(f"[Ablation] Skip finalized {trial['output_dir']}", flush=True)
            continue
        name = f"{trial['group']}_kg{trial['payload']['added_mass_kg']:g}_seed{trial['seed']}"
        request, log = requests / f"{name}.json", logs / f"{name}.log"
        _write_json(request, trial)
        command = [sys.executable, "-m", "unilab.visualization.experiment_ablation", "--trial", str(request)]
        print(f"[Ablation {index + 1}/{len(trials)}] {name}; log={log}", flush=True)
        with log.open("x") as stream:
            subprocess.run(command, cwd=root_dir, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        row = _load_completed_trial(trial)
        if row is None:
            raise RuntimeError(f"Child exited without finalized trial: {trial['output_dir']}")
        rows[trial["output_dir"]] = row
        _save_batch_summary(trials, rows, output_dir)
    print(f"[Ablation] Complete: {len(rows)}/{len(trials)}; summary={output_dir / 'summary.json'}", flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated OAT ablation trial worker")
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    run_ablation_trial(json.loads(args.trial.read_text()), root_dir=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    _main()
