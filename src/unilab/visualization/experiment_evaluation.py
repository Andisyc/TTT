"""Offline OAT evaluation; never creates an environment or loads a policy.

Velocity RMSE arithmetic is adapted from LeCAR-Lab/FADA (Apache-2.0),
revision 290b3a9c5c0ebe2e9851b61350617544be05e9bc, source linked in
FADA_METRIC_SOURCE. Unlike that mocap utility, this adapter does not trim,
filter nonfinite samples, or remove terminal transitions. Protocol choices
are explicit inputs, not inferred from recording/training defaults.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.envs.common.rotation import np_matrix_from_quat, np_wrap_to_pi
from unilab.visualization.experiment_measurement import (
    FADA_METRIC_SOURCE,
    ControlStepMeasurement,
    PhysicalMeasurement,
    velocity_metric_inputs,
)


class EvaluationWindowUnavailable(ValueError):
    """The requested window did not occur in this recorded trial."""


@dataclass
class RecordedRun:
    path: Path
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    steps: list[ControlStepMeasurement]
    submitted: list[np.ndarray]


def read_recorded_run(path: str | Path) -> RecordedRun:
    """Read committed chunks only; reject broken identities/order, never repair."""
    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / "run_config.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported experiment recording format")
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line.strip()]
    if any(event.get("run_id") != metadata["run_id"] for event in events):
        raise ValueError("event run identity mismatch")
    steps, submitted = [], []
    for index, part in enumerate(sorted((path / "measurements").glob("part_*.npz"))):
        if part.name != f"part_{index:06d}.npz":
            raise ValueError("missing or misordered measurement chunk")
        with np.load(part, allow_pickle=False) as data:
            run_id = str(data["run_id"].item())
            dt = float(data["ctrl_dt"].item())
            names = {key: tuple(data[key].tolist()) for key in ("policy_joint_names", "environment_joint_names")}
            if run_id != metadata["run_id"] or dt != metadata["ctrl_dt"]:
                raise ValueError("chunk run identity or ctrl_dt mismatch")
            if any(list(value) != metadata[key] for key, value in names.items()):
                raise ValueError("chunk joint order differs from run metadata")
            count = len(data["control_step"])
            row_keys = ("env_index", "episode_id", "time_before_s", "time_after_s", "velocity_command_body",
                        "policy_action", "environment_action", "submitted_environment_action", "terminated", "truncated")
            row_keys += tuple(f"{key}_{phase}" for phase in ("before", "after") for key in
                              ("base_quat_wxyz", "linear_velocity_world", "angular_velocity_world"))
            if count == 0 or any(len(data[key]) != count for key in row_keys):
                raise ValueError("empty or inconsistent measurement chunk")
            for row in range(count):
                episode, control_step = data["episode_id"][row], data["control_step"][row]
                def physical(phase: str) -> PhysicalMeasurement:
                    return PhysicalMeasurement(
                        run_id, data["env_index"][row], episode,
                        control_step + (phase == "after"), float(data[f"time_{phase}_s"][row]),
                        *(data[f"{key}_{phase}"][row] for key in
                          ("base_quat_wxyz", "linear_velocity_world", "angular_velocity_world")),
                    )
                step = ControlStepMeasurement(
                    physical("before"), physical("after"), dt, data["velocity_command_body"][row],
                    data["policy_action"][row], data["environment_action"][row],
                    names["policy_joint_names"], names["environment_joint_names"],
                    data["terminated"][row], data["truncated"][row],
                )
                if step.before.env_index != 0:
                    raise ValueError("only single-environment recordings are supported")
                if steps:
                    previous = steps[-1]
                    if step.before.episode_id == previous.before.episode_id:
                        if previous.terminated or previous.truncated or step.before.control_step != previous.after.control_step:
                            raise ValueError("duplicate, missing, or post-terminal control step")
                        if not np.isclose(step.before.sim_time_s, previous.after.sim_time_s, atol=1e-9, rtol=0):
                            raise ValueError("noncontiguous measurement time")
                        for key in ("base_quat_wxyz", "linear_velocity_world", "angular_velocity_world"):
                            if not np.allclose(getattr(step.before, key), getattr(previous.after, key), atol=1e-9, rtol=0):
                                raise ValueError("physical discontinuity without an episode reset")
                    elif step.before.episode_id <= previous.before.episode_id or step.before.sim_time_s < previous.after.sim_time_s:
                        raise ValueError("episode or time order is invalid")
                if (not steps or step.before.episode_id != steps[-1].before.episode_id) and step.before.control_step != 0:
                    raise ValueError("episode is missing its initial control step")
                action = np.array(data["submitted_environment_action"][row], copy=True)
                if action.shape != step.environment_action.shape or not np.isfinite(action).all():
                    raise ValueError("invalid submitted environment action")
                steps.append(step)
                submitted.append(action)
    if not steps:
        raise ValueError("recording has no committed measurements")
    if metadata.get("recorded_control_steps", len(steps)) != len(steps):
        raise ValueError("committed row count differs from finalized run metadata")
    return RecordedRun(path, metadata, events, steps, submitted)


def _window(run: RecordedRun, options: dict[str, Any]) -> tuple[float, float, list[int]]:
    start, end = options.get("start_s"), options.get("end_s")
    if start is None or end is None or not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("explicit finite start_s < end_s is required")
    origin = options.get("relative_to")
    offset = 0.0
    if origin == "payload":
        events = [e for e in run.events if e["kind"] == "payload_applied"]
        event_index = options.get("payload_event_index")
        if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
            raise ValueError("payload-relative evaluation requires an explicit nonnegative payload_event_index")
        if event_index >= len(events):
            raise EvaluationWindowUnavailable("requested payload event was not recorded")
        offset = float(events[event_index]["sim_time_s"])
    elif origin != "run_start":
        raise ValueError("relative_to must explicitly select run_start or payload")
    start, end = float(start + offset), float(end + offset)
    if start < 0:
        raise ValueError("requested window begins before run start")
    indices = [i for i, step in enumerate(run.steps)
               if step.before.sim_time_s >= start - 1e-9 and step.after.sim_time_s <= end + 1e-9
               and (options.get("episode_id") is None or step.before.episode_id == options["episode_id"])]
    if not indices:
        raise EvaluationWindowUnavailable("requested window contains no complete control transitions")
    return start, end, indices


def _velocity_rmse(linear_sq: np.ndarray, angular_sq: np.ndarray) -> dict[str, float]:
    # FADA _compute_tracking_returns, lines 578-592: sum XY BEFORE temporal mean.
    # Deliberately no _rmse_pre_fall slicing or finite-value filtering.
    return {"E_lin_m_s": float(np.sqrt(np.mean(linear_sq))),
            "E_ang_rad_s": float(np.sqrt(np.mean(angular_sq)))}


def _optional_metrics(steps: list[ControlStepMeasurement], submitted: list[np.ndarray],
                      options: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None]:
    result: dict[str, Any] = {}
    attitude = [options.get(key) for key in ("attitude_reference", "attitude_unit", "attitude_aggregation")]
    errors = None
    if any(value is not None for value in attitude):
        reference, unit, aggregation = attitude
        if reference is None or unit not in ("rad", "deg") or aggregation not in ("rms", "mae"):
            raise ValueError("attitude requires reference [roll,pitch], unit rad/deg, and aggregation rms/mae")
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ValueError("attitude_reference must be a finite roll/pitch pair")
        states = [step.before if options["sample_phase"] == "query" else step.after for step in steps]
        matrix = np_matrix_from_quat(np.stack([state.base_quat_wxyz for state in states]))
        # Standard XYZ Euler extraction, using the shared quaternion/matrix convention.
        roll = np.arctan2(matrix[:, 2, 1], matrix[:, 2, 2])
        pitch = np.arctan2(-matrix[:, 2, 0], np.hypot(matrix[:, 0, 0], matrix[:, 1, 0]))
        reference_rad = np.deg2rad(reference) if unit == "deg" else reference
        errors = np_wrap_to_pi(np.column_stack((roll, pitch)) - reference_rad)
        if unit == "deg":
            errors = np.rad2deg(errors)
        values = np.sqrt(np.mean(errors ** 2, axis=0)) if aggregation == "rms" else np.mean(np.abs(errors), axis=0)
        result["attitude"] = {"roll": float(values[0]), "pitch": float(values[1]), "unit": unit, "aggregation": aggregation}
    else:
        result["attitude"] = {"status": "not_evaluated", "reason": "attitude convention not supplied"}
    source, reduction = options.get("action_source"), options.get("action_joint_reduction")
    if source is not None or reduction is not None:
        if source not in ("policy_action", "environment_action", "submitted_environment_action") or reduction not in ("sum", "mean"):
            raise ValueError("action diagnostic requires an explicit action_source and action_joint_reduction sum/mean")
        actions = np.stack(submitted if source == "submitted_environment_action" else [getattr(step, source) for step in steps])
        if len(actions) < 2:
            result["action_change"] = {"status": "not_evaluable", "reason": "fewer than two adjacent actions"}
        else:
            squared = np.diff(actions, axis=0) ** 2
            per_step = squared.sum(axis=1) if reduction == "sum" else squared.mean(axis=1)
            result["action_change"] = {"mean_squared_change": float(per_step.mean()), "source": source,
                                       "joint_reduction": reduction, "unit": "normalized_action_squared"}
    else:
        result["action_change"] = {"status": "not_evaluated", "reason": "action convention not supplied"}
    return result, errors


def _evaluate(run: RecordedRun, options: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    start, end, indices = _window(run, options)
    frame, phase = options["linear_frame"], options["sample_phase"]
    window_s = options.get("curve_window_s")
    if window_s is not None and (not np.isfinite(window_s) or window_s <= 0):
        raise ValueError("curve_window_s must be positive when supplied")
    episodes, rows = [], []
    for episode in dict.fromkeys(run.steps[i].before.episode_id for i in indices):
        selected = [i for i in indices if run.steps[i].before.episode_id == episode]
        steps = [run.steps[i] for i in selected]
        inputs = velocity_metric_inputs(steps, linear_frame=frame, sample_phase=phase)
        linear_sq = np.sum(np.square(inputs.command_linear_xy - inputs.actual_linear_xy), axis=1)
        angular_sq = np.square(inputs.command_angular_z - inputs.actual_angular_body_z)
        diagnostics, attitude = _optional_metrics(steps, [run.submitted[i] for i in selected], options)
        covered_start, covered_end = steps[0].before.sim_time_s, steps[-1].after.sim_time_s
        full = np.isclose(covered_start, start, atol=1e-9, rtol=0) and np.isclose(covered_end, end, atol=1e-9, rtol=0)
        curves = []
        if window_s is not None:
            # Non-overlapping full bins; never pad missing/terminal time with zeros.
            for bin_index in range(int(np.floor((end - start) / window_s + 1e-9))):
                left, right = start + bin_index * window_s, start + (bin_index + 1) * window_s
                keep = [i for i, step in enumerate(steps) if step.before.sim_time_s >= left - 1e-9 and step.after.sim_time_s <= right + 1e-9]
                complete = bool(keep) and np.isclose(steps[keep[0]].before.sim_time_s, left, atol=1e-9, rtol=0) and np.isclose(steps[keep[-1]].after.sim_time_s, right, atol=1e-9, rtol=0)
                curve = {"start_s": left, "end_s": right, "complete": bool(complete), "samples": len(keep)}
                if complete:
                    curve.update(_velocity_rmse(linear_sq[keep], angular_sq[keep]))
                    if attitude is not None:
                        error = attitude[keep]
                        values = np.sqrt(np.mean(error ** 2, axis=0)) if options["attitude_aggregation"] == "rms" else np.mean(np.abs(error), axis=0)
                        curve.update(roll_deviation=float(values[0]), pitch_deviation=float(values[1]))
                curves.append(curve)
        episodes.append({"episode_id": int(episode), "samples": len(steps), "covered_start_s": covered_start,
                         "covered_end_s": covered_end, "covered_duration_s": len(steps) * steps[0].ctrl_dt,
                         "full_requested_window": bool(full), "terminated": bool(steps[-1].terminated),
                         "truncated": bool(steps[-1].truncated), "velocity_observed_segment": _velocity_rmse(linear_sq, angular_sq),
                         **diagnostics, "time_windows": curves})
        for i, step in enumerate(steps):
            row = {"episode_id": episode, "control_step": step.before.control_step,
                   "time_s": float(inputs.timestamps_s[i]), "linear_squared_error_m2_s2": float(linear_sq[i]),
                   "angular_squared_error_rad2_s2": float(angular_sq[i]), "terminated": step.terminated, "truncated": step.truncated}
            if attitude is not None:
                row.update(roll_error=float(attitude[i, 0]), pitch_error=float(attitude[i, 1]))
            rows.append(row)
    relevant_events = [event for event in run.events if start <= event["sim_time_s"] <= end]
    summary = {"format_version": 1, "run_dir": str(run.path), "run_id": run.metadata["run_id"],
               "checkpoint_path": run.metadata.get("checkpoint_path"), "checkpoint_sha256": run.metadata.get("checkpoint_sha256"),
               "code_identity": run.metadata.get("code_identity"), "options": options,
               "metric_source": FADA_METRIC_SOURCE, "requested_window_s": [start, end],
               "run_status": run.metadata["status"], "recorded_steps": len(run.steps),
               "incomplete_transition": run.metadata.get("incomplete_transition"),
               "selected_update_count": sum(event["kind"] == "update_completed" for event in relevant_events),
               "events": relevant_events, "episodes": episodes,
               "survival": {"status": "not_evaluable", "reason": "fall predicate and trial horizon are not defined by raw done flags"},
               "normalization": {"status": "not_evaluated", "reason": "no explicit common reference run"},
               "scope": "observed per-episode diagnostics; no failure deletion, cross-reset aggregation, E_pos or route SR"}
    return summary, rows, indices


def evaluate_experiment(options: dict[str, Any]) -> dict[str, Any]:
    """Evaluate recorded files and write summary.json plus per-step errors.csv.

    Missing physical conventions fail explicitly. Optional diagnostics remain
    labelled unavailable rather than being assigned an experiment default.
    Each invocation uses a fresh output directory; inputs are never modified.
    """
    if not options.get("run_dir") or not options.get("output_dir"):
        raise ValueError("run_dir and output_dir must be explicitly supplied")
    run = read_recorded_run(options["run_dir"])
    summary, rows, indices = _evaluate(run, options)
    reference_path = options.get("reference_run_dir")
    if reference_path and options.get("allow_incomplete_window") and any(
        not episode["full_requested_window"] for episode in summary["episodes"]
    ):
        summary["normalization"] = {"status": "not_evaluable", "reason": "incomplete requested window"}
        reference_path = None
    if reference_path:
        reference = read_recorded_run(reference_path)
        reference_summary, _, reference_indices = _evaluate(reference, options)
        for key in ("ctrl_dt", "effective_env_config", "policy_family"):
            if key not in run.metadata or key not in reference.metadata or run.metadata[key] != reference.metadata[key]:
                raise ValueError(f"normalization reference has incompatible {key}")
        def seed(meta: dict[str, Any]) -> Any:
            return meta.get("resolved_config", {}).get("algo", {}).get("seed")
        if seed(run.metadata) is None or seed(run.metadata) != seed(reference.metadata):
            raise ValueError("normalization reference requires the same explicit seed")
        def payload(record: RecordedRun) -> list[dict[str, Any]]:
            return [{key: value for key, value in event.items() if key in
                     ("sim_time_s", "added_mass_kg", "body_name", "attachment_pos_m", "half_size_m")}
                    for event in record.events if event["kind"] == "payload_applied"]
        if payload(run) != payload(reference):
            raise ValueError("normalization reference payload events differ")
        selected, reference_steps = [run.steps[i] for i in indices], [reference.steps[i] for i in reference_indices]
        if len(selected) != len(reference_steps) or any(
            a.before.episode_id != b.before.episode_id or a.before.sim_time_s != b.before.sim_time_s
            or not np.array_equal(a.velocity_command_body, b.velocity_command_body)
            for a, b in zip(selected, reference_steps)
        ):
            raise ValueError("normalization requires identical command and time support")
        if run.metadata["status"] != "completed" or reference.metadata["status"] != "completed":
            raise ValueError("normalization requires finalized completed runs")
        for actual, baseline in zip(summary["episodes"], reference_summary["episodes"], strict=True):
            if not actual["full_requested_window"] or not baseline["full_requested_window"]:
                raise ValueError("normalization requires a full common window within one episode")
            denominator = sum(baseline["velocity_observed_segment"].values())
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("normalization reference denominator must be finite and positive")
            actual["normalized_E_v"] = sum(actual["velocity_observed_segment"].values()) / denominator
        summary["normalization"] = {"status": "computed", "reference_run_id": reference.metadata["run_id"],
                                    "reference_run_dir": str(reference.path),
                                    "definition": "(E_lin+E_ang)/(E_lin_reference+E_ang_reference)",
                                    "reference_identity_is_user_selected_not_inferred_TF_DAgger": True}
    output = Path(options["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "errors.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return summary
