"""Method selection and file-only experiment summaries; no runtime dependencies."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

GROUP_MODES = {
    "fada_ttt": "fada_ttt",
    "online_reference": "trajectory_lora",
    "action_snapshot": "action_base",
    "no_anchor": "trajectory_base",
    "full_oat": "trajectory_base",
}


def _read_trial_metrics(result: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    if result["evaluation"] == "unavailable":
        return False, {}
    path = Path(result["evaluation_dir"]) / "summary.json"
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    episode, = summary["episodes"]
    metrics = dict(episode["velocity_observed_segment"])
    attitude = episode["attitude"]
    if "roll" in attitude:
        for axis in ("roll", "pitch"):
            metrics[f"{axis}_{attitude['aggregation']}_{attitude['unit']}"] = attitude[axis]
    if "mean_squared_change" in episode["action_change"]:
        metrics["mean_squared_action_change"] = episode["action_change"]["mean_squared_change"]
    if "normalized_E_v" in episode:
        metrics["normalized_E_v"] = episode["normalized_E_v"]
    return episode["full_requested_window"], metrics


def _summarize(rows: list[dict[str, Any]], requested: int, evaluation: dict[str, Any]) -> dict[str, Any]:
    names = {"E_lin_m_s", "E_ang_rad_s"}
    if evaluation.get("attitude_reference") is not None:
        names.update(f"{axis}_{evaluation['attitude_aggregation']}_{evaluation['attitude_unit']}" for axis in ("roll", "pitch"))
    if evaluation.get("action_source") is not None:
        names.add("mean_squared_action_change")
    if evaluation.get("reference_run_dir"):
        names.add("normalized_E_v")
    metrics = {}
    for name in sorted(names):
        values = [row["metrics"][name] for row in rows if row["full_evaluation_window"] and name in row["metrics"]]
        metrics[name] = {
            "n": len(values), "missing": len(rows) - len(values),
            "mean": mean(values) if values else None,
            "sample_std": stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
        }
    return {
        "requested_trials": requested, "recorded_trials": len(rows),
        "incomplete_windows": sum(not row["full_evaluation_window"] for row in rows),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)), "metrics": metrics,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
