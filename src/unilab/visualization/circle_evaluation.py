"""Compare measured circle outlines without removing radius or shape differences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from unilab.envs.common.rotation import np_yaw_from_quat


def _load_circle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        stop = str(data["stop_reason"].item())
        if stop != "lap_completed":
            return {"stop_reason": stop}
        time = data["time_s"]
        positions, quats = data["base_positions"], data["base_quats"]
        entry = int(data["circle_start_step"].item())
        payload = data["payload_mass_kg"]
        protocol = {key: data[key] for key in (
            "ctrl_dt", "command", "approach_steps", "startup_steps", "approach_command",
        )}
    dt = float(protocol["ctrl_dt"])
    if (time.ndim != 1 or positions.shape != (len(time), 3) or quats.shape != (len(time), 4)
            or payload.shape != time.shape or not 0 <= entry < len(time) - 1
            or not all(np.isfinite(value).all() for value in (time, positions, quats, payload))
            or not np.isfinite(dt) or dt <= 0
            or not np.allclose(time, np.arange(len(time)) * dt, rtol=0, atol=1e-8)):
        raise ValueError(f"{path}: invalid circle measurement samples")
    if not np.isclose(np.linalg.norm(quats[entry]), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: circle entry quaternion must be unit length")
    yaw = float(np_yaw_from_quat(quats[entry:entry + 1])[0])
    c, s = np.cos(yaw), np.sin(yaw)
    xy = (positions[entry:, :2] - positions[entry, :2]) @ np.array([[c, -s], [s, c]])
    metadata = json.loads((path.parent / "raw" / "run_config.json").read_text())
    return {"stop_reason": stop, "xy": xy, "payload": payload,
            "protocol": protocol, "metadata": metadata}


def _sample_outline(xy: np.ndarray) -> np.ndarray | None:
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    if distance[-1] == 0:
        return None
    # Exact repeated positions have no additional geometric information.
    keep = np.concatenate(([True], np.diff(distance) > 0))
    progress = distance[keep] / distance[-1]
    grid = np.linspace(0.0, 1.0, 1001)
    return np.column_stack([np.interp(grid, progress, xy[keep, axis]) for axis in (0, 1)])


def evaluate_circle_shape(trajectory: Path, reference: Path) -> dict[str, Any]:
    """Score entry-aligned XY RMSE at equal normalized arc-length progress, in metres."""
    trajectory, reference = trajectory.resolve(), reference.resolve()
    if trajectory == reference:
        raise ValueError("circle reference must be a separate unloaded trial")
    result: dict[str, Any] = {
        "status": "DNF", "E_shape_m": None, "reference": str(reference),
        "sample_count": 1001, "alignment": "entry_position_and_body_yaw",
        "sampling": "normalized_arc_length", "unit": "m",
    }
    actual = _load_circle(trajectory)
    if actual["stop_reason"] != "lap_completed":
        return result
    nominal = _load_circle(reference)
    if nominal["stop_reason"] != "lap_completed":
        raise ValueError("circle reference must complete one lap")
    if np.any(nominal["payload"] != 0):
        raise ValueError("circle reference must be unloaded")
    config = nominal["metadata"].get("resolved_config", {})
    if config.get("online_adaptation", {}).get("enabled") is not False:
        raise ValueError("circle reference must use a frozen no-online-adaptation policy")
    for key in ("command", "ctrl_dt"):
        if not np.array_equal(actual["protocol"][key], nominal["protocol"][key]):
            raise ValueError(f"circle reference has incompatible {key}")
    result["adaptation_protocol_differences"] = {
        key: {"trial": actual["protocol"][key].tolist(), "reference": nominal["protocol"][key].tolist()}
        for key in ("approach_steps", "startup_steps", "approach_command")
        if not np.array_equal(actual["protocol"][key], nominal["protocol"][key])
    }
    checkpoint = actual["metadata"].get("checkpoint_sha256")
    if not checkpoint or nominal["metadata"].get("checkpoint_sha256") != checkpoint:
        raise ValueError("circle reference requires the same checkpoint")
    seed = actual["metadata"].get("resolved_config", {}).get("algo", {}).get("seed")
    if seed is None or config.get("algo", {}).get("seed") != seed:
        raise ValueError("circle reference requires the same explicit seed")
    baseline = _sample_outline(nominal["xy"])
    if baseline is None:
        raise ValueError("circle reference has no translational motion")
    measured = _sample_outline(actual["xy"])
    if measured is None:
        result.update(status="not_evaluable", reason="no_translational_motion")
        return result
    result.update(status="scored", E_shape_m=float(np.sqrt(np.mean(np.sum((measured - baseline)**2, axis=1)))))
    return result
