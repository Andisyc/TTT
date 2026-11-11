"""Offline response identification and bounded Gerono feedforward calibration."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def calibration_identity(checkpoint_path, env_cfg, figure_eight, *, task_name, observation_contract):
    digest = hashlib.sha256()
    with Path(checkpoint_path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    control = _get(env_cfg, "control_config")
    return _json_value({
        "checkpoint_sha256": digest.hexdigest(), "backend": "mujoco",
        "task_name": str(task_name), "observation_contract": str(observation_contract),
        "ctrl_dt": float(_get(env_cfg, "ctrl_dt")), "sim_dt": _get(env_cfg, "sim_dt"),
        "action_scale": _get(control, "action_scale"),
        "simulate_action_latency": _get(control, "simulate_action_latency"),
        "geometry": {key: float(_get(figure_eight, key)) for key in (
            "width_m", "height_m", "speed_m_s", "segment_duration_s",
        )},
    })


@dataclass
class ResponseTrace:
    commands: np.ndarray
    poses: np.ndarray
    ctrl_dt: float

    def __post_init__(self):
        self.commands = np.asarray(self.commands, dtype=np.float64)
        self.poses = np.asarray(self.poses, dtype=np.float64)
        if (self.commands.ndim != 2 or self.commands.shape[1] != 3
                or len(self.commands) < 1 or self.poses.shape != (len(self.commands) + 1, 3)
                or not np.isfinite(self.commands).all() or not np.isfinite(self.poses).all()
                or not np.isfinite(self.ctrl_dt) or self.ctrl_dt <= 0):
            raise ValueError("Response trace requires finite commands[N,3], poses[N+1,3], positive ctrl_dt")
        self.ctrl_dt = float(self.ctrl_dt)


def body_twists(trace: ResponseTrace) -> np.ndarray:
    """Recover held planar twists; yaw rate is heading change, not body gyro z."""
    yaw = np.unwrap(trace.poses[:, 2])
    turns = np.diff(yaw)
    factor = trace.ctrl_dt * np.sinc(turns / (2 * np.pi))
    if np.any(np.abs(factor) < trace.ctrl_dt * 1e-6):
        raise ValueError("Response heading changes are under-resolved")
    middle = (yaw[:-1] + yaw[1:]) / 2
    delta = np.diff(trace.poses[:, :2], axis=0)
    c, s = np.cos(middle), np.sin(middle)
    return np.column_stack(((c * delta[:, 0] + s * delta[:, 1]) / factor,
                            (-s * delta[:, 0] + c * delta[:, 1]) / factor,
                            turns / trace.ctrl_dt))


def integrate_twists(twists: np.ndarray, ctrl_dt: float) -> np.ndarray:
    turn = twists[:, 2] * ctrl_dt
    yaw = np.concatenate(([0.0], np.cumsum(turn)))
    middle = (yaw[:-1] + yaw[1:]) / 2
    c, s = np.cos(middle), np.sin(middle)
    factor = ctrl_dt * np.sinc(turn / (2 * np.pi))
    delta = np.column_stack((c * twists[:, 0] - s * twists[:, 1],
                             s * twists[:, 0] + c * twists[:, 1])) * factor[:, None]
    return np.column_stack((np.vstack((np.zeros(2), np.cumsum(delta, axis=0))), yaw))


def _save(path, **arrays):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, format_version=np.array(1), **arrays)


def save_measurements(path, traces: Sequence[ResponseTrace], metadata):
    if not traces or any(trace.ctrl_dt != traces[0].ctrl_dt for trace in traces):
        raise ValueError("Measurements require nonempty trials with one ctrl_dt")
    _save(path, commands=np.concatenate([trace.commands for trace in traces]),
          poses=np.concatenate([trace.poses for trace in traces]),
          trial_steps=np.array([len(trace.commands) for trace in traces], dtype=np.int64),
          ctrl_dt=np.array(traces[0].ctrl_dt),
          metadata_json=np.array(json.dumps(_json_value(metadata), allow_nan=False)))


def load_measurements(path):
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        if data["format_version"].shape != () or int(data["format_version"]) != 1:
            raise ValueError("Unsupported response measurement format")
        commands, poses = data["commands"], data["poses"]
        counts, dt = data["trial_steps"], float(data["ctrl_dt"])
        metadata = json.loads(str(data["metadata_json"]))
    if (counts.ndim != 1 or counts.dtype.kind not in "iu" or len(counts) == 0
            or np.any(counts <= 0) or sum(map(int, counts)) != len(commands)
            or sum(map(int, counts)) + len(counts) != len(poses)
            or not isinstance(metadata, dict)):
        raise ValueError("Invalid response measurement trial boundaries")
    trials, ci, pi = [], 0, 0
    for count in map(int, counts):
        trials.append(ResponseTrace(commands[ci:ci+count], poses[pi:pi+count+1], dt))
        ci, pi = ci + count, pi + count + 1
    return trials, metadata


def _features(commands):
    if np.any(np.abs(commands[:, 1]) > 1e-12):
        raise ValueError("Gerono calibration supports forward/yaw commands, not lateral commands")
    v, w = commands[:, 0], commands[:, 2]
    pos, neg = np.maximum(w, 0), np.minimum(w, 0)
    return np.column_stack((np.ones(len(commands)), v, pos, neg, v * pos, v * neg))


@dataclass
class ResponseModel:
    decay: np.ndarray
    coefficients: np.ndarray
    command_bounds: np.ndarray
    ctrl_dt: float

    @classmethod
    def fit(cls, traces: Sequence[ResponseTrace]):
        from scipy.optimize import lsq_linear
        if not traces or any(trace.ctrl_dt != traces[0].ctrl_dt for trace in traces):
            raise ValueError("Response fitting requires trials with identical ctrl_dt")
        commands = np.concatenate([trace.commands for trace in traces])
        features = _features(commands)
        if np.linalg.matrix_rank(features) < features.shape[1]:
            raise ValueError("Unidentifiable response: measure multiple forward speeds and both turn directions")
        actual = [body_twists(trace) for trace in traces]
        values = np.concatenate(actual)
        previous = np.concatenate([np.vstack((np.zeros((1, 3)), value[:-1])) for value in actual])
        decay, coefficients = np.zeros(3), np.zeros((6, 3))
        for axis in range(3):
            if np.max(np.abs(values[:, axis])) < 1e-10:
                continue
            design = np.column_stack((previous[:, axis], features))
            if np.linalg.matrix_rank(design) < design.shape[1]:
                raise ValueError("Unidentifiable transient response: include starting and changing commands")
            result = lsq_linear(design, values[:, axis],
                                bounds=(np.r_[0.0, np.full(6, -np.inf)],
                                        np.r_[0.9999, np.full(6, np.inf)]), tol=1e-12)
            if not result.success or not np.isfinite(result.x).all():
                raise ValueError("Response dynamics fitting failed")
            decay[axis], coefficients[:, axis] = result.x[0], result.x[1:]
        return cls(decay, coefficients, np.vstack((commands.min(axis=0), commands.max(axis=0))), traces[0].ctrl_dt)

    def steady(self, commands):
        return (_features(np.atleast_2d(commands)) @ self.coefficients) / (1 - self.decay)

    def predict(self, commands, initial_twist=None):
        from scipy.signal import lfilter
        commands = np.asarray(commands, dtype=np.float64)
        if commands.ndim != 2 or commands.shape[1] != 3 or not np.isfinite(commands).all():
            raise ValueError("Response prediction requires finite commands[N,3]")
        initial = np.zeros(3) if initial_twist is None else np.asarray(initial_twist, dtype=float)
        forcing = _features(commands) @ self.coefficients
        output = np.empty_like(forcing)
        for axis in range(3):
            output[:, axis], _ = lfilter([1.0], [1.0, -self.decay[axis]], forcing[:, axis],
                                        zi=[self.decay[axis] * initial[axis]])
        return output

    def validation_rmse(self, traces):
        if not traces or any(trace.ctrl_dt != self.ctrl_dt for trace in traces):
            raise ValueError("Heldout trials must use the fitted control timestep")
        errors = np.concatenate([self.predict(trace.commands) - body_twists(trace) for trace in traces])
        return np.sqrt(np.mean(errors * errors, axis=0))


def build_calibrated_program(model, reference_program, velocity_limits, *, min_segment_steps,
                             time_scales, max_nfev, position_tolerance_m, heading_tolerance_rad):
    from scipy.optimize import least_squares
    if (reference_program.ctrl_dt != model.ctrl_dt or min_segment_steps < 1
            or max_nfev < 1 or not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0
            or not np.isfinite(heading_tolerance_rad) or heading_tolerance_rad <= 0
            or not time_scales or any(not np.isfinite(s) or s < 1 for s in time_scales)):
        raise ValueError("Invalid Gerono calibration timing/tolerance limits")
    limits = np.asarray(velocity_limits, dtype=float)
    if limits.shape != (2, 3) or not np.isfinite(limits).all():
        raise ValueError("Calibration velocity_limits must have shape (2,3)")
    low = np.maximum(model.command_bounds[0, [0, 2]], limits[0, [0, 2]])
    high = np.minimum(model.command_bounds[1, [0, 2]], limits[1, [0, 2]])
    low[0] = max(low[0], 1e-5)
    if np.any(low >= high) or low[1] >= 0 or high[1] <= 0:
        raise ValueError("Measured command domain cannot support forward motion and both turn directions")
    count = len(reference_program.commands)
    reference = reference_program.reference_points
    heading = np.cumsum(reference_program.commands[:, 2] * reference_program.segment_duration_s)
    desired = np.tile(np.column_stack((reference[1:], heading)), (3, 1))
    bounds = (np.tile(low, count), np.tile(high, count))
    best = None
    for scale in time_scales:
        holds = np.full(count, max(min_segment_steps, int(np.ceil(reference_program.segment_steps * scale - 1e-10))), dtype=np.int64)
        target_twists = reference_program.commands * (reference_program.segment_steps / holds[:, None])
        initial = np.empty((count, 2))
        for index, target in enumerate(target_twists):
            def inverse(value):
                return model.steady([[value[0], 0.0, value[1]]])[0] - target
            initial[index] = least_squares(inverse, np.clip(target[[0, 2]], low, high),
                                           bounds=(low, high), max_nfev=max_nfev).x
        boundaries = np.cumsum(np.tile(holds, 3))

        def trajectory(values):
            commands = np.column_stack((values.reshape(count, 2)[:, 0], np.zeros(count),
                                        values.reshape(count, 2)[:, 1]))
            expanded = np.repeat(np.tile(commands, (3, 1)), np.tile(holds, 3), axis=0)
            return integrate_twists(model.predict(expanded), model.ctrl_dt)[boundaries]

        def residual(values):
            difference = trajectory(values) - desired
            difference[:, :2] /= position_tolerance_m
            difference[:, 2] /= heading_tolerance_rad
            return difference.ravel()

        result = least_squares(residual, initial.ravel(), bounds=bounds, max_nfev=max_nfev)
        predicted = trajectory(result.x)
        difference = predicted - desired
        position_error = np.linalg.norm(difference[:, :2], axis=1)
        heading_error = np.abs(difference[:, 2])
        score = max(position_error.max() / position_tolerance_m, heading_error.max() / heading_tolerance_rad)
        report = {"predicted_only": True, "time_scale": float(scale),
                  "max_boundary_position_error_m": float(position_error.max()),
                  "max_boundary_heading_error_rad": float(heading_error.max()),
                  "optimizer_nfev": int(result.nfev), "prediction_cycles": 3,
                  "crossing_errors": [{"boundary": i + 1, "position_m": float(position_error[i]),
                                       "heading_rad": float(heading_error[i])}
                                      for i in range(count // 2 - 1, 3 * count, count // 2)]}
        if best is None or score < best[0]:
            best = (score, report)
        if score <= 1:
            commands = np.column_stack((result.x.reshape(count, 2)[:, 0], np.zeros(count),
                                        result.x.reshape(count, 2)[:, 1]))
            return {"commands": commands, "hold_steps": holds, "reference_points": reference.copy(),
                    "ctrl_dt": model.ctrl_dt, "report": report}
    assert best is not None
    raise ValueError(f"No feasible Gerono table within measured response domain and bounded time scales: {best[1]}")


def save_program(path, commands, hold_steps, reference_points, ctrl_dt, metadata):
    commands, holds, reference = np.asarray(commands), np.asarray(hold_steps), np.asarray(reference_points)
    if (commands.ndim != 2 or commands.shape[1] != 3 or len(commands) < 2
            or not np.isfinite(commands).all() or np.any(commands[:, 0] <= 0)
            or holds.shape != (len(commands),) or holds.dtype.kind not in "iu" or np.any(holds <= 0)
            or reference.shape != (len(commands) + 1, 2) or not np.isfinite(reference).all()
            or not np.allclose(reference[0], reference[-1], atol=1e-9, rtol=0)
            or not np.isfinite(ctrl_dt) or ctrl_dt <= 0 or "identity" not in metadata):
        raise ValueError("Invalid calibrated Gerono program")
    _save(path, commands=commands, hold_steps=holds, reference_points=reference,
          ctrl_dt=np.array(float(ctrl_dt)),
          metadata_json=np.array(json.dumps(_json_value(metadata), allow_nan=False)))
