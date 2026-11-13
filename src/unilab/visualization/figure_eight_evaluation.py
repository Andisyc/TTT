"""Score the first recorded figure-eight against a measured nominal replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.envs.common.rotation import np_yaw_from_quat


@dataclass(frozen=True)
class _Replay:
    xy: np.ndarray
    commands: np.ndarray
    reference_points: np.ndarray
    segment_steps: int
    dt: float
    stop_reason: str

    @property
    def lobe_steps(self) -> int:
        return len(self.commands) // 2 * self.segment_steps


def _load_replay(path: Path) -> _Replay:
    with np.load(path, allow_pickle=False) as data:
        t = data['time_s']
        xyz = data['base_positions']
        quats = data['base_quats']
        commands = data['commands']
        route = data['reference_points']
        hold = data['segment_steps'].item()
        stop = data['stop_reason'].item()
    if (t.ndim != 1 or len(t) < 2 or xyz.shape != (len(t), 3)
            or quats.shape != (len(t), 4)
            or not all(np.isfinite(x).all() for x in (t, xyz, quats))):
        raise ValueError(f'{path}: expected finite, timestamped base positions and wxyz poses')
    dt = float(t[1] - t[0])
    if dt <= 0 or not np.allclose(t, np.arange(len(t)) * dt, rtol=0, atol=1e-8):
        raise ValueError(f'{path}: expected contiguous control samples starting at zero')
    if (commands.ndim != 2 or commands.shape[1] != 3 or len(commands) < 4
            or len(commands) % 2 or not np.isfinite(commands).all()
            or route.shape != (len(commands) + 1, 2) or not np.isfinite(route).all()
            or type(hold) is not int or hold <= 0):
        raise ValueError(f'{path}: invalid figure-eight command or reference geometry')
    if not np.allclose(route[[len(commands) // 2, -1]], route[0], rtol=0, atol=1e-8):
        raise ValueError(f'{path}: reference lobes must start and end at the same crossing')
    if stop not in {'command_cycle_completed', 'duration_reached', 'terminated', 'truncated'}:
        raise ValueError(f'{path}: unknown stop reason {stop!r}')
    norm = np.linalg.norm(quats[0])
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0):
        raise ValueError(f'{path}: initial quaternion must be unit length')
    yaw = np_yaw_from_quat(quats[:1] / norm)[0]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = (xyz[:, :2] - xyz[0, :2]) @ np.array([[c, -s], [s, c]])
    return _Replay(xy, commands, route, hold, dt, stop)


def _signed_area(xy: np.ndarray) -> float:
    x, y = (xy - xy[0]).T
    return float(0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def _resample_lobe(xy: np.ndarray, progress: np.ndarray) -> np.ndarray:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 0]
    return np.column_stack([
        np.interp(progress, distance[keep] / distance[-1], coord[keep]) for coord in xy.T
    ])


def _first_cycle(replay: _Replay, closure_tol_m: float) -> tuple[str | None, list[np.ndarray], list[float]]:
    half = replay.lobe_steps
    end = 2 * half
    if len(replay.xy) <= end or (len(replay.xy) == end + 1 and replay.stop_reason in {'terminated', 'truncated'}):
        return 'first_cycle_incomplete', [], []
    lobes = [replay.xy[:half + 1], replay.xy[half:end + 1]]
    closure_errors = np.linalg.norm(replay.xy[[half, end]] - replay.xy[0], axis=1).tolist()
    if max(closure_errors) > closure_tol_m:
        return 'missed_crossing', [], closure_errors
    mid = len(replay.commands) // 2
    intended = (replay.reference_points[:mid + 1], replay.reference_points[mid:])
    for actual, nominal in zip(lobes, intended, strict=True):
        area, nominal_area = _signed_area(actual), _signed_area(nominal)
        if abs(area) <= 1e-8 or area * nominal_area <= 0:
            return 'degenerate_or_wrong_lobe_direction', [], closure_errors
    return None, lobes, closure_errors


def evaluate_figure_eight(trajectory: Path, reference: Path, *, closure_tol_m: float) -> dict[str, Any]:
    """Compare matching command lobes, without refitting a frame or scoring partial cycles."""
    if not np.isfinite(closure_tol_m) or closure_tol_m <= 0:
        raise ValueError('closure tolerance must be finite and positive')
    nominal = _load_replay(reference)
    actual = _load_replay(trajectory)
    if (nominal.segment_steps != actual.segment_steps
            or not np.isclose(nominal.dt, actual.dt, rtol=0, atol=1e-9)
            or not np.array_equal(nominal.commands, actual.commands)
            or not np.array_equal(nominal.reference_points, actual.reference_points)):
        raise ValueError('trajectory and reference must replay the same command table and geometry')
    ref_failure, ref_lobes, _ = _first_cycle(nominal, closure_tol_m)
    if ref_failure is not None:
        raise ValueError(f'reference first cycle is not scoreable: {ref_failure}')
    failure, lobes, closure_errors = _first_cycle(actual, closure_tol_m)
    result: dict[str, Any] = {
        'trajectory': str(trajectory), 'reference': str(reference),
        'cycle_duration_s': 2 * actual.lobe_steps * actual.dt,
        'observed_duration_s': (len(actual.xy) - 1) * actual.dt,
        'stop_reason': actual.stop_reason, 'closure_tol_m': closure_tol_m,
        'status': 'incomplete' if failure else 'scored', 'reason': failure,
        'crossing_return_errors_m': closure_errors,
        'lobe_area_error_pct': None, 'e_pos_m': None, 'lobes': [],
    }
    if failure is not None:
        return result

    progress = np.linspace(0.0, 1.0, 201)
    for actual_lobe, ref_lobe in zip(lobes, ref_lobes, strict=True):
        area, ref_area = abs(_signed_area(actual_lobe)), abs(_signed_area(ref_lobe))
        delta = _resample_lobe(actual_lobe, progress) - _resample_lobe(ref_lobe, progress)
        result['lobes'].append({
            'area_m2': area, 'reference_area_m2': ref_area,
            'area_error_pct': 100 * abs(area - ref_area) / ref_area,
            'e_pos_m': float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        })
    scores = result['lobes']
    result['lobe_area_error_pct'] = float(np.mean([row['area_error_pct'] for row in scores]))
    result['e_pos_m'] = float(np.sqrt(np.mean([row['e_pos_m'] ** 2 for row in scores])))
    return result
