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
    approach_steps: int | None

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
        approach_steps = data['approach_steps'].item() if 'approach_steps' in data else None
        dt = float(data['ctrl_dt'].item()) if 'ctrl_dt' in data else None
    if (t.ndim != 1 or xyz.shape != (len(t), 3) or quats.shape != (len(t), 4)
            or not all(np.isfinite(x).all() for x in (t, xyz, quats))):
        raise ValueError(f'{path}: expected finite, timestamped base positions and wxyz poses')
    if dt is None:
        if len(t) < 2:
            raise ValueError(f'{path}: expected at least two timestamps or an explicit control period')
        dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(t, np.arange(len(t)) * dt, rtol=0, atol=1e-8):
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
    if len(t) == 0:
        if stop not in {'terminated', 'truncated'}:
            raise ValueError(f'{path}: empty trajectory requires an early stop')
        return _Replay(xyz[:, :2], commands, route, hold, dt, stop, approach_steps)
    norm = np.linalg.norm(quats[0])
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0):
        raise ValueError(f'{path}: initial quaternion must be unit length')
    yaw = np_yaw_from_quat(quats[:1] / norm)[0]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = (xyz[:, :2] - xyz[0, :2]) @ np.array([[c, -s], [s, c]])
    return _Replay(xy, commands, route, hold, dt, stop, approach_steps)


def evaluate_figure_eight(trajectory: Path, reference: Path) -> dict[str, Any]:
    """Score XY RMSE at matching command steps, aligned only at the entry pose."""
    nominal = _load_replay(reference)
    actual = _load_replay(trajectory)
    if (nominal.segment_steps != actual.segment_steps
            or not np.isclose(nominal.dt, actual.dt, rtol=0, atol=1e-9)
            or not np.array_equal(nominal.commands, actual.commands)
            or not np.array_equal(nominal.reference_points, actual.reference_points)):
        raise ValueError('trajectory and reference must replay the same command table and geometry')
    if actual.approach_steps != nominal.approach_steps:
        raise ValueError('trajectory and reference must use the same approach duration')
    end = 2 * actual.lobe_steps

    def complete(replay: _Replay) -> bool:
        return len(replay.xy) > end and not (
            len(replay.xy) == end + 1 and replay.stop_reason in {'terminated', 'truncated'}
        )

    if not complete(nominal):
        raise ValueError('reference first cycle is incomplete')
    completed = complete(actual)
    result: dict[str, Any] = {
        'trajectory': str(trajectory), 'reference': str(reference),
        'cycle_duration_s': 2 * actual.lobe_steps * actual.dt,
        'observed_duration_s': max(len(actual.xy) - 1, 0) * actual.dt,
        'stop_reason': actual.stop_reason,
        'status': 'scored' if completed else 'incomplete',
        'reason': None if completed else 'first_cycle_incomplete',
        'e_pos_m': None,
    }
    if completed:
        delta = actual.xy[:end + 1] - nominal.xy[:end + 1]
        result['e_pos_m'] = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return result
