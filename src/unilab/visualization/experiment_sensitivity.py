"""Run explicit OAT sensitivity points and aggregate independent trials."""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.visualization.experiment_common import _read_trial_metrics, _summarize, _write_json

PARAMETERS = {"T": "update_interval", "sigma": "sigma", "gamma": "gamma"}


def configure_sensitivity_timing(
    protocol: DictConfig, *, duration_s: float | None = None, attach_time_s: float | None = None,
) -> DictConfig:
    """Keep video, episode limit, warm-up and evaluation aligned when retiming."""
    result = OmegaConf.merge(protocol)
    if duration_s is None and attach_time_s is None:
        return result
    dt = float(result.get("timing_ctrl_dt_s", 0.0))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("timing overrides require explicit positive timing_ctrl_dt_s in the protocol")
    duration = float(result.duration_s if duration_s is None else duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("--duration-s must be finite and positive")
    end = math.ceil(duration / dt - 1e-9) * dt
    if attach_time_s is not None:
        if not result.payload.enabled:
            raise ValueError("--attach-time-s requires a loaded protocol")
        if not math.isfinite(attach_time_s) or attach_time_s < 0:
            raise ValueError("--attach-time-s must be finite and nonnegative")
        result.payload.attach_time_s = attach_time_s
        # Runtime payloads attach before physics at ceil(attach_time / ctrl_dt).
        result.online.warmup_steps = math.ceil(attach_time_s / dt) + 1
    if result.payload.enabled:
        attachment = math.ceil(float(result.payload.attach_time_s) / dt) * dt
        if attachment >= end:
            raise ValueError("payload attachment must occur before the recording ends")
        result.evaluation.relative_to = "payload"
        result.evaluation.payload_event_index = 0
        result.evaluation.start_s = 0.0
        result.evaluation.end_s = end - attachment
    else:
        if end <= float(result.evaluation.start_s):
            raise ValueError("recording must extend beyond the unloaded evaluation start")
        result.evaluation.relative_to = "run_start"
        result.evaluation.end_s = end
    result.duration_s = duration
    current_limit = result.env_overrides.get("max_episode_seconds")
    result.env_overrides.max_episode_seconds = max(float(current_limit or 0.0), end + dt)
    return result


def plan_sensitivity_trials(
    protocol: DictConfig, *, param: str, values: Sequence[float], repeats: int,
    first_seed: int, output_dir: Path,
) -> list[dict[str, Any]]:
    field = PARAMETERS[param]
    if repeats < 1 or first_seed < 0 or first_seed + repeats > 2**32:
        raise ValueError("repeats must be positive and all trial seeds must be in [0, 2**32)")
    if not values or len(set(values)) != len(values):
        raise ValueError("provide one or more distinct parameter values")
    trials = []
    for value in values:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{param} values must be finite and nonnegative")
        if param == "T" and (value < 1 or not float(value).is_integer()):
            raise ValueError("T values must be positive integers")
        point = int(value) if param == "T" else float(value or 0)
        for seed in range(first_seed, first_seed + repeats):
            path = output_dir / f"{param}={point}" / f"seed_{seed}"
            trial = OmegaConf.merge(protocol, {
                "group": "full_oat", "seed": seed, "output_dir": str(path),
                "online": {field: point},
                "video": {"enabled": True, "label": f"{param}={point}"},
                "evaluation": {"enabled": True, "allow_incomplete_window": True},
            })
            trials.append(OmegaConf.to_container(trial, resolve=True, throw_on_missing=True))
    return trials




def run_sensitivity_trials(
    trials: list[dict[str, Any]], *, param: str, repeats: int, output_dir: Path, root_dir: Path,
) -> None:
    """Run serially, persisting each trial and refreshing its point summary."""
    from unilab.visualization.experiment_execution import run_oat_experiment

    output_dir.mkdir(parents=True, exist_ok=False)
    point_rows: dict[int | float, list[dict[str, Any]]] = {}
    points: dict[int | float, dict[str, Any]] = {}
    for trial in trials:
        point = trial["online"][PARAMETERS[param]]
        path = Path(trial["output_dir"])
        print(f"[Sensitivity] {param}={point} seed={trial['seed']} -> {path}", flush=True)
        result = run_oat_experiment(trial, root_dir=root_dir)
        full, metrics = _read_trial_metrics(result)
        row = {**result, "full_evaluation_window": full, "metrics": metrics}
        _write_json(path / "trial.json", row)
        rows = point_rows.setdefault(point, [])
        rows.append(row)
        points[point] = {"value": point, **_summarize(rows, repeats, trial["evaluation"])}
        _write_json(output_dir / "summary.json", {
            "parameter": param, "payload": trial["payload"], "points": list(points.values()),
        })
