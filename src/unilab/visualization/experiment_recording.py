"""Opt-in raw experiment recording; no reward, learning, or metric decisions.

Committed NPZ chunks are atomic and pickle-free. Events are flushed as JSONL.
A hard kill can lose the uncommitted chunk; running metadata is NOT success.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from unilab.visualization.experiment_measurement import ControlStepMeasurement, PhysicalMeasurement


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, np.dtype)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"unsupported experiment metadata type: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, ensure_ascii=False)


@dataclass(frozen=True)
class ExperimentRecordingConfig:
    enabled: bool = False
    output_dir: str | None = None
    chunk_steps: int = 256  # Storage buffering only, not a learning/experiment parameter.

    def __post_init__(self) -> None:
        if isinstance(self.chunk_steps, bool) or not isinstance(self.chunk_steps, Integral) or self.chunk_steps <= 0:
            raise ValueError("experiment_recording.chunk_steps must be a positive integer")
        if self.enabled and (not isinstance(self.output_dir, str) or not self.output_dir.strip()):
            raise ValueError("experiment_recording.output_dir must be explicitly provided")


def experiment_run_metadata(cfg: Any, env: Any, checkpoint: str | Path, root_dir: str | Path, official_adapter: Any) -> dict[str, Any]:
    """Cold-path identity only; preserve actual settings without approving them."""
    from omegaconf import OmegaConf

    checkpoint = Path(checkpoint).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    identity = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("worktree_status", ["status", "--porcelain"])):
        result = subprocess.run(["git", "-C", str(root_dir), *args], capture_output=True, text=True, timeout=15)
        identity[key] = result.stdout.strip() if result.returncode == 0 else None
    target_names = tuple(env.get_actuated_joint_names())
    profile = official_adapter.profile if official_adapter is not None else None
    return {
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "effective_env_config": env.cfg,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
        "code_identity": identity,
        "ctrl_dt": float(env.cfg.ctrl_dt),
        "policy_family": "official" if profile is not None else "legacy",
        "official_control_profile": profile,
        "policy_joint_names": profile.joint_names if profile is not None else target_names,
        "environment_joint_names": target_names,
        "policy_action_semantics": "saved normalized policy command, source order, existing playback clipping only",
        "environment_action_semantics": "env.state.info.current_actions, target order; not physical torque; delay/fault semantics in effective_env_config",
        "submitted_environment_action_semantics": "command passed to wrapped_env.step before environment authority/fault/delay handling",
        "formal_protocol_approved": False,
    }


class ExperimentRecorder:
    """Single-env recorder attached to the normal Session, never a collector."""

    def __init__(self, env: Any, config: ExperimentRecordingConfig, metadata: dict[str, Any]) -> None:
        if not config.enabled or env.num_envs != 1:
            raise ValueError("experiment recorder requires enabled single-env playback")
        if config.output_dir is None:
            raise ValueError("experiment recording requires an explicit output directory")
        for name in ("set_step_measurement_callback", "get_base_quat", "get_base_lin_vel", "get_base_ang_vel"):
            if not callable(getattr(env, name, None)):
                raise ValueError(f"experiment recording requires public env.{name}()")
        self.env = env
        self.config = config
        self.ctrl_dt = float(env.cfg.ctrl_dt)
        if not np.isfinite(self.ctrl_dt) or self.ctrl_dt <= 0:
            raise ValueError("experiment recording requires finite positive ctrl_dt")
        self.policy_joint_names = tuple(metadata["policy_joint_names"])
        self.environment_joint_names = tuple(metadata["environment_joint_names"])
        self.run_id = uuid.uuid4().hex
        self._started = time.monotonic()
        self._run_steps = 0
        self._episode_id = -1
        self._episode_step = 0
        self._update_count = 0
        self._chunk_index = 0
        self._committed_steps = 0
        self._records: list[tuple[ControlStepMeasurement, np.ndarray]] = []
        self._pending: dict[str, Any] | None = None
        self._ended = False
        self.last_done = False
        self._closed = False
        self._metadata = json.loads(_json({
            **metadata, "format_version": 1, "run_id": self.run_id, "status": "running",
            "max_uncommitted_control_steps": int(config.chunk_steps),
            "measurement_boundary": "action_query and completed_before_autoreset",
        }))
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._parts = self.output_dir / "measurements"
        self._parts.mkdir()
        self._write_metadata()
        self._events = (self.output_dir / "events.jsonl").open("x", encoding="utf-8")

    def _write_metadata(self) -> None:
        temporary = self.output_dir / "run_config.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_json(self._metadata) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.output_dir / "run_config.json")

    def event(self, kind: str, *, sim_time_s: float | None = None, **details: Any) -> None:
        if self._closed:
            raise RuntimeError("experiment recorder is closed")
        self._events.write(_json({
            "kind": kind, "run_id": self.run_id, "episode_id": self._episode_id,
            "control_step": self._episode_step, "run_control_step": self._run_steps,
            "sim_time_s": self._run_steps * self.ctrl_dt if sim_time_s is None else sim_time_s,
            "wall_elapsed_s": time.monotonic() - self._started, **details,
        }) + "\n")
        self._events.flush()

    def on_reset(self, reason: str) -> None:
        self.flush()
        previous_episode = self._episode_id
        self._episode_id += 1
        self._episode_step = 0
        self._pending = None
        self._ended = False
        self.last_done = False
        self.event("reset", reason=reason, previous_episode_id=previous_episode)

    def _physical(self, episode_step: int, run_step: int) -> PhysicalMeasurement:
        return PhysicalMeasurement(
            run_id=self.run_id, env_index=0, episode_id=self._episode_id,
            control_step=episode_step, sim_time_s=run_step * self.ctrl_dt,
            base_quat_wxyz=np.asarray(self.env.get_base_quat())[0],
            linear_velocity_world=np.asarray(self.env.get_base_lin_vel())[0],
            angular_velocity_world=np.asarray(self.env.get_base_ang_vel())[0],
        )

    def begin_query(self, commands: np.ndarray) -> None:
        if self._closed or self._episode_id < 0:
            raise RuntimeError("experiment recording requires an open, reset Session")
        if self._pending is not None:
            raise RuntimeError("previous experiment transition is incomplete")
        if self._ended:
            return  # Physics can still run without autoreset; do not invent a new trial.
        command = np.asarray(commands, dtype=np.float64)
        if command.shape != (1, 3) or not np.isfinite(command).all():
            raise ValueError("measurement command must be finite with shape (1, 3)")
        self._pending = {"before": self._physical(self._episode_step, self._run_steps), "command": command[0].copy()}

    def submit_actions(self, policy_action: Any, environment_action: Any) -> None:
        if self._pending is None:
            return
        def row(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)[0].copy()
        self._pending["policy_action"] = row(policy_action)
        self._pending["submitted_action"] = row(environment_action)

    def on_step_completed(self, state: Any, invalid_physics_rows: np.ndarray | None) -> None:
        """Called by NpEnv after state completion, before any automatic reset."""
        if self._pending is None:
            self._run_steps += 1
            return
        if invalid_physics_rows is not None and 0 in invalid_physics_rows:
            self.event("invalid_measurement", reason="physics_guard_replaced_original_state")
            raise RuntimeError("original physical truth unavailable after physics guard; refusing reset-state measurement")
        pending = self._pending
        terminated, truncated = bool(state.terminated[0]), bool(state.truncated[0])
        record = ControlStepMeasurement(
            before=pending["before"], after=self._physical(self._episode_step + 1, self._run_steps + 1),
            ctrl_dt=self.ctrl_dt, velocity_command_body=pending["command"],
            policy_action=pending["policy_action"],
            environment_action=np.asarray(state.info["current_actions"])[0],
            policy_joint_names=self.policy_joint_names, environment_joint_names=self.environment_joint_names,
            terminated=terminated, truncated=truncated,
        )
        self._records.append((record, pending["submitted_action"]))
        self._pending = None
        if bool(np.asarray(state.info.get("runtime_payload_attached_this_step", [False]))[0]):
            payload = self.env.cfg.runtime_payload
            self.event(
                "payload_applied", sim_time_s=self._run_steps * self.ctrl_dt,
                phase="before_physics", added_mass_kg=float(state.info["runtime_payload_added_mass_kg"][0]),
                body_name=payload.body_name, attachment_pos_m=payload.attachment_pos_m,
                half_size_m=payload.half_size_m,
            )
        self._run_steps += 1
        self._episode_step += 1
        self.last_done = terminated or truncated
        self._ended = self.last_done
        if terminated:
            self.event("terminated")
        if truncated:
            self.event("truncated")
        if len(self._records) >= self.config.chunk_steps or self.last_done:
            self.flush()

    def on_update(self, stats: dict[str, Any]) -> None:
        self._update_count += 1
        self.event("update_completed", update_index=self._update_count, **stats)

    def flush(self) -> None:
        if not self._records:
            return
        records = [item[0] for item in self._records]
        arrays = {
            "run_id": np.array(self.run_id), "env_index": np.zeros(len(records), dtype=np.int64),
            "episode_id": np.array([r.before.episode_id for r in records]),
            "control_step": np.array([r.before.control_step for r in records]),
            "time_before_s": np.array([r.before.sim_time_s for r in records]),
            "time_after_s": np.array([r.after.sim_time_s for r in records]),
            "ctrl_dt": np.array(self.ctrl_dt),
            "velocity_command_body": np.stack([r.velocity_command_body for r in records]),
            "policy_action": np.stack([r.policy_action for r in records]),
            "environment_action": np.stack([r.environment_action for r in records]),
            "submitted_environment_action": np.stack([item[1] for item in self._records]),
            "policy_joint_names": np.array(self.policy_joint_names),
            "environment_joint_names": np.array(self.environment_joint_names),
            "terminated": np.array([r.terminated for r in records]),
            "truncated": np.array([r.truncated for r in records]),
        }
        for phase in ("before", "after"):
            for field in ("base_quat_wxyz", "linear_velocity_world", "angular_velocity_world"):
                arrays[f"{field}_{phase}"] = np.stack([getattr(getattr(r, phase), field) for r in records])
        destination = self._parts / f"part_{self._chunk_index:06d}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        stream = temporary.open("xb")
        try:
            with stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._committed_steps += len(self._records)
        self._records.clear()
        self._chunk_index += 1

    def close(self, status: str, error: str | None = None) -> None:
        if self._closed:
            return
        if status not in ("completed", "interrupted", "failed"):
            raise ValueError("experiment status must be completed, interrupted or failed")
        try:
            self.flush()
            self.event(status, error=error, incomplete_transition=self._pending is not None)
            os.fsync(self._events.fileno())
            self._metadata.update(
                status=status, error=error, recorded_control_steps=self._committed_steps,
                simulated_control_steps=self._run_steps, completed_updates=self._update_count,
                incomplete_transition=self._pending is not None,
            )
            self._write_metadata()
        finally:
            self._events.close()
            self._closed = True


@contextmanager
def experiment_recording_run(session: Any):
    """Finalize raw records on normal return, Ctrl-C, SIGTERM or an exception."""
    status, error = "completed", None
    previous_handler = None
    install_signal = session.experiment_recorder is not None and threading.current_thread() is threading.main_thread()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    if install_signal:
        previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    except KeyboardInterrupt as exc:
        status, error = "interrupted", str(exc)
        raise
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            session.finish_experiment_recording(status, error)
        finally:
            if install_signal:
                signal.signal(signal.SIGTERM, previous_handler)
