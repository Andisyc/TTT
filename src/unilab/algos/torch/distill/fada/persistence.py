"""Single-flight paired persistence for one official FADA DAgger iteration.

Owns at most one writer thread at a time. One commit writes, in order:

1. a distinct ``online_buffer_<run_token>_iter_<it:04d>.npz`` snapshot (plus the
   ``_val`` sidecar) through the existing Replay owner;
2. the latest official student checkpoint through the existing atomic
   ``torch.save`` + ``replace`` owner, carrying the counters and metadata
   captured at ``submit`` time and the paired online-Replay reference;
3. at ``save_every_iter`` and at the final iteration, an immutable
   ``student_iter_<it:04d>.pt`` hardlink of the just-committed checkpoint.

The caller keeps ``learner`` and ``replay`` frozen (no update, no Replay
admission, no sampling mutation) from ``submit`` until ``wait`` returns. This
owner neither clones nor synchronizes that live state and never saves/restores
RNG state: a failed commit leaves the process exactly where it was except for
this owner's own files. Files already on disk (resume checkpoint, resume
snapshot, history checkpoints from an earlier run) are never read, rewritten,
or deleted. Only snapshots this owner itself created, no longer referenced by
the committed latest checkpoint, and never linked by a history checkpoint are
removed, using only its own retention record -- never by scanning checkpoints.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from unilab.algos.torch.distill.fada.checkpoint import save_official_fada_checkpoint
from unilab.algos.torch.distill.fada.official_learning import OfficialFADALearner
from unilab.algos.torch.distill.fada.replay import OfficialFADAReplayBuffer

__all__ = ["FADACommitResult", "FADAIterationCommitter"]

_REQUIRED_WALL_KEYS = ("online_replay", "iteration_model")
_OPTIONAL_WALL_KEYS = ("total",)


@dataclass(frozen=True)
class FADACommitResult:
    """Evidence of one committed iteration: counters, pair reference and writer wall time."""

    completed_iterations: int
    online_buffer_path: Path | None
    wall_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.completed_iterations, bool)
            or not isinstance(self.completed_iterations, int)
            or self.completed_iterations < 1
        ):
            raise ValueError(
                "official FADA commit result requires a positive completed_iterations, "
                f"got {self.completed_iterations!r}"
            )
        if self.online_buffer_path is not None and not isinstance(self.online_buffer_path, Path):
            object.__setattr__(self, "online_buffer_path", Path(self.online_buffer_path))
        wall = self.wall_seconds
        if not isinstance(wall, Mapping):
            raise TypeError(
                f"official FADA commit wall_seconds must be a mapping, got {type(wall).__name__}"
            )
        allowed = (*_REQUIRED_WALL_KEYS, *_OPTIONAL_WALL_KEYS)
        missing = [name for name in _REQUIRED_WALL_KEYS if name not in wall]
        unknown = sorted(set(wall) - set(allowed))
        if missing or unknown:
            raise ValueError(
                f"official FADA commit wall_seconds must report {list(_REQUIRED_WALL_KEYS)} "
                f"and optionally {list(_OPTIONAL_WALL_KEYS)}: missing={missing} "
                f"unknown={unknown}"
            )
        frozen: dict[str, float] = {}
        for name in allowed:
            if name not in wall:
                continue
            value = wall[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"official FADA commit wall_seconds.{name} must be a number")
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0.0:
                raise ValueError(f"official FADA commit wall_seconds.{name} must be finite >= 0")
            frozen[name] = seconds
        object.__setattr__(self, "wall_seconds", MappingProxyType(frozen))


@dataclass(frozen=True)
class _CommitRequest:
    """Everything the writer may touch, captured by the submitting thread."""

    learner: OfficialFADALearner
    replay: OfficialFADAReplayBuffer
    completed_iterations: int
    samples_seen: int
    source_identity: Mapping[str, Any]
    rollout_step_count: int


@dataclass
class _CommitJob:
    request: _CommitRequest
    thread: threading.Thread | None = None
    result: FADACommitResult | None = None
    error: BaseException | None = None


class FADAIterationCommitter:
    """Commit one iteration's Replay snapshot and checkpoint in a single writer thread.

    ``submit`` starts the only allowed write and ``wait`` is the only place its
    outcome is consumed; a completed-but-unwaited commit still counts as pending
    so the caller can never start a second writer behind the first. The live
    learner, optimizers and Replay must stay frozen until ``wait`` returns: this
    owner does not clone them, does not synchronize their devices, and does not
    roll back RNG state if the write fails.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        checkpoints_dir: Path,
        run_token: str,
        save_every_iter: int,
        dagger_iters: int,
        save_online_buffer: bool,
    ) -> None:
        self._checkpoint_path = self._as_path(checkpoint_path, "checkpoint_path")
        self._checkpoints_dir = self._as_path(checkpoints_dir, "checkpoints_dir")
        if not isinstance(run_token, str) or not run_token:
            raise ValueError(
                f"official FADA run_token must be a non-empty string, got {run_token!r}"
            )
        if any(separator in run_token for separator in ("/", "\\", "\0")):
            raise ValueError(f"official FADA run_token must be path-free, got {run_token!r}")
        self._run_token = run_token
        self._save_every_iter = self._positive_int(save_every_iter, "save_every_iter")
        self._dagger_iters = self._positive_int(dagger_iters, "dagger_iters")
        if not isinstance(save_online_buffer, bool):
            raise TypeError(
                "official FADA save_online_buffer must be a bool, "
                f"got {type(save_online_buffer).__name__}"
            )
        self._save_online_buffer = save_online_buffer
        self._lock = threading.Lock()
        self._job: _CommitJob | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._last_submitted = 0
        # Writer-thread-owned retention record: (snapshot, kept for history).
        # The main thread never reads or writes it.
        self._retention: tuple[Path | None, bool] = (None, False)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-flight surface
    # ------------------------------------------------------------------
    @property
    def pending(self) -> bool:
        """True while a commit is in flight or its outcome has not been waited for."""

        with self._lock:
            return self._job is not None

    def submit(
        self,
        learner: OfficialFADALearner,
        replay: OfficialFADAReplayBuffer,
        *,
        completed_iterations: int,
        samples_seen: int,
        source_identity: Mapping[str, Any],
        rollout_step_count: int,
    ) -> None:
        """Capture the counters and start the one allowed writer; never blocks on it."""

        if not isinstance(replay, OfficialFADAReplayBuffer):
            raise TypeError(
                "official FADA commit requires the official replay owner, "
                f"got {type(replay).__name__}"
            )
        if not isinstance(learner, OfficialFADALearner):
            raise TypeError(
                "official FADA commit requires the official learner, "
                f"got {type(learner).__name__}"
            )
        iteration = self._positive_int(completed_iterations, "completed_iterations")
        if iteration > self._dagger_iters:
            raise ValueError(
                f"official FADA commit iteration {iteration} is outside 1..{self._dagger_iters}"
            )
        samples = self._non_negative_int(samples_seen, "samples_seen")
        rollout = self._non_negative_int(rollout_step_count, "rollout_step_count")
        if not isinstance(source_identity, Mapping):
            raise TypeError(
                "official FADA commit source_identity must be a mapping, "
                f"got {type(source_identity).__name__}"
            )
        identity = dict(source_identity)
        with self._lock:
            if self._closed:
                raise RuntimeError("official FADA iteration committer is closed")
            if self._job is not None:
                raise RuntimeError(
                    "official FADA iteration committer already holds an unwaited commit; "
                    "wait for it before submitting the next iteration"
                )
            if self._failure is not None:
                raise RuntimeError(
                    "official FADA iteration committer already failed; the run must not continue"
                ) from self._failure
            if iteration <= self._last_submitted:
                # The latest committed checkpoint references this iteration's
                # snapshot path; rewriting it would break that committed pair.
                raise ValueError(
                    f"official FADA commit iteration {iteration} was already submitted; "
                    f"iterations must increase past {self._last_submitted}"
                )
            if self._history_due(iteration):
                history_path = self._history_path(iteration)
                # Reject before anything is saved rather than after the latest
                # checkpoint moved: a file that already exists belongs to an
                # earlier run and is never replaced or overwritten.
                if history_path.exists():
                    raise FileExistsError(
                        f"official FADA history checkpoint already exists: {history_path}"
                    )
            job = _CommitJob(
                request=_CommitRequest(
                    learner=learner,
                    replay=replay,
                    completed_iterations=iteration,
                    samples_seen=samples,
                    source_identity=identity,
                    rollout_step_count=rollout,
                )
            )
            thread = threading.Thread(
                target=self._run,
                args=(job,),
                name="fada-iteration-commit",
                daemon=True,
            )
            job.thread = thread
            previous_iteration = self._last_submitted
            self._job = job
            self._last_submitted = iteration
            try:
                thread.start()
            except BaseException:
                # No writer exists, so nothing may look pending afterwards.
                self._job = None
                self._last_submitted = previous_iteration
                raise

    def wait(self) -> FADACommitResult | None:
        """Join the writer and return its result; re-raise its original failure."""

        with self._lock:
            job = self._job
        if job is None:
            return None
        try:
            return self._join(job)
        finally:
            with self._lock:
                if self._job is job:
                    self._job = None
                if job.error is not None:
                    self._failure = job.error

    def close(self) -> None:
        """Wait for the writer, mark this owner closed, and propagate any failure."""

        with self._lock:
            job = self._job
            self._closed = True
        if job is None:
            return
        try:
            self._join(job)
        finally:
            with self._lock:
                if self._job is job:
                    self._job = None
                if job.error is not None:
                    self._failure = job.error

    @staticmethod
    def _join(job: _CommitJob) -> FADACommitResult | None:
        """Join the single writer; a captured failure is re-raised unchanged."""

        assert job.thread is not None
        job.thread.join()
        if job.error is not None:
            raise job.error
        return job.result

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------
    def _run(self, job: _CommitJob) -> None:
        try:
            job.result = self._write(job.request)
        except BaseException as error:  # reported verbatim to the waiting caller
            job.error = error

    def _write(self, request: _CommitRequest) -> FADACommitResult:
        iteration = request.completed_iterations
        started = time.perf_counter()
        wall: dict[str, float] = {}
        snapshot: Path | None = None
        target = self._snapshot_path(iteration) if self._save_online_buffer else None
        if target is None:
            wall["online_replay"] = 0.0
        else:
            # Order matters: the Replay snapshot must be durable before the
            # checkpoint that references it replaces the previous pair.
            online_start = time.perf_counter()
            try:
                snapshot = Path(request.replay.save_online_buffer(target))
            except BaseException:
                # A partial write of this attempt was never referenced, so the
                # previously committed pair stands; cleanup never masks the
                # writer's original error.
                self._discard_quietly(target)
                raise
            wall["online_replay"] = time.perf_counter() - online_start
        model_start = time.perf_counter()
        try:
            save_official_fada_checkpoint(
                self._checkpoint_path,
                request.learner,
                completed_iterations=iteration,
                samples_seen=request.samples_seen,
                source_identity=request.source_identity,
                rollout_step_count=request.rollout_step_count,
                online_buffer_path=snapshot,
            )
        except BaseException:
            # The atomic replace did not happen, so the old latest checkpoint and
            # its old pairing remain the committed truth; this attempt's own
            # snapshot is unreferenced and is not kept.
            if snapshot is not None:
                self._discard_quietly(snapshot)
            raise
        wall["iteration_model"] = time.perf_counter() - model_start
        # The latest checkpoint now references this snapshot and this attempt
        # created it, so record that before anything else can fail: a later
        # commit may retire it, an earlier one may not.
        retained = self._history_due(iteration)
        previous, previous_retained = self._retention
        self._retention = (snapshot, retained)
        if retained:
            # The latest checkpoint is atomically replaced, so this inode stays
            # an immutable history snapshot and keeps its own Replay pairing.
            self._history_path(iteration).hardlink_to(self._checkpoint_path)
        if previous is not None and not previous_retained and previous != snapshot:
            # Superseded and no retained history checkpoint references it. A
            # snapshot this owner did not create is not in this record, so a
            # resume snapshot or an earlier run's files are never removed here.
            # Retention is best-effort housekeeping after a committed iteration.
            self._discard_quietly(previous)
        wall["total"] = time.perf_counter() - started
        return FADACommitResult(
            completed_iterations=iteration,
            online_buffer_path=snapshot,
            wall_seconds=wall,
        )

    def _discard(self, snapshot: Path) -> None:
        snapshot.unlink(missing_ok=True)
        self._validation_path(snapshot).unlink(missing_ok=True)

    def _discard_quietly(self, snapshot: Path) -> None:
        try:
            self._discard(snapshot)
        except OSError:
            pass

    @staticmethod
    def _validation_path(snapshot: Path) -> Path:
        return snapshot.with_name(f"{snapshot.stem}_val.npz")

    # ------------------------------------------------------------------
    # Naming and validation
    # ------------------------------------------------------------------
    def _snapshot_path(self, iteration: int) -> Path:
        return self._checkpoints_dir / f"online_buffer_{self._run_token}_iter_{iteration:04d}.npz"

    def _history_due(self, iteration: int) -> bool:
        return iteration % self._save_every_iter == 0 or iteration == self._dagger_iters

    def _history_path(self, iteration: int) -> Path:
        return self._checkpoints_dir / f"student_iter_{iteration:04d}.pt"

    @staticmethod
    def _as_path(value: Any, name: str) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(
                f"official FADA {name} must be a path, got {type(value).__name__}"
            )
        return Path(value)

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"official FADA {name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _non_negative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"official FADA {name} must be a non-negative integer, got {value!r}"
            )
        return value