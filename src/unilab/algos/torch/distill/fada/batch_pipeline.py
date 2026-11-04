"""Bounded per-stage batch pipeline: CPU prefetch, pinned staging, asynchronous H2D.

One stage plans a fixed number of updates. A single sampler thread owns the sampling callable
for the whole stage and draws exactly ``planned_updates`` batches in the existing order (the
replay owner draws Planner first and then the five IDM roles, skipping zero quotas). The thread
never reseeds or creates a generator, and it runs no model, augmentation, Torch random operator
or optimizer work: it only calls ``sample_cpu(mode=..., device="cpu")`` and copies the returned
CPU tensors into preallocated slots.

Slots are allocated once per stage and reused. ``limits.prefetch_batches`` is the total slot
count including the slot being consumed and any in-flight copies, so a single slot is legal but
cannot overlap sampling with learning. A slot is acquired before sampling, never after.

The budget is checked against every retained staging copy (one host copy on CPU; a pinned host
copy plus a device copy on CUDA) plus a conservative transient reserve for the CPU sample result
and the Planner concatenation peak. A stage that does not fit is rejected before a slot is
allocated or the sampler is called.

On CPU the batches are the slot tensors themselves and no CUDA object is created. On CUDA the
copy stream runs the H2D transfer, the leased batch makes the consumer's current stream wait on
that per-slot copy event, and a lease exit records a per-slot consumed event on the consumer
stream. A slot is only rewritten after its own copy and consumed events have completed, and the
pinned source is never rewritten before its copy finished. Nothing synchronizes the whole device
per batch.

A lease is the only way to obtain batches and they are valid solely inside it. A successful
lease consumes its batch; a lease that raises cancels the rest of the stage. ``__exit__`` on a
clean stage requires drawn == consumed == planned with nothing pending, joins the sampler and
awaits this pipeline's own outstanding events; a failing stage cancels, joins and cleans up
without replacing the caller's exception. The replay buffer must stay read-only until the
sampler is joined.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from unilab.algos.torch.distill.fada.collection_contract import (
    OfficialFADACollectionSpec,
    OfficialFADASourceBatch,
    official_window_batch_layout,
)
from unilab.algos.torch.distill.fada.execution_contract import (
    FADAExecutionLimits,
    Layout,
    layout_nbytes,
    validate_phase_boundary,
    validate_prefetch_budget,
)
from unilab.algos.torch.distill.fada.official_learning import (
    canonical_official_training_config,
)
from unilab.algos.torch.distill.fada.replay_sampling import official_replay_source_counts

__all__ = ["OfficialFADABatchPipeline"]

_SAMPLE_DEVICE = "cpu"
# The replay sampler concatenates exactly these physical sources into the Planner batch, in this
# order; the pipeline only needs their total row count for the slot layout.
_PLANNER_SOURCES: tuple[str, ...] = ("optimal", "suboptimal", "online")
# The CPU sample result and the Planner concatenation peak are transient, but finite, so they
# are reserved up front instead of being discovered after a slot already exists.
_TRANSIENT_PAYLOADS = 2


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "official FADA batch pipeline needs a CUDA runtime for device "
                f"{resolved}, none is available"
            )
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        return torch.device("cuda", index)
    if resolved.type != "cpu":
        raise ValueError(
            f"official FADA batch pipeline supports cpu/cuda devices, got {resolved.type}"
        )
    return resolved


def _new_storage(layout: Layout, *, pinned: bool) -> dict[str, torch.Tensor]:
    return {
        name: torch.empty(shape, dtype=dtype, pin_memory=pinned)
        for name, (shape, dtype) in layout.items()
    }


class _Slot:
    """One preallocated staging buffer owned by exactly one side at a time.

    Slot membership in the free/ready queues and the per-slot fill counter are only touched
    while the pipeline lock is held; the buffer contents are written only by the sampler thread
    while it owns the slot, and read only by the consumer during its lease.
    """

    __slots__ = (
        "planner_host",
        "idm_host",
        "planner_device",
        "idm_device",
        "ready",
        "consumed",
        "fills",
    )

    def __init__(
        self,
        planner_host: dict[str, torch.Tensor],
        idm_host: dict[str, dict[str, torch.Tensor]],
        planner_device: dict[str, torch.Tensor] | None,
        idm_device: dict[str, dict[str, torch.Tensor]] | None,
    ) -> None:
        self.planner_host = planner_host
        self.idm_host = idm_host
        self.planner_device = planner_device
        self.idm_device = idm_device
        self.ready: torch.cuda.Event | None = None
        self.consumed: torch.cuda.Event | None = None
        # Zero means the slot was never filled, so neither event has been recorded yet.
        self.fills = 0


class OfficialFADABatchPipeline:
    """Prefetch sampled FADA training batches for one bounded stage.

    ``sample_cpu`` is the existing replay sampling callable, invoked as
    ``sample_cpu(mode=mode, device="cpu")`` and returning ``(idm_batches_by_source,
    planner_batch)``. The pipeline owns the sampler thread, the slot pool, the copy stream and
    its events; it owns no model, optimizer, checkpoint or RNG.
    """

    def __init__(
        self,
        *,
        sample_cpu: Callable[..., tuple[Mapping[str, Any], Any]],
        spec: OfficialFADACollectionSpec,
        training_config: Mapping[str, Any],
        mode: str,
        planned_updates: int,
        device: str | torch.device,
        limits: FADAExecutionLimits,
    ) -> None:
        if not callable(sample_cpu):
            raise TypeError("official FADA batch pipeline requires a callable sample_cpu")
        if not isinstance(spec, OfficialFADACollectionSpec):
            raise TypeError(
                "official FADA batch pipeline requires an OfficialFADACollectionSpec, "
                f"got {type(spec).__name__}"
            )
        if not isinstance(limits, FADAExecutionLimits):
            raise TypeError(
                "official FADA batch pipeline requires FADAExecutionLimits, "
                f"got {type(limits).__name__}"
            )
        if isinstance(planned_updates, bool) or not isinstance(planned_updates, int):
            raise ValueError(
                f"official FADA planned_updates must be an integer, got {planned_updates!r}"
            )
        if planned_updates < 0:
            raise ValueError(
                f"official FADA planned_updates must be non-negative, got {planned_updates!r}"
            )

        config = canonical_official_training_config(training_config)
        # Quotas come from the same owner the replay sampler uses, in the same key order, so the
        # sampler's own draws produce exactly these per-source row counts.
        planner_counts = official_replay_source_counts(config, mode=mode, planner=True)
        idm_counts = official_replay_source_counts(config, mode=mode, planner=False)
        planner_rows = sum(planner_counts[source] for source in _PLANNER_SOURCES)
        if planner_rows <= 0:
            raise ValueError(
                f"official FADA planner quota is empty in mode {mode!r}; nothing to prefetch"
            )

        self._sample_cpu = sample_cpu
        self._mode = mode
        self._planned_updates = planned_updates
        self._requested_device = device
        self._limits = limits
        self._planner_layout = official_window_batch_layout(spec, planner_rows)
        self._idm_layouts: dict[str, Layout] = {
            role: official_window_batch_layout(spec, size)
            for role, size in idm_counts.items()
            if size
        }

        self._lock = threading.Condition()
        self._device = torch.device("cpu")
        self._cuda = False
        self._copy_stream: torch.cuda.Stream | None = None
        self._slots: list[_Slot] = []
        self._free: deque[_Slot] = deque()
        self._ready: deque[_Slot] = deque()
        self._worker: threading.Thread | None = None
        self._error: BaseException | None = None
        self._stop = False
        self._cancelled = False
        self._worker_done = False
        self._entered = False
        self._closed = False
        self._drawn = 0
        self._consumed = 0
        self._pending = 0

    # ------------------------------------------------------------------
    # Read-only stage counters
    # ------------------------------------------------------------------
    @property
    def drawn_updates(self) -> int:
        with self._lock:
            return self._drawn

    @property
    def consumed_updates(self) -> int:
        with self._lock:
            return self._consumed

    @property
    def pending_batches(self) -> int:
        with self._lock:
            return self._pending

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> OfficialFADABatchPipeline:
        if self._entered:
            raise RuntimeError(
                "official FADA batch pipeline serves exactly one stage; "
                "build a new pipeline for the next stage"
            )
        self._entered = True
        try:
            self._prepare()
        except BaseException:
            self._abort()
            self._closed = True
            self._dispose()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        try:
            if exc is None:
                try:
                    self._finish_success()
                except BaseException:
                    self._abort()
                    raise
            else:
                self._abort()
        finally:
            self._closed = True
            self._dispose()
        return False

    def _prepare(self) -> None:
        if self._planned_updates == 0:
            # A zero-update stage samples nothing, starts no worker and initializes no CUDA
            # object; its boundary is trivially satisfied by __exit__.
            return
        self._device = _resolve_device(self._requested_device)
        self._cuda = self._device.type == "cuda"
        # Reject an undersized budget before any slot or stream exists and before any sampling.
        self._validate_budget()
        if self._cuda:
            with torch.cuda.device(self._device):
                self._copy_stream = torch.cuda.Stream(device=self._device)
        for _index in range(self._limits.prefetch_batches):
            slot = self._allocate_slot()
            self._slots.append(slot)
            self._free.append(slot)
        if self._cuda:
            assert self._copy_stream is not None
            # Cached device storage may still have work queued on its allocation
            # stream, even when the first copy overwrites every element.
            self._copy_stream.wait_stream(torch.cuda.current_stream(self._device))
        worker = threading.Thread(
            target=self._run,
            name="fada-batch-pipeline",
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def _validate_budget(self) -> None:
        per_step = [self._planner_layout, *self._idm_layouts.values()]
        # Pinned host and device copies are both retained for every slot on CUDA.
        retained = [*per_step, *per_step] if self._cuda else list(per_step)
        retained_bytes = validate_prefetch_budget(retained, self._limits)
        transient_bytes = (
            sum(layout_nbytes(layout) for layout in per_step) * _TRANSIENT_PAYLOADS
        )
        total_bytes = retained_bytes + transient_bytes
        if total_bytes > self._limits.prefetch_budget_bytes:
            raise ValueError(
                "prefetch staging plus transient sampling peak exceeds the prefetch budget: "
                f"budget={self._limits.prefetch_budget_bytes} retained={retained_bytes} "
                f"transient={transient_bytes} total={total_bytes}"
            )

    def _allocate_slot(self) -> _Slot:
        planner_host = _new_storage(self._planner_layout, pinned=self._cuda)
        idm_host = {
            role: _new_storage(layout, pinned=self._cuda)
            for role, layout in self._idm_layouts.items()
        }
        if not self._cuda:
            return _Slot(planner_host, idm_host, None, None)
        with torch.cuda.device(self._device):
            planner_device = {
                name: torch.empty(shape, dtype=dtype, device=self._device)
                for name, (shape, dtype) in self._planner_layout.items()
            }
            idm_device = {
                role: {
                    name: torch.empty(shape, dtype=dtype, device=self._device)
                    for name, (shape, dtype) in layout.items()
                }
                for role, layout in self._idm_layouts.items()
            }
            slot = _Slot(planner_host, idm_host, planner_device, idm_device)
            slot.ready = torch.cuda.Event()
            slot.consumed = torch.cuda.Event()
        return slot

    def _finish_success(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        self._join_worker()
        self._await_own_events()
        with self._lock:
            drawn, consumed, pending = self._drawn, self._consumed, self._pending
        validate_phase_boundary(
            planned_updates=self._planned_updates,
            drawn_updates=drawn,
            consumed_updates=consumed,
            pending_batches=pending,
        )

    def _abort(self) -> None:
        """Cancel, wake and join the sampler; never replaces the caller's exception."""

        with self._lock:
            self._stop = True
            self._cancelled = True
            self._lock.notify_all()
        try:
            self._join_worker(suppress=True)
        except BaseException:
            pass
        try:
            self._await_own_events()
        except BaseException:
            pass

    def _join_worker(self, *, suppress: bool = False) -> None:
        worker = self._worker
        if worker is not None:
            worker.join()
            self._worker = None
        with self._lock:
            error = self._error
        if error is not None and not suppress:
            raise error

    def _await_own_events(self) -> None:
        if not self._cuda:
            return
        with torch.cuda.device(self._device):
            for slot in self._slots:
                if slot.fills == 0:
                    continue
                for event in (slot.ready, slot.consumed):
                    if event is not None and not event.query():
                        event.synchronize()
        # A fill that failed to submit its ready event is not covered by any per-slot event, so
        # the copy stream itself is fenced before the slots are dropped.
        self._fence_copy_stream()

    def _dispose(self) -> None:
        with self._lock:
            self._ready.clear()
            self._free.clear()
        self._slots = []
        self._copy_stream = None

    # ------------------------------------------------------------------
    # Sampler thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            for _draw in range(self._planned_updates):
                slot = self._acquire_slot()
                if slot is None:
                    return
                try:
                    self._await_slot_release(slot)
                    idm_batches, planner_batch = self._sample_cpu(
                        mode=self._mode, device=_SAMPLE_DEVICE
                    )
                    self._stage(slot, idm_batches, planner_batch)
                except BaseException:
                    with self._lock:
                        self._release_locked(slot)
                    raise
                self._publish(slot)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the next lease or exit
            with self._lock:
                self._error = exc
                self._lock.notify_all()
        finally:
            with self._lock:
                self._worker_done = True
                self._lock.notify_all()

    def _acquire_slot(self) -> _Slot | None:
        with self._lock:
            while True:
                if self._stop or self._cancelled:
                    return None
                if self._free:
                    slot = self._free.popleft()
                    self._pending += 1
                    return slot
                self._lock.wait()

    def _await_slot_release(self, slot: _Slot) -> None:
        if not self._cuda:
            return
        assert self._copy_stream is not None
        if slot.fills == 0:
            return
        with torch.cuda.device(self._device):
            # Device memory may only be overwritten after the consumer's stream passed the
            # position where this slot was last consumed; the copy stream waits on that event
            # instead of stalling the host.
            assert slot.consumed is not None and slot.ready is not None
            self._copy_stream.wait_event(slot.consumed)
            # The pinned host copies are rewritten by this thread, so the previous H2D transfer
            # must have completed before that happens.
            slot.ready.synchronize()

    def _stage(
        self,
        slot: _Slot,
        idm_batches: Mapping[str, Any],
        planner_batch: Any,
    ) -> None:
        if not isinstance(idm_batches, Mapping):
            raise TypeError(
                "official FADA sample_cpu must return a mapping of IDM role to batch, "
                f"got {type(idm_batches).__name__}"
            )
        roles = set(idm_batches)
        expected = set(self._idm_layouts)
        if roles != expected:
            raise ValueError(
                "official FADA sample_cpu IDM roles contradict the planned quotas: "
                f"missing={sorted(expected - roles)} extra={sorted(roles - expected)}"
            )
        self._stage_batch(slot.planner_host, planner_batch, self._planner_layout, "planner")
        for role, layout in self._idm_layouts.items():
            self._stage_batch(slot.idm_host[role], idm_batches[role], layout, role)
        if self._cuda:
            self._launch_copy(slot)

    def _stage_batch(
        self,
        storage: dict[str, torch.Tensor],
        batch: Any,
        layout: Layout,
        label: str,
    ) -> None:
        if not isinstance(batch, OfficialFADASourceBatch):
            raise TypeError(
                f"official FADA sample_cpu returned {type(batch).__name__} for {label!r}, "
                "expected OfficialFADASourceBatch"
            )
        for name, (shape, dtype) in layout.items():
            value = getattr(batch, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"official FADA {label}.{name} must be a torch.Tensor")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"official FADA {label}.{name} must have shape {shape}, "
                    f"got {tuple(value.shape)}"
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"official FADA {label}.{name} must have dtype {dtype}, got {value.dtype}"
                )
            if value.device.type != _SAMPLE_DEVICE:
                raise ValueError(
                    "official FADA batch pipeline samples on CPU only, "
                    f"{label}.{name} is on {value.device}"
                )
            storage[name].copy_(value)

    def _launch_copy(self, slot: _Slot) -> None:
        assert self._copy_stream is not None
        assert slot.planner_device is not None and slot.idm_device is not None
        with torch.cuda.device(self._device):
            try:
                with torch.cuda.stream(self._copy_stream):
                    for name, host in slot.planner_host.items():
                        slot.planner_device[name].copy_(host, non_blocking=True)
                    for role, host_storage in slot.idm_host.items():
                        device_storage = slot.idm_device[role]
                        for name, host in host_storage.items():
                            device_storage[name].copy_(host, non_blocking=True)
                    assert slot.ready is not None
                    slot.ready.record(self._copy_stream)
            except BaseException:
                # A partially submitted fill has no complete ready event, so the per-slot event
                # cannot stand in for these transfers. Fence this pipeline's own copy stream
                # before the pinned sources and device buffers can be rewritten or freed, then
                # keep the original failure.
                try:
                    self._fence_copy_stream()
                except BaseException:
                    pass
                raise
        slot.fills += 1

    def _fence_copy_stream(self) -> None:
        """Wait for this pipeline's own copy stream at a failure or stage-end boundary."""

        if self._copy_stream is None:
            return
        with torch.cuda.device(self._device):
            self._copy_stream.synchronize()

    def _publish(self, slot: _Slot) -> None:
        with self._lock:
            self._ready.append(slot)
            self._drawn += 1
            self._lock.notify_all()

    # ------------------------------------------------------------------
    # Lease
    # ------------------------------------------------------------------
    @contextmanager
    def batch(
        self,
    ) -> Iterator[tuple[dict[str, OfficialFADASourceBatch], OfficialFADASourceBatch]]:
        """Borrow the next prefetched batch; it is only valid inside this lease."""

        slot = self._take_slot()
        try:
            yield self._borrow(slot)
        except BaseException:
            self._cancel(slot)
            raise
        else:
            self._consume(slot)

    def _take_slot(self) -> _Slot:
        with self._lock:
            if not self._entered:
                raise RuntimeError(
                    "official FADA batch pipeline lease requires an active stage"
                )
            if self._closed:
                raise RuntimeError(
                    "official FADA batch pipeline stage is finished; "
                    "borrowed batches do not outlive their stage"
                )
            if self._planned_updates == 0:
                raise RuntimeError("official FADA batch pipeline plans no updates")
            while True:
                if self._error is not None:
                    raise self._error
                if self._cancelled:
                    raise RuntimeError("official FADA batch pipeline lease was cancelled")
                if self._ready:
                    slot = self._ready.popleft()
                    break
                if self._worker_done:
                    raise RuntimeError(
                        "official FADA batch pipeline holds no pending batch: "
                        f"drawn={self._drawn} consumed={self._consumed} "
                        f"planned={self._planned_updates}"
                    )
                self._lock.wait()
        if self._cuda and slot.ready is not None:
            with torch.cuda.device(self._device):
                torch.cuda.current_stream(self._device).wait_event(slot.ready)
        return slot

    def _borrow(
        self, slot: _Slot
    ) -> tuple[dict[str, OfficialFADASourceBatch], OfficialFADASourceBatch]:
        if self._cuda:
            assert slot.planner_device is not None and slot.idm_device is not None
            planner_storage, idm_storage = slot.planner_device, slot.idm_device
        else:
            planner_storage, idm_storage = slot.planner_host, slot.idm_host
        idm_batches = {
            role: OfficialFADASourceBatch.from_mapping(idm_storage[role])
            for role in self._idm_layouts
        }
        return idm_batches, OfficialFADASourceBatch.from_mapping(planner_storage)

    def _consume(self, slot: _Slot) -> None:
        if self._cuda:
            self._record_consumed(slot)
        with self._lock:
            self._consumed += 1
            self._release_locked(slot)

    def _cancel(self, slot: _Slot) -> None:
        if self._cuda:
            self._record_consumed(slot)
        with self._lock:
            self._cancelled = True
            self._release_locked(slot)

    def _record_consumed(self, slot: _Slot) -> None:
        assert slot.consumed is not None
        with torch.cuda.device(self._device):
            slot.consumed.record(torch.cuda.current_stream(self._device))

    def _release_locked(self, slot: _Slot) -> None:
        self._pending -= 1
        self._free.append(slot)
        self._lock.notify_all()