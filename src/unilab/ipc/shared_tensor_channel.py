"""Typed, bounded, lossless chunk transport over one shared-memory segment.

The producer copies a chunk into a free slot and only then publishes it; the consumer borrows
that published slot until its read context exits. Slots recycle strictly in publish order, so
a live chunk is never overwritten or dropped and no unread data is evicted.

Slot identity and slot generation come from the transport's own lifetime sequence, which
lives in the shared segment and therefore spans requests; a chunk index inside one request
never determines them. The layout is the only authority for shape and dtype — this module
keeps no field table — so a caller's typed fields (bool/int32/int64 included) travel
unchanged, with the common leading dimension as the slot row capacity.

Synchronization handles travel with the endpoint through spawn startup kwargs, exactly like
the other transports in this package, and never through a running worker's request queue.
Payload and the fixed per-slot metadata storage are both charged against shared-memory
capacity before anything is allocated. The creator owns the segment's lifetime: it unlinks
after workers and read leases have stopped, while an attached worker only ever closes its own
local handles.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from unilab.ipc.memory_budget import raise_if_shared_memory_over_budget
from unilab.ipc.shared_tensor_protocol import (
    MAX_SHARED_METADATA_BYTES,
    SharedTensorChannelAbortedError,
    SharedTensorChannelProtocol,
    SharedTensorFrame,
    TensorLayout,
)

_SPAWN_CTX = mp.get_context("spawn")

__all__ = [
    "SharedTensorChannel",
    "SharedTensorChannelTimeoutError",
    "SharedTensorEndpoint",
]

# Segment header: magic/version, then lifetime counters, then per-slot rows and lengths.
_MAGIC = 0x53484D43484E4B31
_VERSION = 1

_HEADER_MAGIC = 0
_HEADER_VERSION = 1
_HEADER_WRITE_SEQ = 2
_HEADER_READ_SEQ = 3
_HEADER_ABORTED = 4
_HEADER_PRODUCER_BUSY = 5
_HEADER_CONSUMER_BUSY = 6
_FIXED_HEADER_INTS = 7

# Longest single semaphore wait, so abort and timeout checks stay responsive while blocked.
_WAIT_SLICE_SECONDS = 0.05

_HEADER_ALIGNMENT = 64
_SLOT_ALIGNMENT = 64

# Only fixed-width, object-free numeric/bool fields are transportable.
_ALLOWED_DTYPE_KINDS = frozenset("biufc")


class SharedTensorChannelTimeoutError(SharedTensorChannelAbortedError):
    """A bounded wait expired; the transport is permanently invalidated."""


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _require_count(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value!r}")
    return int(value)


def _require_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"timeout must be a number of seconds, got {timeout!r}")
    seconds = float(timeout)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"timeout must be finite and positive, got {timeout!r}")
    return seconds


def _require_transport_id(transport_id: Any) -> str:
    if transport_id is None:
        return f"tensor-{uuid.uuid4().hex}"
    if not isinstance(transport_id, str) or not transport_id:
        raise ValueError("transport_id must be a nonempty string or None")
    return transport_id


def _require_metadata(metadata: Any) -> bytes:
    if isinstance(metadata, (bytes, bytearray, memoryview)):
        blob = bytes(metadata)
    else:
        raise ValueError(
            "chunk metadata must be bytes-like, got "
            f"{type(metadata).__name__}; tensors never travel as metadata"
        )
    if len(blob) > MAX_SHARED_METADATA_BYTES:
        raise ValueError(
            f"chunk metadata is {len(blob)} bytes, above the "
            f"{MAX_SHARED_METADATA_BYTES}-byte per-slot limit"
        )
    return blob


def _normalise_dtype(dtype_spec: Any, field: str) -> np.dtype:
    if isinstance(dtype_spec, np.dtype):
        dtype = dtype_spec
    elif isinstance(dtype_spec, str) and dtype_spec:
        try:
            dtype = np.dtype(dtype_spec)
        except TypeError as exc:
            raise ValueError(f"layout field {field!r} dtype {dtype_spec!r} is invalid") from exc
    else:
        raise ValueError(
            f"layout field {field!r} dtype must be a numpy dtype string, got {dtype_spec!r}"
        )
    if dtype.hasobject:
        raise ValueError(f"layout field {field!r} dtype {dtype} carries Python objects")
    if dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValueError(
            f"layout field {field!r} dtype {dtype} is not a fixed-width numeric/bool dtype"
        )
    return dtype


def _normalise_layout(
    layout: Any,
) -> tuple[tuple[tuple[str, tuple[int, ...], np.dtype], ...], int]:
    """Return the frozen field order and the common leading row capacity."""

    if not isinstance(layout, Mapping):
        raise ValueError("layout must be a mapping of field name to (shape, dtype)")
    if not layout:
        raise ValueError("layout must declare at least one field")
    fields: list[tuple[str, tuple[int, ...], np.dtype]] = []
    capacity: int | None = None
    for name, entry in layout.items():
        if not isinstance(name, str) or not name:
            raise ValueError("layout field names must be nonempty strings")
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"layout field {name!r} must be a (shape, dtype) pair")
        shape, dtype_spec = entry
        if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
            raise ValueError(f"layout field {name!r} shape must be a sequence of dimensions")
        dims = tuple(shape)
        if not dims:
            raise ValueError(f"layout field {name!r} shape must have at least one dimension")
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"layout field {name!r} dimensions must be positive integers, got {dims}"
                )
        dtype = _normalise_dtype(dtype_spec, name)
        if capacity is None:
            capacity = int(dims[0])
        elif int(dims[0]) != capacity:
            raise ValueError(
                f"layout field {name!r} leading row dimension {dims[0]} does not match the "
                f"shared slot capacity {capacity}"
            )
        fields.append((name, tuple(int(dim) for dim in dims), dtype))
    assert capacity is not None
    return tuple(fields), capacity


@dataclass(frozen=True)
class _LayoutPlan:
    """Byte plan of one segment; identical on creator and attacher for a layout/slot pair."""

    fields: tuple[tuple[str, tuple[int, ...], np.dtype], ...]
    capacity: int
    slots: int
    field_offsets: tuple[tuple[str, int], ...]
    slot_stride: int
    payload_base: int
    metadata_base: int
    header_ints: int
    header_nbytes: int
    rows_start: int
    lengths_start: int
    total_bytes: int

    @property
    def layout(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {name: (dims, dtype.str) for name, dims, dtype in self.fields}


def _plan_layout(layout: Any, slots: Any) -> _LayoutPlan:
    fields, capacity = _normalise_layout(layout)
    slot_count = _require_count(slots, "shared tensor channel slots", minimum=1)
    header_ints = _FIXED_HEADER_INTS + 2 * slot_count
    header_nbytes = _align_up(header_ints * 8, _HEADER_ALIGNMENT)

    offsets: list[tuple[str, int]] = []
    cursor = 0
    for name, dims, dtype in fields:
        cursor = _align_up(cursor, max(int(dtype.alignment), 1))
        offsets.append((name, cursor))
        cursor += math.prod(dims) * dtype.itemsize
    slot_stride = _align_up(max(cursor, 1), _SLOT_ALIGNMENT)

    payload_base = header_nbytes
    metadata_base = payload_base + slot_count * slot_stride
    total_bytes = metadata_base + slot_count * MAX_SHARED_METADATA_BYTES
    return _LayoutPlan(
        fields=fields,
        capacity=capacity,
        slots=slot_count,
        field_offsets=tuple(offsets),
        slot_stride=slot_stride,
        payload_base=payload_base,
        metadata_base=metadata_base,
        header_ints=header_ints,
        header_nbytes=header_nbytes,
        rows_start=_FIXED_HEADER_INTS,
        lengths_start=_FIXED_HEADER_INTS + slot_count,
        total_bytes=total_bytes,
    )


def _check_memory_budget(plan: _LayoutPlan, transport_id: str) -> None:
    """Charge payload and fixed per-slot metadata storage before allocating any of it."""

    raise_if_shared_memory_over_budget(
        {
            "total": plan.total_bytes,
            "payload": plan.slots * plan.slot_stride,
            "metadata": plan.slots * MAX_SHARED_METADATA_BYTES,
            "total_slots": plan.slots,
            "breakdown": (
                f"Shared tensor channel {transport_id}: {plan.total_bytes / 1024**2:.1f} MB "
                f"({plan.slots} slots × ({plan.slot_stride} B payload + "
                f"{MAX_SHARED_METADATA_BYTES} B metadata) + {plan.header_nbytes} B header)"
            ),
        },
        f"Shared tensor channel {transport_id}",
    )


@dataclass(frozen=True)
class SharedTensorEndpoint:
    """Picklable spawn handle for a created channel's segment and synchronization.

    The endpoint travels in a worker's startup kwargs, carries no payload, and only ever
    opens local handles: the creator keeps unlink ownership for the segment's whole lifetime,
    so an attached worker closing its channel never destroys it.
    """

    layout: TensorLayout
    slot_capacity: int
    slots: int
    transport_id: str
    _name: str
    _free: Any
    _ready: Any

    def __post_init__(self) -> None:
        plan = _plan_layout(self.layout, self.slots)
        _require_count(self.slot_capacity, "endpoint slot_capacity", minimum=1)
        if plan.capacity != self.slot_capacity:
            raise ValueError(
                f"endpoint slot_capacity {self.slot_capacity} does not match layout "
                f"capacity {plan.capacity}"
            )
        _require_transport_id(self.transport_id)
        if not isinstance(self._name, str) or not self._name:
            raise ValueError("endpoint requires a nonempty shared-memory name")
        if self._free is None or self._ready is None:
            raise ValueError(
                "endpoint requires its free and ready semaphores; it cannot be built from a "
                "channel that was already closed"
            )

    def attach(self) -> SharedTensorChannel:
        """Attach in a spawned worker; never unlinks the creator's segment."""

        return SharedTensorChannel._attach(self)


class _ReadLease(AbstractContextManager):
    """Holds one published slot; that slot returns to the pool only when the context exits."""

    __slots__ = ("_channel", "_timeout", "_frame", "_held", "_slot", "_sequence")

    def __init__(self, channel: SharedTensorChannel, timeout: float) -> None:
        self._channel = channel
        self._timeout = timeout
        self._frame: SharedTensorFrame | None = None
        self._held = False
        self._slot = -1
        self._sequence = -1

    def __enter__(self) -> SharedTensorFrame:
        if self._held or self._frame is not None:
            raise RuntimeError("a shared tensor read lease can only be entered once")
        channel = self._channel
        plan = channel._plan
        channel._require_usable()
        if channel._consumer_depth == 0:
            channel._begin_role(_HEADER_CONSUMER_BUSY, "consumer")
        channel._consumer_depth += 1
        try:
            channel._acquire_token(channel._ready, self._timeout, "waiting for a ready chunk")
            sequence = int(channel._header[_HEADER_READ_SEQ])
            published = int(channel._header[_HEADER_WRITE_SEQ])
            if sequence >= published:
                raise ValueError(
                    f"read cursor {sequence} is not behind the published cursor {published}"
                )
            slot = sequence % plan.slots
            row_count = int(channel._header[plan.rows_start + slot])
            metadata_length = int(channel._header[plan.lengths_start + slot])
            if not 1 <= row_count <= plan.capacity:
                raise ValueError(
                    f"slot {slot} advertises {row_count} rows outside 1..{plan.capacity}"
                )
            if not 0 <= metadata_length <= MAX_SHARED_METADATA_BYTES:
                raise ValueError(
                    f"slot {slot} advertises {metadata_length} metadata bytes outside "
                    f"0..{MAX_SHARED_METADATA_BYTES}"
                )
            metadata_start = plan.metadata_base + slot * MAX_SHARED_METADATA_BYTES
            metadata = bytes(channel._buf[metadata_start : metadata_start + metadata_length])
            slot_views = channel._slot_views
            assert slot_views is not None
            views = {
                name: field_views[:row_count] for name, field_views in slot_views[slot].items()
            }
        except BaseException as exc:
            channel._leave_consumer()
            if not isinstance(exc, SharedTensorChannelAbortedError):
                channel._invalidate(f"cannot borrow a slot: {exc}")
            raise

        self._slot = slot
        self._sequence = sequence
        self._held = True
        channel._lease_depth += 1
        self._frame = SharedTensorFrame(
            arrays=views,
            row_count=row_count,
            slot=slot,
            slot_generation=sequence // plan.slots,
            metadata=metadata,
        )
        return self._frame

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if not self._held:
            return None
        channel = self._channel
        plan = channel._plan
        self._held = False
        try:
            header = channel._header
            if header is not None:
                header[_HEADER_READ_SEQ] = self._sequence + 1
                header[plan.rows_start + self._slot] = 0
                header[plan.lengths_start + self._slot] = 0
        finally:
            channel._lease_depth -= 1
            self._frame = None
            channel._leave_consumer()
            free = channel._free
            if free is not None:
                free.release()
        return None


class SharedTensorChannel(SharedTensorChannelProtocol):
    """Bounded single-producer/single-consumer chunk channel over one shared segment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "SharedTensorChannel is created with SharedTensorChannel.create(...) or "
            "SharedTensorEndpoint.attach(...)"
        )

    # ---------------------------------------------------------------- construction

    @classmethod
    def create(
        cls,
        layout: TensorLayout,
        *,
        slots: int = 2,
        transport_id: str | None = None,
    ) -> SharedTensorChannel:
        """Create the segment, its bounded semaphore pair and shared counters.

        ``layout`` carries full per-slot shapes, so the common leading dimension is the slot
        row capacity. Payload and metadata bytes are checked against shared-memory capacity
        before anything is allocated; a partial failure releases whatever already existed.
        """

        plan = _plan_layout(layout, slots)
        resolved_id = _require_transport_id(transport_id)
        _check_memory_budget(plan, resolved_id)

        shm: shared_memory.SharedMemory | None = None
        try:
            shm = shared_memory.SharedMemory(create=True, size=plan.total_bytes)
            free = _SPAWN_CTX.Semaphore(plan.slots)
            ready = _SPAWN_CTX.Semaphore(0)
            return cls._from_segment(plan, shm, free, ready, resolved_id, creator=True)
        except BaseException:
            if shm is not None:
                try:
                    shm.close()
                except BaseException:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            raise

    @classmethod
    def _attach(cls, endpoint: SharedTensorEndpoint) -> SharedTensorChannel:
        plan = _plan_layout(endpoint.layout, endpoint.slots)
        if plan.capacity != endpoint.slot_capacity:
            raise ValueError(
                f"endpoint slot_capacity {endpoint.slot_capacity} does not match layout "
                f"capacity {plan.capacity}"
            )
        shm: shared_memory.SharedMemory | None = None
        try:
            shm = shared_memory.SharedMemory(name=endpoint._name, create=False)
            # POSIX shared segments may be rounded up to a page; the plan needs a lower bound.
            if shm.size < plan.total_bytes:
                raise ValueError(
                    f"attached segment {endpoint._name!r} is {shm.size} bytes, smaller than "
                    f"the {plan.total_bytes} bytes the layout and {plan.slots} slots require"
                )
            return cls._from_segment(
                plan,
                shm,
                endpoint._free,
                endpoint._ready,
                endpoint.transport_id,
                creator=False,
            )
        except BaseException:
            if shm is not None:
                shm.close()
            raise

    @classmethod
    def _from_segment(
        cls,
        plan: _LayoutPlan,
        shm: shared_memory.SharedMemory,
        free: Any,
        ready: Any,
        transport_id: str,
        *,
        creator: bool,
    ) -> SharedTensorChannel:
        channel = cls.__new__(cls)
        channel._plan = plan
        channel._shm = shm
        channel._name = shm.name
        channel._free = free
        channel._ready = ready
        channel._transport_id = transport_id
        channel._creator = creator
        channel._closed = False
        channel._unlinked = False
        channel._abort_reason: str | None = None
        channel._lease_depth = 0
        channel._consumer_depth = 0
        channel._header: np.ndarray | None = None
        channel._buf: memoryview | None = None
        channel._slot_views: list[dict[str, np.ndarray]] | None = None
        channel._build_views()
        header = channel._header
        assert header is not None
        if creator:
            header[:] = 0
            header[_HEADER_MAGIC] = _MAGIC
            header[_HEADER_VERSION] = _VERSION
        elif int(header[_HEADER_MAGIC]) != _MAGIC or int(header[_HEADER_VERSION]) != _VERSION:
            raise ValueError(
                f"shared segment {shm.name!r} is not a version {_VERSION} tensor channel"
            )
        return channel

    def _build_views(self) -> None:
        plan = self._plan
        buf = self._shm.buf
        if buf is None:
            raise RuntimeError("shared segment is closed")
        self._buf = buf
        self._header = np.ndarray((plan.header_ints,), dtype=np.int64, buffer=buf, offset=0)
        offsets = dict(plan.field_offsets)
        slot_views: list[dict[str, np.ndarray]] = []
        for slot in range(plan.slots):
            slot_base = plan.payload_base + slot * plan.slot_stride
            views: dict[str, np.ndarray] = {}
            for name, dims, dtype in plan.fields:
                views[name] = np.ndarray(
                    dims, dtype=dtype, buffer=buf, offset=slot_base + offsets[name]
                )
            slot_views.append(views)
        self._slot_views = slot_views

    # ---------------------------------------------------------------- identity

    @property
    def transport_id(self) -> str:
        return self._transport_id

    @property
    def layout(self) -> TensorLayout:
        """Frozen field order with full per-slot shapes and NumPy dtype strings."""

        return self._plan.layout

    @property
    def slot_capacity(self) -> int:
        """Rows per slot: the layout's common leading dimension."""

        return self._plan.capacity

    @property
    def slots(self) -> int:
        return self._plan.slots

    @property
    def metadata_capacity_bytes(self) -> int:
        return MAX_SHARED_METADATA_BYTES

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_creator(self) -> bool:
        return self._creator

    @property
    def size_bytes(self) -> int:
        return self._plan.total_bytes

    @property
    def endpoint(self) -> SharedTensorEndpoint:
        """Picklable spawn handle; send it in worker startup kwargs, not a request queue."""

        return SharedTensorEndpoint(
            layout=self._plan.layout,
            slot_capacity=self._plan.capacity,
            slots=self._plan.slots,
            transport_id=self._transport_id,
            _name=self._name,
            _free=self._free,
            _ready=self._ready,
        )

    # ---------------------------------------------------------------- cursors

    @property
    def write_sequence(self) -> int:
        header = self._header
        return 0 if header is None else int(header[_HEADER_WRITE_SEQ])

    @property
    def read_sequence(self) -> int:
        header = self._header
        return 0 if header is None else int(header[_HEADER_READ_SEQ])

    @property
    def pending_frames(self) -> int:
        """Published-but-unread chunks; unchanged by the number of slots."""

        header = self._header
        if header is None:
            return 0
        return int(header[_HEADER_WRITE_SEQ]) - int(header[_HEADER_READ_SEQ])

    @property
    def aborted(self) -> bool:
        header = self._header
        if header is None:
            return self._closed
        return int(header[_HEADER_ABORTED]) != 0

    def next_write_slot(self) -> tuple[int, int]:
        """Slot and slot generation the producer publishes into next."""

        sequence = self.write_sequence
        return sequence % self._plan.slots, sequence // self._plan.slots

    def next_read_slot(self) -> tuple[int, int]:
        """Slot and slot generation the consumer borrows next."""

        sequence = self.read_sequence
        return sequence % self._plan.slots, sequence // self._plan.slots

    # ---------------------------------------------------------------- producer

    def write(
        self,
        arrays: Mapping[str, np.ndarray],
        metadata: bytes,
        *,
        timeout: float,
    ) -> None:
        """Copy one chunk into a free slot, then publish it; never overwrite unread data.

        Payload is copied into the slot before the ready signal, so a consumer can only ever
        observe a complete chunk. A chunk whose fields, dtypes or row counts leave the layout
        is rejected before any slot is claimed and leaves the transport usable; a failure once
        a slot is claimed invalidates the transport rather than recycling half-written data.
        """

        wait_seconds = _require_timeout(timeout)
        payload, row_count = self._validate_payload(arrays)
        blob = _require_metadata(metadata)
        self._require_usable()
        header = self._header
        assert header is not None

        self._begin_role(_HEADER_PRODUCER_BUSY, "producer")
        slot = -1
        try:
            self._acquire_token(self._free, wait_seconds, "waiting for a free slot")
            sequence = int(header[_HEADER_WRITE_SEQ])
            slot = sequence % self._plan.slots
            slot_views = self._slot_views
            assert slot_views is not None
            views = slot_views[slot]
            for name, values in payload.items():
                views[name][:row_count] = values
            self._publish(sequence, slot, row_count, blob)
        except SharedTensorChannelAbortedError:
            raise
        except BaseException as exc:
            self._invalidate(f"publishing into slot {slot} failed: {exc}")
            raise
        finally:
            self._end_role(_HEADER_PRODUCER_BUSY)

    def _publish(self, sequence: int, slot: int, row_count: int, blob: bytes) -> None:
        plan = self._plan
        buf = self._buf
        header = self._header
        assert buf is not None and header is not None
        if blob:
            metadata_start = plan.metadata_base + slot * MAX_SHARED_METADATA_BYTES
            buf[metadata_start : metadata_start + len(blob)] = blob
        header[plan.rows_start + slot] = row_count
        header[plan.lengths_start + slot] = len(blob)
        # The publish cursor is the last write: the ready token below makes the chunk visible.
        header[_HEADER_WRITE_SEQ] = sequence + 1
        self._ready.release()

    def _validate_payload(self, arrays: Any) -> tuple[dict[str, np.ndarray], int]:
        if not isinstance(arrays, Mapping):
            raise ValueError("chunk payload must be a mapping of field name to numpy array")
        expected = [name for name, _dims, _dtype in self._plan.fields]
        known = set(expected)
        missing = [name for name in expected if name not in arrays]
        unexpected = sorted(repr(name) for name in arrays if name not in known)
        if missing or unexpected:
            raise ValueError(
                f"chunk fields do not match the layout: missing={missing} unexpected={unexpected}"
            )
        row_count: int | None = None
        payload: dict[str, np.ndarray] = {}
        for name, dims, dtype in self._plan.fields:
            values = arrays[name]
            if not isinstance(values, np.ndarray):
                raise ValueError(
                    f"chunk field {name!r} must be a numpy array, got {type(values).__name__}"
                )
            if values.dtype != dtype:
                raise ValueError(
                    f"chunk field {name!r} dtype {values.dtype} does not match layout {dtype}"
                )
            if values.ndim != len(dims) or tuple(values.shape[1:]) != dims[1:]:
                raise ValueError(
                    f"chunk field {name!r} trailing shape {tuple(values.shape[1:])} does not "
                    f"match layout {dims[1:]}"
                )
            rows = int(values.shape[0])
            if not 1 <= rows <= self._plan.capacity:
                raise ValueError(
                    f"chunk field {name!r} has {rows} rows outside 1..{self._plan.capacity}"
                )
            if row_count is None:
                row_count = rows
            elif rows != row_count:
                raise ValueError(
                    f"chunk field {name!r} has {rows} rows while the chunk carries "
                    f"{row_count} leading rows"
                )
            payload[name] = values
        assert row_count is not None
        return payload, row_count

    # ---------------------------------------------------------------- consumer

    def read(self, *, timeout: float) -> AbstractContextManager[SharedTensorFrame]:
        """Borrow one published slot; the slot returns to the pool when the context exits.

        Arrays in the frame are live shared views, valid only inside that context: a consumer
        copies into owned storage before leaving it. The frame's slot and slot generation are
        this transport's lifetime cursors, which is what a chunk descriptor must carry.
        """

        return _ReadLease(self, _require_timeout(timeout))

    # ---------------------------------------------------------------- lifecycle

    def abort(self) -> None:
        """Wake blocked operations and permanently invalidate this transport."""

        self._invalidate("transport aborted")

    def close(self) -> None:
        """Close local handles; never unlinks. Ends only after local read leases exit.

        The segment handle itself is retained so a creator that closes before cleaning up can
        still unlink; ``SharedMemory.close`` already released the mapping it owned.
        """

        if self._closed:
            return
        if self._lease_depth:
            raise RuntimeError(
                f"shared tensor channel {self._transport_id!r} still has "
                f"{self._lease_depth} open read lease(s); copy out and exit them first"
            )
        self._closed = True
        self._header = None
        self._slot_views = None
        self._buf = None
        self._free = None
        self._ready = None
        shm = self._shm
        if shm is not None:
            shm.close()

    def cleanup(self) -> None:
        """Close, and unlink the segment when this channel is its creator.

        Idempotent, and only valid once workers and read leases have stopped. An attached
        channel's cleanup is its own close: the creator keeps unlink ownership, so a worker
        exiting never destroys a segment the parent still owns.
        """

        if self._lease_depth:
            raise RuntimeError(
                f"shared tensor channel {self._transport_id!r} still has "
                f"{self._lease_depth} open read lease(s); copy out and exit them first"
            )
        shm = self._shm
        self.close()
        if not self._creator or self._unlinked or shm is None:
            return
        self._unlinked = True
        self._shm = None
        try:
            shm.unlink()
        except FileNotFoundError:
            # The segment is already gone: that is cleanup's goal, not a failure.
            return

    # ---------------------------------------------------------------- internals

    def _begin_role(self, index: int, role: str) -> None:
        """Best-effort guard that only one process writes and only one process reads."""

        header = self._header
        if header is None:
            raise RuntimeError(f"shared tensor channel {self._transport_id!r} is closed")
        if int(header[index]) != 0:
            raise RuntimeError(
                f"shared tensor channel {self._transport_id!r} already has a {role} in "
                "flight; this transport is single-producer, single-consumer"
            )
        header[index] = 1

    def _end_role(self, index: int) -> None:
        header = self._header
        if header is None:
            return
        header[index] = 0

    def _leave_consumer(self) -> None:
        self._consumer_depth -= 1
        if self._consumer_depth <= 0:
            self._consumer_depth = 0
            self._end_role(_HEADER_CONSUMER_BUSY)

    def _require_usable(self) -> None:
        if self._closed:
            raise RuntimeError(f"shared tensor channel {self._transport_id!r} is closed")
        header = self._header
        if header is not None and int(header[_HEADER_ABORTED]) != 0:
            reason = self._abort_reason or "the peer aborted the transport"
            raise SharedTensorChannelAbortedError(
                f"shared tensor channel {self._transport_id!r} is unusable: {reason}"
            )

    def _acquire_token(self, semaphore: Any, timeout: float, waiting_for: str) -> None:
        """Acquire a bounded token in short slices so abort and timeout stay observable."""

        deadline = time.monotonic() + timeout
        while True:
            self._require_usable()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                reason = f"{waiting_for} exceeded {timeout:g}s"
                self._invalidate(reason)
                raise SharedTensorChannelTimeoutError(
                    f"shared tensor channel {self._transport_id!r} {reason}; "
                    "the transport is permanently invalid"
                )
            if semaphore.acquire(timeout=min(_WAIT_SLICE_SECONDS, remaining)):
                self._require_usable()
                return

    def _invalidate(self, reason: str) -> None:
        """Mark the transport permanently unusable; blocked peers see it within one slice."""

        if self._abort_reason is None:
            self._abort_reason = reason
        header = self._header
        if header is None:
            return
        header[_HEADER_ABORTED] = 1

    def __repr__(self) -> str:
        mode = "creator" if self._creator else "attached"
        return (
            f"SharedTensorChannel(transport_id={self._transport_id!r}, {mode}, "
            f"slots={self._plan.slots}, slot_capacity={self._plan.capacity}, "
            f"fields={list(self._plan.layout)})"
        )
