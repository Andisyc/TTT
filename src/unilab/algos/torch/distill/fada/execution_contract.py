"""Pure execution limits, byte budgets, chunk descriptors and phase boundaries.

Every helper here is executable validation/planning over caller-supplied metadata. Nothing
allocates a tensor, starts a thread, owns an RNG or moves bytes: the transport (6.2) and the
learning pipeline (6.3) both consume these checks instead of guessing each other's contract.

The layouts passed in are the single authority for shape and dtype; this module keeps no
FADA field table of its own. A layout is ``Mapping[str, tuple[tuple[int, ...], torch.dtype]]``
whose shapes are full shapes, so a raw chunk field carries its leading row dimension.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from unilab.algos.torch.distill.fada.collection_contract import OfficialFADACollectRequest

Layout = Mapping[str, tuple[tuple[int, ...], torch.dtype]]

__all__ = [
    "FADAChunkDescriptor",
    "FADAExecutionLimits",
    "Layout",
    "layout_nbytes",
    "raw_chunk_rows",
    "validate_phase_boundary",
    "validate_prefetch_budget",
]


def _require_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value!r}")
    return int(value)


def _iter_layout_fields(
    layout: Layout,
) -> Iterator[tuple[str, tuple[int, ...], torch.dtype]]:
    if not isinstance(layout, Mapping):
        raise ValueError("layout must be a mapping of field name to (shape, dtype)")
    if not layout:
        raise ValueError("layout must declare at least one field")
    for name, entry in layout.items():
        if not isinstance(name, str) or not name:
            raise ValueError("layout field names must be nonempty strings")
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"layout field {name!r} must be a (shape, dtype) pair")
        shape, dtype = entry
        if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
            raise ValueError(f"layout field {name!r} shape must be a sequence of dimensions")
        dims = tuple(shape)
        if not dims:
            raise ValueError(f"layout field {name!r} shape must have at least one dimension")
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"layout field {name!r} dimensions must be positive integers")
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"layout field {name!r} dtype must be a torch.dtype")
        yield name, dims, dtype


def _field_nbytes(dims: tuple[int, ...], dtype: torch.dtype) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count * dtype.itemsize


def layout_nbytes(layout: Layout) -> int:
    """Payload bytes of one layout, derived from dtype/shape metadata only."""

    total = 0
    for _name, dims, dtype in _iter_layout_fields(layout):
        total += _field_nbytes(dims, dtype)
    return total


@dataclass(frozen=True)
class FADAExecutionLimits:
    """Fixed transport/prefetch byte budgets and slot counts for one execution stage."""

    transport_budget_bytes: int
    prefetch_budget_bytes: int
    slots: int = 2
    prefetch_batches: int = 1

    def __post_init__(self) -> None:
        for name in (
            "transport_budget_bytes",
            "prefetch_budget_bytes",
            "slots",
            "prefetch_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"FADA execution limits {name} must be a positive integer, got {value!r}"
                )


def raw_chunk_rows(
    row_layout: Layout,
    limits: FADAExecutionLimits,
    *,
    max_rows: int | None = None,
) -> int:
    """Largest chunk row count whose raw payload fits the transport budget across slots.

    ``row_layout`` describes the raw chunk layout for a single row, so every supplied field
    carries a leading dimension of 1. The budget covers array payload buffers of all slots
    together, not the whole process/native/model RAM. A layout that cannot hold one row per
    slot is rejected rather than silently planned as an empty chunk.
    """

    if not isinstance(limits, FADAExecutionLimits):
        raise ValueError("raw_chunk_rows requires FADAExecutionLimits")
    row_nbytes = 0
    for name, dims, dtype in _iter_layout_fields(row_layout):
        if dims[0] != 1:
            raise ValueError(
                f"raw chunk layout field {name!r} must have a leading row dimension of 1, "
                f"got {dims[0]}"
            )
        row_nbytes += _field_nbytes(dims, dtype)
    per_slot_row_bytes = row_nbytes * limits.slots
    if per_slot_row_bytes > limits.transport_budget_bytes:
        raise ValueError(
            "transport budget cannot hold one raw chunk row per slot: "
            f"budget={limits.transport_budget_bytes} slots={limits.slots} "
            f"row_bytes={row_nbytes}"
        )
    rows = limits.transport_budget_bytes // per_slot_row_bytes
    if max_rows is not None:
        rows = min(rows, _require_int(max_rows, "raw chunk max_rows", minimum=1))
    return rows


def validate_prefetch_budget(
    batch_layouts: Sequence[Layout],
    limits: FADAExecutionLimits,
) -> int:
    """Check retained prefetch staging against ``limits.prefetch_budget_bytes``.

    The caller passes every staging layout it retains for one stage step, including each host
    or device copy it keeps alive; those are summed per step and multiplied by
    ``limits.prefetch_batches``. This is a bounded buffer budget, not a whole-process memory
    control, and it says nothing about allocator fragmentation or model/simulator residency.
    """

    if not isinstance(limits, FADAExecutionLimits):
        raise ValueError("validate_prefetch_budget requires FADAExecutionLimits")
    if isinstance(batch_layouts, (str, bytes, Mapping)) or not isinstance(
        batch_layouts, Sequence
    ):
        raise ValueError("prefetch batch layouts must be a sequence of layouts")
    if not batch_layouts:
        raise ValueError("prefetch budget requires at least one staged batch layout")
    per_step_bytes = 0
    for index, layout in enumerate(batch_layouts):
        if not isinstance(layout, Mapping):
            raise ValueError(f"prefetch batch layout {index} must be a mapping")
        per_step_bytes += layout_nbytes(layout)
    total_bytes = per_step_bytes * limits.prefetch_batches
    if total_bytes > limits.prefetch_budget_bytes:
        raise ValueError(
            "prefetch staging exceeds the prefetch budget: "
            f"budget={limits.prefetch_budget_bytes} per_step={per_step_bytes} "
            f"batches={limits.prefetch_batches} total={total_bytes}"
        )
    return total_bytes


@dataclass(frozen=True)
class FADAChunkDescriptor:
    """Identity and cursor of one transported raw chunk.

    Validation is pure comparison against the collect request and the caller's authoritative
    receiver state; it does not take a slot lock, prove exclusive ownership, or prevent a later
    chunk from overwriting the same slot.
    """

    transport_id: str
    request_id: str
    source: str
    weight_version: int
    chunk_index: int
    row_offset: int
    valid_rows: int
    slot: int
    slot_generation: int
    is_last: bool

    def validate(
        self,
        request: OfficialFADACollectRequest,
        *,
        transport_id: str,
        total_rows: int,
        slot_capacity: int,
        slots: int,
        expected_chunk_index: int,
        expected_row_offset: int,
        expected_slot: int,
        expected_slot_generation: int,
    ) -> FADAChunkDescriptor:
        """Match this chunk against its request and the receiver's expected cursors.

        ``expected_*`` values come from the caller's authoritative receiver state; they are not
        recomputed from ``chunk_index``, because a slot generation can span requests. A final
        chunk may be partial (``valid_rows < slot_capacity``) and must be the only chunk whose
        rows end at ``total_rows``.
        """

        if not isinstance(request, OfficialFADACollectRequest):
            raise ValueError("chunk descriptor requires an OfficialFADACollectRequest")
        for label, value in (
            ("expected transport_id", transport_id),
            ("chunk transport_id", self.transport_id),
            ("chunk request_id", self.request_id),
            ("chunk source", self.source),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        weight_version = _require_int(self.weight_version, "chunk weight_version", minimum=0)
        chunk_index = _require_int(self.chunk_index, "chunk chunk_index", minimum=0)
        row_offset = _require_int(self.row_offset, "chunk row_offset", minimum=0)
        valid_rows = _require_int(self.valid_rows, "chunk valid_rows", minimum=1)
        slot = _require_int(self.slot, "chunk slot", minimum=0)
        generation = _require_int(self.slot_generation, "chunk slot_generation", minimum=0)
        total = _require_int(total_rows, "chunk total_rows", minimum=1)
        capacity = _require_int(slot_capacity, "chunk slot_capacity", minimum=1)
        slot_count = _require_int(slots, "chunk slots", minimum=1)
        expected_index = _require_int(expected_chunk_index, "expected chunk_index", minimum=0)
        expected_offset = _require_int(expected_row_offset, "expected row_offset", minimum=0)
        expected_owner = _require_int(expected_slot, "expected slot", minimum=0)
        expected_generation = _require_int(
            expected_slot_generation, "expected slot_generation", minimum=0
        )
        if not isinstance(self.is_last, bool):
            raise ValueError(f"chunk is_last must be a bool, got {self.is_last!r}")
        if expected_owner >= slot_count:
            raise ValueError(f"expected slot {expected_owner} is outside {slot_count} slots")
        if self.transport_id != transport_id:
            raise ValueError("chunk transport id does not match the expected transport")
        if self.request_id != request.request_id:
            raise ValueError("chunk request id does not match the collect request")
        if self.source != request.source:
            raise ValueError("chunk source does not match the collect request source")
        if weight_version != request.expected_weight_version:
            raise ValueError(
                "chunk weight version does not match the collect request: "
                f"chunk={weight_version} request={request.expected_weight_version}"
            )
        if slot >= slot_count:
            raise ValueError(f"chunk slot {slot} is outside {slot_count} slots")
        if slot != expected_owner:
            raise ValueError(
                f"chunk slot {slot} does not match the expected slot {expected_owner}"
            )
        if generation != expected_generation:
            raise ValueError(
                "chunk slot generation does not match the receiver state: "
                f"chunk={generation} expected={expected_generation}"
            )
        if chunk_index != expected_index:
            raise ValueError(
                f"chunk index {chunk_index} does not match expected {expected_index}"
            )
        if row_offset != expected_offset:
            raise ValueError(
                f"chunk row offset {row_offset} does not match expected {expected_offset}"
            )
        if valid_rows > capacity:
            raise ValueError(
                f"chunk valid rows {valid_rows} exceed slot capacity {capacity}"
            )
        end = row_offset + valid_rows
        if end > total:
            raise ValueError(
                f"chunk rows end at {end}, beyond the {total} collected rows"
            )
        if self.is_last != (end == total):
            raise ValueError(
                "chunk is_last marker contradicts the row cursor: "
                f"is_last={self.is_last} end={end} total={total}"
            )
        return self


def validate_phase_boundary(
    *,
    planned_updates: int,
    drawn_updates: int,
    consumed_updates: int,
    pending_batches: int,
) -> None:
    """Check a successful stage boundary: all planned updates drawn and consumed, nothing left.

    A phase that plans zero updates is valid. This is a commit/save precondition only; it does
    not reset counters for an aborted phase, and it neither executes the phase nor touches RNG.
    """

    planned = _require_int(planned_updates, "phase planned_updates", minimum=0)
    drawn = _require_int(drawn_updates, "phase drawn_updates", minimum=0)
    consumed = _require_int(consumed_updates, "phase consumed_updates", minimum=0)
    pending = _require_int(pending_batches, "phase pending_batches", minimum=0)
    if drawn != consumed or drawn != planned:
        raise ValueError(
            "phase boundary requires drawn == consumed == planned: "
            f"planned={planned} drawn={drawn} consumed={consumed}"
        )
    if pending != 0:
        raise ValueError(
            f"phase boundary requires zero pending batches before commit, got {pending}"
        )