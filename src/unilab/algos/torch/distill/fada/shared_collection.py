"""Bounded, lossless shared-memory transport adapter for official FADA raw collection.

Producer side: split one validated raw-transition batch into row chunks that fit the injected
channel's slot capacity and copy each chunk with its frozen JSON cursor/identity metadata. No
trajectory file, whole-file hash, or read-back sits on this path.

Consumer side: drain the channel into caller-owned storage *while* the blocking
``runtime.collect`` RPC is in flight (the pump starts before the RPC so neither side can wait on
the other), match the reassembled raw batch against that RPC's result, and return both.

Ownership boundaries: the core channel owns slots, generations, wake-ups and unlink; this module
owns only the FADA-side chunking, cursor bookkeeping and bounded staging. It therefore injects
the structural protocol objects and never imports a concrete core implementation. Chunk layouts
come from ``collection_contract`` and cursor/limit checks from ``execution_contract``, so no
second field table exists here. A consumer must copy out of a borrowed frame before that frame's
read context exits; this module always copies into its own arrays and releases afterwards.

Failures (malformed descriptor, producer/receiver error, peer exit, timeout) abort the transport
permanently and raise: a partially transferred request never returns a partial batch, and a
failed channel is never silently reset or retried as a disk fallback.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADACollectionSpec,
    OfficialFADACollectRequest,
    OfficialFADARawTransitionBatch,
    official_raw_transition_layout,
    validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.execution_contract import (
    FADAChunkDescriptor,
    FADAExecutionLimits,
    raw_chunk_rows,
)
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectResult,
    validate_dagger_collect_result,
)
from unilab.ipc.shared_tensor_protocol import (
    MAX_SHARED_METADATA_BYTES,
    SharedTensorChannelProtocol,
    TensorLayout,
)

__all__ = [
    "SHARED_PAYLOAD_KIND",
    "abort_shared_transport",
    "collect_shared_raw",
    "raw_transport_layout",
    "send_shared_collection",
    "validate_shared_request",
]

# Response metadata marker: the raw batch arrived over the shared channel, not a file.
SHARED_PAYLOAD_KIND = "shared_memory"

# Exact frozen chunk-metadata key set; the frame itself carries rows/slot/generation.
_CHUNK_METADATA_KEYS = (
    "chunk_index",
    "is_last",
    "request_id",
    "row_offset",
    "source",
    "total_rows",
    "transport_id",
    "weight_version",
)

# Only these dtypes travel; the core transport never sees an object dtype.
_NUMPY_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}

# Identity keys a caller must supply (prepare metadata) and a shared result must repeat.
_REQUIRED_IDENTITY_KEYS = (
    "expert_checkpoint",
    "expert_checkpoint_sha256",
    "experiment_config_sha256",
    "preprocessing",
    "upstream_revision",
)

# Bounded wait for a receiver thread that abort() has already woken.
_JOIN_GRACE_SECONDS = 30.0


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return int(value)


def _core_int(value: Any, label: str) -> int:
    """Read an integer supplied by the core transport (frame rows/slot/generation)."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} from the shared transport must be an integer, got {value!r}")
    return int(value)


def _positive_seconds(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number of seconds, got {value!r}")
    return float(value)


def _resolved(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"official FADA path identity must be a nonempty string, got {path!r}")
    return str(Path(path).expanduser().resolve())


def _sha256_identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"official FADA {label} must be a lowercase SHA256 identity")
    return value


def _numpy_dtype(dtype: Any) -> np.dtype[Any]:
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:  # pragma: no cover - only for exotic caller layouts
        raise ValueError(f"shared transport dtype {dtype!r} is not a NumPy dtype") from exc
    if resolved.hasobject:
        raise ValueError(f"shared transport dtype {dtype!r} contains Python objects")
    return resolved


def _chunk_layout(spec: OfficialFADACollectionSpec, rows: int) -> TensorLayout:
    """Canonical transport layout for ``rows`` raw rows; shapes only, no allocation."""

    row_count = _positive_int(rows, "shared chunk row count")
    layout = official_raw_transition_layout(spec, row_count)
    resolved: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, (shape, dtype) in layout.items():
        try:
            name_dtype = _NUMPY_DTYPE_NAMES[dtype]
        except KeyError:
            raise ValueError(
                f"official raw field {name!r} has no lossless NumPy transport dtype: {dtype}"
            ) from None
        resolved[name] = (tuple(shape), name_dtype)
    return resolved


def _require_layout_matches(actual: Any, expected: TensorLayout, *, label: str) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{label} must be a mapping of field name to (shape, dtype)")
    if set(actual) != set(expected):
        raise ValueError(
            f"{label} fields mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for name, (shape, dtype_name) in expected.items():
        entry = actual[name]
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"{label} field {name!r} must be a (shape, dtype) pair")
        actual_shape, actual_dtype = entry
        if isinstance(actual_shape, (str, bytes)) or not isinstance(
            actual_shape, (tuple, list)
        ):
            raise ValueError(f"{label} field {name!r} shape must be a sequence of dimensions")
        if tuple(actual_shape) != tuple(shape):
            raise ValueError(
                f"{label} field {name!r} shape mismatch: "
                f"transport={tuple(actual_shape)} expected={tuple(shape)}"
            )
        if _numpy_dtype(actual_dtype) != _numpy_dtype(dtype_name):
            raise ValueError(
                f"{label} field {name!r} dtype mismatch: "
                f"transport={actual_dtype!r} expected={dtype_name!r}"
            )


def validate_shared_request(
    channel: SharedTensorChannelProtocol,
    request: OfficialFADACollectRequest,
    *,
    spec: OfficialFADACollectionSpec,
) -> str:
    """Reject a non-shared request or endpoint/spec drift before any native sampling.

    Returns the request's transport id so callers cannot proceed on an unchecked value.
    """

    if not isinstance(request, OfficialFADACollectRequest):
        raise TypeError(
            "shared FADA collection requires an OfficialFADACollectRequest, "
            f"got {type(request).__name__}"
        )
    if not isinstance(spec, OfficialFADACollectionSpec):
        raise TypeError(
            "shared FADA collection requires an OfficialFADACollectionSpec, "
            f"got {type(spec).__name__}"
        )
    transport_id = request.transport_id
    if not isinstance(transport_id, str) or not transport_id:
        raise ValueError(
            "a shared FADA collection request needs an explicit transport_id; a request "
            "without one belongs to the artifact path"
        )
    if request.output_path != "":
        raise ValueError(
            "a shared FADA collection request must carry an empty output_path (no file "
            f"artifact), got {request.output_path!r}"
        )
    channel_transport = getattr(channel, "transport_id", None)
    if not isinstance(channel_transport, str) or not channel_transport:
        raise ValueError("the attached shared channel does not expose a transport_id")
    if channel_transport != transport_id:
        raise ValueError(
            "shared FADA channel transport id does not match the collect request: "
            f"channel={channel_transport!r} request={transport_id!r}"
        )
    capacity = _positive_int(
        getattr(channel, "slot_capacity", None), "shared channel slot_capacity"
    )
    _positive_int(getattr(channel, "slots", None), "shared channel slot count")
    _require_layout_matches(
        getattr(channel, "layout", None),
        _chunk_layout(spec, capacity),
        label="shared channel layout",
    )
    return transport_id


def raw_transport_layout(
    spec: OfficialFADACollectionSpec,
    limits: FADAExecutionLimits,
    *,
    max_rows: int,
) -> TensorLayout:
    """Canonical chunk layout for the endpoint that ``limits`` sizes.

    The chunk row count is the existing ``raw_chunk_rows`` budget result capped by ``max_rows``,
    so the producer and the core endpoint agree on one capacity by construction. This only reads
    shape/dtype metadata: no tensor, NumPy array or shared segment is allocated here.
    """

    if not isinstance(spec, OfficialFADACollectionSpec):
        raise TypeError(
            "raw_transport_layout requires an OfficialFADACollectionSpec, "
            f"got {type(spec).__name__}"
        )
    if not isinstance(limits, FADAExecutionLimits):
        raise ValueError("raw_transport_layout requires FADAExecutionLimits")
    row_layout = official_raw_transition_layout(spec, 1)
    rows = raw_chunk_rows(
        row_layout,
        limits,
        max_rows=_positive_int(max_rows, "raw transport max_rows"),
    )
    return _chunk_layout(spec, rows)


def _encode_chunk_metadata(
    *,
    request_id: str,
    transport_id: str,
    source: str,
    weight_version: int,
    chunk_index: int,
    row_offset: int,
    total_rows: int,
    is_last: bool,
) -> bytes:
    payload = {
        "chunk_index": int(chunk_index),
        "is_last": bool(is_last),
        "request_id": request_id,
        "row_offset": int(row_offset),
        "source": source,
        "total_rows": int(total_rows),
        "transport_id": transport_id,
        "weight_version": int(weight_version),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_SHARED_METADATA_BYTES:
        raise ValueError(
            "shared FADA chunk metadata exceeds the transport limit: "
            f"{len(encoded)} > {MAX_SHARED_METADATA_BYTES} bytes"
        )
    return encoded


def _decode_chunk_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, (bytes, bytearray, memoryview)):
        raise ValueError(
            "shared FADA chunk metadata must be bytes, "
            f"got {type(metadata).__name__}"
        )
    raw = bytes(metadata)
    if len(raw) > MAX_SHARED_METADATA_BYTES:
        raise ValueError(
            "shared FADA chunk metadata exceeds the transport limit: "
            f"{len(raw)} > {MAX_SHARED_METADATA_BYTES} bytes"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"shared FADA chunk metadata is not frozen JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("shared FADA chunk metadata must be a JSON object")
    if set(payload) != set(_CHUNK_METADATA_KEYS):
        raise ValueError(
            "shared FADA chunk metadata keys mismatch: "
            f"missing={sorted(set(_CHUNK_METADATA_KEYS) - set(payload))}, "
            f"extra={sorted(set(payload) - set(_CHUNK_METADATA_KEYS))}"
        )
    return payload


def _chunk_arrays(
    batch: OfficialFADARawTransitionBatch,
    *,
    start: int,
    stop: int,
    layout: TensorLayout,
    total_rows: int,
) -> dict[str, np.ndarray]:
    """NumPy row views of one chunk; the transport copies them before its write returns."""

    rows = stop - start
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in layout.items():
        value = getattr(batch, name, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"official raw batch field {name!r} is not a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"official raw batch field {name!r} must be CPU-resident")
        if tuple(value.shape) != (total_rows, *tuple(shape[1:])):
            raise ValueError(
                f"official raw batch field {name!r} shape mismatch: "
                f"batch={tuple(value.shape)} expected={(total_rows, *tuple(shape[1:]))}"
            )
        chunk = value[start:stop]
        if tuple(chunk.shape) != (rows, *tuple(shape[1:])):
            raise ValueError(
                f"official raw chunk field {name!r} shape mismatch: "
                f"chunk={tuple(chunk.shape)} expected={(rows, *tuple(shape[1:]))}"
            )
        values = chunk.detach().numpy()
        if values.dtype != _numpy_dtype(dtype_name):
            raise ValueError(
                f"official raw chunk field {name!r} dtype mismatch: "
                f"batch={values.dtype} transport={dtype_name}"
            )
        arrays[name] = values
    return arrays


def send_shared_collection(
    channel: SharedTensorChannelProtocol,
    request: OfficialFADACollectRequest,
    batch: OfficialFADARawTransitionBatch,
    *,
    spec: OfficialFADACollectionSpec,
    timeout: float,
) -> None:
    """Stream one raw batch over the shared channel as capacity-sized chunks.

    ``timeout`` bounds the whole send, not each chunk: a slow reader can never extend the
    transfer past that monotonic deadline. Any failure (layout drift, timeout, transport abort)
    aborts the channel permanently and re-raises, so the consumer never observes a truncated
    batch as success. Nothing is written to disk on this path.
    """

    transport_id = validate_shared_request(channel, request, spec=spec)
    # Past the request/endpoint check this transport is committed to the request, so any
    # failure leaves its slot state uncertain: abort permanently instead of retrying.
    try:
        if not isinstance(batch, OfficialFADARawTransitionBatch):
            raise TypeError(
                "shared FADA send requires an OfficialFADARawTransitionBatch, "
                f"got {type(batch).__name__}"
            )
        transfer_timeout = _positive_seconds(timeout, "shared FADA send timeout")
        capacity = _positive_int(
            getattr(channel, "slot_capacity", None), "shared channel slot_capacity"
        )
        total_rows = int(batch.num_samples)
        if total_rows <= 0:
            raise ValueError(
                "shared FADA send requires at least one raw transition, "
                f"got {total_rows}"
            )
        layout = _chunk_layout(spec, capacity)
        deadline = time.monotonic() + transfer_timeout
        offset = 0
        chunk_index = 0
        while offset < total_rows:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "shared FADA send exceeded its transfer deadline after "
                    f"{chunk_index} of {-(-total_rows // capacity)} chunks "
                    f"({offset}/{total_rows} rows)"
                )
            rows = min(capacity, total_rows - offset)
            is_last = offset + rows == total_rows
            arrays = _chunk_arrays(
                batch,
                start=offset,
                stop=offset + rows,
                layout=layout,
                total_rows=total_rows,
            )
            channel.write(
                arrays,
                _encode_chunk_metadata(
                    request_id=request.request_id,
                    transport_id=transport_id,
                    source=request.source,
                    weight_version=int(request.expected_weight_version),
                    chunk_index=chunk_index,
                    row_offset=offset,
                    total_rows=total_rows,
                    is_last=is_last,
                ),
                timeout=remaining,
            )
            offset += rows
            chunk_index += 1
    except BaseException:
        abort_shared_transport(channel)
        raise


def abort_shared_transport(channel: Any) -> None:
    """Invalidate a shared transport without masking the failure that triggered the abort.

    Used both by this module's own failure paths and by a producer that failed before/around a
    send, so a request that cannot complete never leaves a peer blocked on a live transport.
    """

    abort = getattr(channel, "abort", None)
    if abort is None:
        return
    try:
        abort()
    except BaseException:  # noqa: BLE001 - the primary failure stays primary
        pass


class _SharedRawReceiver:
    """Drain shared chunks into owned, bounded storage while the collect RPC is blocked.

    Slack for one chunk is reclaimed as soon as its frame leaves the read context, so the
    borrowing lease never outlives a copy. Storage is allocated once, from the first chunk's
    ``total_rows`` cursor, and checked against ``max_rows`` and the pending byte budget before
    a single row is retained.
    """

    def __init__(
        self,
        channel: SharedTensorChannelProtocol,
        *,
        request: OfficialFADACollectRequest,
        spec: OfficialFADACollectionSpec,
        transport_id: str,
        max_rows: int,
        pending_budget_bytes: int,
        deadline: float,
    ) -> None:
        self._channel = channel
        self._request = request
        self._spec = spec
        self._transport_id = transport_id
        self._max_rows = max_rows
        self._budget_bytes = pending_budget_bytes
        self._deadline = deadline
        self._capacity = int(channel.slot_capacity)
        self._slots = int(channel.slots)
        self._layout = _chunk_layout(spec, self._capacity)
        self._buffers: dict[str, np.ndarray] | None = None
        self._total_rows: int | None = None
        self._last_position: int | None = None
        self._next_index = 0
        self._next_offset = 0
        self._complete = False
        self._error: BaseException | None = None
        self._joined = False
        self._thread = threading.Thread(
            target=self._run,
            name="fada-shared-receiver",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wait(self) -> None:
        """Join the pump at the transfer deadline and surface its failure, if any."""

        self._thread.join(timeout=max(self._deadline - time.monotonic(), 0.0))
        if self._thread.is_alive():
            abort_shared_transport(self._channel)
            self._thread.join(timeout=_JOIN_GRACE_SECONDS)
            self._joined = not self._thread.is_alive()
            raise TimeoutError(
                "shared FADA transfer did not complete within its deadline after "
                f"{self._next_index} chunks ({self._next_offset} rows)"
            )
        self._joined = True
        if self._error is not None:
            raise self._error
        if not self._complete:
            raise RuntimeError(
                "shared FADA receiver stopped without a final chunk: "
                f"{self._next_offset} of {self._total_rows} rows"
            )

    def abort_and_join(self) -> None:
        """Wake a blocked read, then join; never replaces the caller's primary failure."""

        abort_shared_transport(self._channel)
        if self._joined:
            return
        self._thread.join(timeout=_JOIN_GRACE_SECONDS)
        self._joined = not self._thread.is_alive()

    def finish(self) -> OfficialFADARawTransitionBatch:
        if self._buffers is None or self._total_rows is None:
            raise RuntimeError("shared FADA receiver holds no raw transitions")
        mapping = {
            name: torch.from_numpy(buffer) for name, buffer in self._buffers.items()
        }
        self._buffers = None
        batch = OfficialFADARawTransitionBatch.from_mapping(mapping)
        if batch.num_samples != self._total_rows:
            raise ValueError(
                "shared FADA reassembled row count mismatch: "
                f"batch={batch.num_samples} expected={self._total_rows}"
            )
        return batch.validate(self._spec)

    def _run(self) -> None:
        try:
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "shared FADA receiver exceeded its transfer deadline after "
                        f"{self._next_index} chunks ({self._next_offset} rows)"
                    )
                with self._channel.read(timeout=remaining) as frame:
                    is_last = self._admit(frame)
                if is_last:
                    self._complete = True
                    return
        except BaseException as exc:  # noqa: BLE001 - surfaced by wait()
            self._error = exc
            abort_shared_transport(self._channel)

    def _admit(self, frame: Any) -> bool:
        payload = _decode_chunk_metadata(getattr(frame, "metadata", None))
        row_count = _core_int(getattr(frame, "row_count", None), "shared frame row_count")
        slot = _core_int(getattr(frame, "slot", None), "shared frame slot")
        generation = _core_int(
            getattr(frame, "slot_generation", None), "shared frame slot_generation"
        )
        total_rows = payload["total_rows"]
        descriptor = FADAChunkDescriptor(
            transport_id=payload["transport_id"],
            request_id=payload["request_id"],
            source=payload["source"],
            weight_version=payload["weight_version"],
            chunk_index=payload["chunk_index"],
            row_offset=payload["row_offset"],
            valid_rows=row_count,
            slot=slot,
            slot_generation=generation,
            is_last=payload["is_last"],
        ).validate(
            self._request,
            transport_id=self._transport_id,
            total_rows=total_rows,
            slot_capacity=self._capacity,
            slots=self._slots,
            expected_chunk_index=self._next_index,
            expected_row_offset=self._next_offset,
            # The core owns slot ownership and generations; the frame is authoritative and
            # this descriptor must agree with it.
            expected_slot=slot,
            expected_slot_generation=generation,
        )
        # Core owns slot ownership and generations; the frame is authoritative, so ordering is
        # checked through the channel's lifetime position (generation × slots + slot) which must
        # strictly increase for every admitted chunk. A request must not depend on the chunk
        # index to infer a cross-request generation.
        position = generation * self._slots + slot
        if self._last_position is not None and position <= self._last_position:
            raise ValueError(
                "shared FADA chunk arrived out of order: "
                f"slot={slot} generation={generation} position={position} "
                f"previous={self._last_position}"
            )
        self._last_position = position
        if self._total_rows is None:
            self._allocate(int(total_rows))
        elif int(total_rows) != self._total_rows:
            raise ValueError(
                "shared FADA chunks disagree on the total row count: "
                f"chunk={total_rows} first={self._total_rows}"
            )
        start = descriptor.row_offset
        stop = start + descriptor.valid_rows
        self._copy(getattr(frame, "arrays", None), start=start, stop=stop)
        self._next_index += 1
        self._next_offset = stop
        return descriptor.is_last

    def _allocate(self, total_rows: int) -> None:
        if total_rows > self._max_rows:
            raise ValueError(
                "shared FADA transfer exceeds the retained row limit: "
                f"total_rows={total_rows} max_rows={self._max_rows}"
            )
        layout = _chunk_layout(self._spec, total_rows)
        required = 0
        for shape, dtype_name in layout.values():
            required += int(np.prod(shape, dtype=np.int64)) * _numpy_dtype(dtype_name).itemsize
        if required > self._budget_bytes:
            raise ValueError(
                "shared FADA staging exceeds the pending byte budget: "
                f"required={required} budget={self._budget_bytes} rows={total_rows}"
            )
        self._total_rows = total_rows
        self._buffers = {
            name: np.empty(shape, dtype=_numpy_dtype(dtype_name))
            for name, (shape, dtype_name) in layout.items()
        }

    def _copy(self, arrays: Any, *, start: int, stop: int) -> None:
        if self._buffers is None:
            raise RuntimeError("shared FADA receiver copied before sizing its staging area")
        if not isinstance(arrays, Mapping):
            raise ValueError("shared FADA frame arrays must be a mapping")
        if set(arrays) != set(self._layout):
            raise ValueError(
                "shared FADA chunk array fields mismatch: "
                f"missing={sorted(set(self._layout) - set(arrays))}, "
                f"extra={sorted(set(arrays) - set(self._layout))}"
            )
        rows = stop - start
        for name, (shape, dtype_name) in self._layout.items():
            buffer = self._buffers[name]
            value = arrays[name]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"shared FADA chunk field {name!r} is not a NumPy array")
            if value.dtype != _numpy_dtype(dtype_name):
                raise ValueError(
                    f"shared FADA chunk field {name!r} dtype mismatch: "
                    f"frame={value.dtype} expected={dtype_name}"
                )
            tail = tuple(shape[1:])
            if tuple(value.shape) == (rows, *tail):
                view = value
            elif tuple(value.shape) == (self._capacity, *tail):
                view = value[:rows]
            else:
                raise ValueError(
                    f"shared FADA chunk field {name!r} has an unexpected shape: "
                    f"frame={tuple(value.shape)} rows={rows} capacity={self._capacity}"
                )
            buffer[start:stop] = view


def _required_identity(
    expected_metadata: Mapping[str, Any],
    *,
    spec: OfficialFADACollectionSpec,
) -> dict[str, Any]:
    """Validate the caller's expected upstream/preprocessing identity once, up front."""

    if not isinstance(expected_metadata, Mapping):
        raise ValueError(
            "shared FADA collection requires the expected identity mapping "
            f"(prepare metadata), got {type(expected_metadata).__name__}"
        )
    missing = [key for key in _REQUIRED_IDENTITY_KEYS if key not in expected_metadata]
    if missing:
        raise ValueError(
            f"shared FADA expected identity is incomplete: missing {missing}"
        )
    revision = expected_metadata["upstream_revision"]
    if revision != OFFICIAL_FADA_SOURCE_REVISION:
        raise ValueError(
            "shared FADA expected upstream revision does not match the pinned checkout: "
            f"expected={revision!r} pinned={OFFICIAL_FADA_SOURCE_REVISION}"
        )
    expert_checkpoint = _resolved(expected_metadata["expert_checkpoint"])
    if expert_checkpoint != _resolved(spec.expert_checkpoint):
        raise ValueError(
            "shared FADA expected expert checkpoint does not match the collection spec: "
            f"expected={expert_checkpoint} spec={_resolved(spec.expert_checkpoint)}"
        )
    return {
        "upstream_revision": revision,
        "expert_checkpoint": expert_checkpoint,
        "expert_checkpoint_sha256": _sha256_identity(
            expected_metadata["expert_checkpoint_sha256"], "expert_checkpoint_sha256"
        ),
        "experiment_config_sha256": _sha256_identity(
            expected_metadata["experiment_config_sha256"], "experiment_config_sha256"
        ),
        "preprocessing": validate_official_fada_preprocessing(
            expected_metadata["preprocessing"]
        ),
    }


def _validate_shared_result(
    result: Any,
    request: OfficialFADACollectRequest,
    *,
    batch: OfficialFADARawTransitionBatch,
    identity: Mapping[str, Any],
) -> None:
    """Bind the RPC result to the exact request and to the transported raw rows."""

    if not isinstance(result, DaggerCollectResult):
        raise TypeError(
            "shared FADA collection expects a DaggerCollectResult, "
            f"got {type(result).__name__}"
        )
    validate_dagger_collect_result(request, result)
    metadata = result.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("shared FADA collection result metadata must be a mapping")
    if metadata.get("payload_kind") != SHARED_PAYLOAD_KIND:
        raise ValueError(
            "shared FADA collection result is not a shared payload: "
            f"payload_kind={metadata.get('payload_kind')!r}"
        )
    if metadata.get("transport_id") != request.transport_id:
        raise ValueError(
            "shared FADA collection result transport id mismatch: "
            f"result={metadata.get('transport_id')!r} request={request.transport_id!r}"
        )
    if metadata.get("source") != request.source:
        raise ValueError(
            "shared FADA collection result source mismatch: "
            f"result={metadata.get('source')!r} request={request.source!r}"
        )
    if metadata.get("sample_unit") != "transition":
        raise ValueError(
            "shared FADA collection result must keep raw transition sample units, got "
            f"{metadata.get('sample_unit')!r}"
        )
    collected_steps = metadata.get("collected_steps")
    if (
        isinstance(collected_steps, bool)
        or not isinstance(collected_steps, int)
        or collected_steps <= 0
    ):
        raise ValueError(
            "shared FADA collected_steps must be a positive transition count, "
            f"got {collected_steps!r}"
        )
    weak_checkpoint = (
        str(request.weak_checkpoint) if request.weak_checkpoint is not None else None
    )
    if metadata.get("weak_checkpoint") != weak_checkpoint:
        raise ValueError(
            "shared FADA collection result weak checkpoint mismatch: "
            f"result={metadata.get('weak_checkpoint')!r} request={weak_checkpoint!r}"
        )
    # The producer copies these request fields into its metadata (the same binding the artifact
    # path uses); require them so a result can never be matched to a different request body.
    # A caller that additionally needs prepare-only fields can compare them itself.
    for name in (
        "request_id",
        "scenario",
        "iteration",
        "checkpoint_path",
        "expected_weight_version",
    ):
        if name not in metadata:
            raise ValueError(
                f"shared FADA collection result metadata is missing {name!r}"
            )
        if metadata[name] != getattr(request, name):
            raise ValueError(
                f"shared FADA collection result {name} does not match its request: "
                f"result={metadata[name]!r} request={getattr(request, name)!r}"
            )
    if metadata.get("observed_weight_version") != int(result.observed_weight_version):
        raise ValueError(
            "shared FADA collection result observed weight version mismatch: "
            f"metadata={metadata.get('observed_weight_version')!r} "
            f"result={result.observed_weight_version}"
        )
    if metadata.get("producer_pid") != int(result.worker_pid):
        raise ValueError(
            "shared FADA collection result producer pid mismatch: "
            f"result={metadata.get('producer_pid')!r} worker={result.worker_pid}"
        )
    for key in _REQUIRED_IDENTITY_KEYS:
        if key not in metadata:
            raise ValueError(f"shared FADA collection result is missing identity {key!r}")
    for key in _REQUIRED_IDENTITY_KEYS:
        actual = metadata[key]
        expected = identity[key]
        if key == "preprocessing":
            if validate_official_fada_preprocessing(actual) != expected:
                raise ValueError(
                    "shared FADA collection result preprocessing identity mismatch"
                )
        elif key == "expert_checkpoint":
            if _resolved(actual) != expected:
                raise ValueError(
                    "shared FADA collection result expert checkpoint mismatch: "
                    f"result={_resolved(actual)} expected={expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"shared FADA collection result {key} identity mismatch: "
                f"result={actual!r} expected={expected!r}"
            )
    if int(result.num_samples) != int(batch.num_samples):
        raise ValueError(
            "shared FADA result/transfer row mismatch: "
            f"result={int(result.num_samples)} transfer={int(batch.num_samples)}"
        )


def collect_shared_raw(
    runtime: Any,
    request: OfficialFADACollectRequest,
    channel: SharedTensorChannelProtocol,
    *,
    spec: OfficialFADACollectionSpec,
    max_rows: int,
    pending_budget_bytes: int,
    expected_metadata: Mapping[str, Any],
    timeout: float,
) -> tuple[OfficialFADARawTransitionBatch, DaggerCollectResult]:
    """Collect one shared raw batch and match it against the collector's own result.

    The receive pump starts *before* the blocking ``runtime.collect`` call: the collector can
    only finish after every chunk has been written, so a receiver that started later could
    deadlock against a full slot ring. ``timeout`` bounds the whole transfer (one monotonic
    deadline anchored at pump start, covering the collector's native sampling plus the drain),
    while the RPC keeps its existing runtime request timeout; it is therefore the caller's job
    to pass a transfer timeout that exceeds the expected collect wall time. The collector's
    native sampling is not claimed to be interruptible: a failed transfer aborts the channel
    and waits for the peer.
    """

    transport_id = validate_shared_request(channel, request, spec=spec)
    rows_limit = _positive_int(max_rows, "shared FADA max_rows")
    budget_bytes = _positive_int(
        pending_budget_bytes, "shared FADA pending_budget_bytes"
    )
    transfer_timeout = _positive_seconds(timeout, "shared FADA collection timeout")
    identity = _required_identity(expected_metadata, spec=spec)
    receiver = _SharedRawReceiver(
        channel,
        request=request,
        spec=spec,
        transport_id=transport_id,
        max_rows=rows_limit,
        pending_budget_bytes=budget_bytes,
        deadline=time.monotonic() + transfer_timeout,
    )
    receiver.start()
    try:
        result = runtime.collect(request)
        receiver.wait()
        batch = receiver.finish()
        _validate_shared_result(result, request, batch=batch, identity=identity)
    except BaseException:
        receiver.abort_and_join()
        raise
    # wait() already joined the pump; the channel stays live for the next request, because a
    # successful transfer must not invalidate a transport whose slots and generations persist.
    return batch, result