"""Structural interface for bounded, lossless, typed shared-memory payloads.

Endpoints (including synchronization handles) travel in spawn startup kwargs,
not a running worker's request queue. Metadata is small opaque bytes; tensor
payloads stay in shared memory. This module owns no allocation or process.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Full per-slot array shapes, including the common leading row capacity.
# Dtypes are NumPy dtype strings; object-containing dtypes are not transportable.
TensorLayout = Mapping[str, tuple[tuple[int, ...], str]]
MAX_SHARED_METADATA_BYTES = 64 * 1024


class SharedTensorChannelAbortedError(RuntimeError):
    """The transport failed or was cancelled and cannot be reused."""


@dataclass(frozen=True)
class SharedTensorFrame:
    """Borrowed arrays, valid only inside the channel's read context.

    A consumer must copy into owned storage before leaving that context.
    Slot generations belong to channel lifetime, not an individual request.
    """

    arrays: Mapping[str, np.ndarray]
    row_count: int
    slot: int
    slot_generation: int
    metadata: bytes


class SharedTensorChannelProtocol(Protocol):
    transport_id: str
    slot_capacity: int
    slots: int
    layout: TensorLayout

    def write(
        self,
        arrays: Mapping[str, np.ndarray],
        metadata: bytes,
        *,
        timeout: float,
    ) -> None:
        """Copy a chunk, waiting for a free slot; never overwrite unread data."""
        ...

    def read(self, *, timeout: float) -> AbstractContextManager[SharedTensorFrame]:
        """Borrow a ready slot, returning it only after the context exits."""
        ...

    def abort(self) -> None:
        """Wake blocked operations and permanently invalidate this transport."""
        ...

    def close(self) -> None:
        """Close local handles after local operations/leases have ended; no unlink."""
        ...


class SharedTensorEndpointProtocol(Protocol):
    transport_id: str
    slot_capacity: int
    slots: int
    layout: TensorLayout

    def attach(self) -> SharedTensorChannelProtocol:
        """Attach in a spawned worker; the creator retains unlink ownership."""
        ...
