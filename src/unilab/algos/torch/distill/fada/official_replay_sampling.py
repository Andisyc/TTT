"""Batched window gathering over the pinned official ReplayBuffer's index cache."""

from __future__ import annotations

from typing import Any

import numpy as np


class OfficialFADAChunkSampler:
    """Preserve upstream anchor selection/RNG while gathering rows in NumPy.

    The upstream buffer still owns episode grouping, valid-anchor filtering,
    eviction and cache invalidation. Only its per-sample window-copy loop is
    replaced. Index storage is linear in replay size, rather than N * H/K.
    """

    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer
        self._chrono_idx: np.ndarray | None = None
        self._sequence_offsets = np.zeros(1, dtype=np.int64)
        self._sequence_raw = np.zeros(0, dtype=np.int64)
        self._anchors = np.zeros(0, dtype=np.int64)

    def _refresh_indices(self) -> None:
        buffer = self.buffer
        # Upstream replaces this array whenever its sampling cache is rebuilt
        # (including add, clear, load and episode eviction). Retaining the old
        # array makes identity comparison safe against object-id reuse.
        if self._chrono_idx is buffer._cached_chrono_idx:
            return
        sequences = buffer._cached_seq_positions
        lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
        self._sequence_offsets = np.concatenate((np.zeros(1, dtype=np.int64), lengths.cumsum()))
        self._sequence_raw = buffer._cached_chrono_idx[np.concatenate(sequences)]
        self._anchors = np.concatenate(buffer._cached_seq_anchor_positions)
        self._chrono_idx = buffer._cached_chrono_idx

    def sample_chunk(self, batch_size: int) -> dict[str, np.ndarray]:
        buffer = self.buffer
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(buffer) <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        buffer._refresh_sampling_cache()
        total_valid = int(buffer._cached_num_valid_chunks)
        if total_valid <= 0 or buffer._cached_chrono_idx is None:
            raise RuntimeError(
                "No valid chunks available in replay buffer. "
                "Collect more transitions or reduce pred_horizon."
            )
        self._refresh_indices()

        # Same call, population order and replacement rule as upstream. No other
        # random draws occur here, so Planner/IDM source draw order is unchanged.
        sampled = np.random.choice(
            total_valid, size=batch_size, replace=total_valid < batch_size
        ).astype(np.int64, copy=False)
        seq_ids = np.searchsorted(buffer._cached_seq_usable_cumsum, sampled, side="right")
        offsets = self._sequence_offsets[seq_ids]
        lengths = self._sequence_offsets[seq_ids + 1] - offsets
        anchors = self._anchors[sampled]
        anchor_raw = self._sequence_raw[offsets + anchors]

        history_positions = anchors[:, None] + np.arange(1 - buffer.history_len, 1)
        history_valid = history_positions >= 0
        history_raw = self._sequence_raw[offsets[:, None] + np.maximum(history_positions, 0)]
        history_obs = buffer.obs[history_raw]
        history_obs[~history_valid] = 0.0
        previous_positions = history_positions - 1
        previous_raw = self._sequence_raw[offsets[:, None] + np.maximum(previous_positions, 0)]
        history_act = buffer.executed_act[previous_raw]
        history_act[previous_positions < 0] = 0.0

        horizon = buffer.pred_horizon
        future_offsets = np.arange(horizon)
        strict = buffer.has_expert_chunk[anchor_raw]
        target_actions = np.zeros((batch_size, horizon, buffer.act_dim), dtype=np.float32)
        target_actions[strict] = buffer.expert_chunk[anchor_raw[strict]]
        fallback = ~strict
        fallback_raw = self._sequence_raw[
            offsets[fallback, None] + anchors[fallback, None] + future_offsets
        ]
        target_actions[fallback] = buffer.expert_act[fallback_raw]

        action_valid = anchors + horizon <= lengths
        trajectory_actions = np.zeros_like(target_actions)
        trajectory_raw = self._sequence_raw[
            offsets[action_valid, None] + anchors[action_valid, None] + future_offsets
        ]
        trajectory_actions[action_valid] = buffer.executed_act[trajectory_raw]
        trajectory_valid = action_valid.copy()
        out = {
            "history_obs": history_obs,
            "history_act": history_act,
            "current_command": buffer.current_command[anchor_raw],
            "history_valid_mask": history_valid,
            "target_actions": target_actions,
            "trajectory_target_actions": trajectory_actions,
            "trajectory_target_valid": trajectory_valid,
        }
        if buffer.require_future_obs_targets:
            missing = ~buffer.has_expert_future_obs_chunk[anchor_raw]
            if np.any(missing):
                raw_index = int(anchor_raw[np.flatnonzero(missing)[0]])
                raise RuntimeError(
                    "Missing strict expert future-observation chunk for sampled anchor "
                    f"(raw_index={raw_index}). Next-observation fallback is disabled."
                )
            out["target_next_observations"] = buffer.expert_future_obs_chunk[anchor_raw]
            future_valid = anchors + 1 + horizon <= lengths
            trajectory_future = np.zeros((batch_size, horizon, buffer.obs_dim), dtype=np.float32)
            future_raw = self._sequence_raw[
                offsets[future_valid, None] + anchors[future_valid, None] + 1 + future_offsets
            ]
            trajectory_future[future_valid] = buffer.obs[future_raw]
            trajectory_valid &= future_valid
            out["trajectory_target_next_observations"] = trajectory_future
        return out
