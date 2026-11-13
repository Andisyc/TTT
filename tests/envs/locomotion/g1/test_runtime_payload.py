"""Rigid payload composition and per-episode lifecycle, without a simulator."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.g1.walk_config import G1RuntimePayloadConfig
from unilab.envs.locomotion.g1.walk_runtime_payload import (
    G1RuntimePayload,
    TRUTH_KEY_ACTIVE,
    TRUTH_KEY_ADDED_MASS,
    TRUTH_KEY_ATTACHED,
)


class _FakeBackend:
    backend_type = "mujoco"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, ...]] = []

    def get_body_id(self, name: str) -> int:
        assert name == "torso_link"
        return 1

    def get_body_mass(self) -> np.ndarray:
        return np.array([0.0, 2.0])

    def get_body_ipos(self) -> np.ndarray:
        return np.zeros((2, 3))

    def get_body_inertia_tensor(self) -> np.ndarray:
        return np.stack((np.zeros((3, 3)), np.diag([1.0, 2.0, 2.0])))

    def set_body_mass_and_com(
        self,
        env_indices: np.ndarray,
        body_ids: np.ndarray,
        *,
        mass: np.ndarray,
        center_of_mass: np.ndarray | None = None,
        inertia_tensor: np.ndarray | None = None,
    ) -> None:
        assert center_of_mass is not None and inertia_tensor is not None
        self.calls.append(tuple(np.array(v, copy=True) for v in (
            env_indices, body_ids, mass, center_of_mass, inertia_tensor
        )))


def _owner(*, position=(0.0, 1.0, 0.0), attach_time=0.03):
    backend = _FakeBackend()
    owner = G1RuntimePayload(
        backend, 3,
        G1RuntimePayloadConfig(
            enabled=True, added_mass_kg=2.0, attachment_pos_m=list(position),
            half_size_m=[0.1, 0.2, 0.3], attach_time_s=attach_time,
        ),
        ctrl_dt=0.02,
    )
    info = {"steps": np.zeros(3, dtype=np.int64)}
    owner.init_info(info)
    return backend, owner, info


def test_rigid_payload_composition_timing_restore_and_rearm() -> None:
    backend, owner, info = _owner()
    owner.before_physics(info)
    assert backend.calls == []
    info["steps"][:] = [1, 2, 8]  # ceil(0.03/0.02) == 2, independently per row.
    owner.before_physics(info)
    rows, bodies, mass, com, inertia = backend.calls[0]
    np.testing.assert_array_equal(rows, [1, 2])
    np.testing.assert_array_equal(bodies, [1])
    np.testing.assert_allclose(mass, [[4.0], [4.0]])
    np.testing.assert_allclose(com, [[[0.0, 0.5, 0.0]]] * 2)
    # Hand oracle: two equal masses centered at y=0 and y=1 have combined
    # COM at y=.5; their parallel-axis contribution is diag(1,0,1).
    # The 2kg box contributes diag(13/150, 1/15, 1/30).
    expected = np.diag([2.0 + 13.0 / 150.0, 2.0 + 1.0 / 15.0, 3.0 + 1.0 / 30.0])
    np.testing.assert_allclose(inertia, np.broadcast_to(expected, (2, 1, 3, 3)), atol=1e-12)
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 2.0, 2.0])

    owner.before_physics(info)
    assert len(backend.calls) == 1
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, False, False])
    owner.restore_before_reset(info, np.array([0, 2]))
    rows, _, mass, com, inertia = backend.calls[1]
    np.testing.assert_array_equal(rows, [2])
    np.testing.assert_allclose(mass, [[2.0]])
    np.testing.assert_allclose(com, [[[0.0, 0.0, 0.0]]])
    np.testing.assert_allclose(inertia, [np.diag([1.0, 2.0, 2.0])[None]])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, False])
    info["steps"][:] = [0, 10, 0]
    owner.before_physics(info)
    assert len(backend.calls) == 2
    info["steps"][2] = 2
    owner.before_physics(info)
    np.testing.assert_array_equal(backend.calls[2][0], [2])
    np.testing.assert_allclose(backend.calls[2][4][0, 0], expected)


def test_payload_left_right_mirror_preserves_diagonals_and_reverses_cross_inertia() -> None:
    tensors = []
    for side in (1.0, -1.0):
        backend, owner, info = _owner(position=(1.0, side, 0.0), attach_time=0.0)
        owner.before_physics(info)
        _, _, mass, com, inertia = backend.calls[0]
        np.testing.assert_allclose(mass, [[4.0]] * 3)
        np.testing.assert_allclose(com[0, 0], [0.5, 0.5 * side, 0.0])
        # Reduced mass is 1kg: Ixy = -dx*dy, not zero and not +dx*dy.
        assert inertia[0, 0, 0, 1] == pytest.approx(-side)
        np.testing.assert_allclose(np.diag(inertia[0, 0]), [2 + 13/150, 3 + 1/15, 4 + 1/30])
        tensors.append(inertia[0, 0])
    reflection = np.diag([1.0, -1.0, 1.0])
    np.testing.assert_allclose(tensors[1], reflection @ tensors[0] @ reflection)


def test_zero_time_payload_rearms_on_first_tick_and_cold_reset_is_safe() -> None:
    backend, owner, info = _owner(attach_time=0.0)
    owner.restore_before_reset({"steps": np.zeros(3)}, np.array([0, 1, 2]))
    assert backend.calls == []
    owner.before_physics(info)
    owner.restore_before_reset(info, np.array([1]))
    owner.before_physics(info)
    assert len(backend.calls) == 3
    np.testing.assert_array_equal(backend.calls[-1][0], [1])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [True, True, True])


def test_disabled_payload_leaves_backend_and_state_unmodified() -> None:
    backend = _FakeBackend()
    owner = G1RuntimePayload(backend, 3, G1RuntimePayloadConfig(), ctrl_dt=0.02)
    info = {"steps": np.full(3, 100)}
    owner.init_info(info)
    owner.before_physics(info)
    owner.restore_before_reset(info, np.array([0, 1, 2]))
    assert backend.calls == []
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False] * 3)
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0] * 3)


@pytest.mark.parametrize("half_size", [[0.0, 0.1, 0.1], [-0.1, 0.1, 0.1], [0.1, np.inf, 0.1]])
def test_payload_rejects_invalid_box_dimensions(half_size) -> None:
    with pytest.raises(ValueError, match="half_size_m"):
        G1RuntimePayloadConfig(enabled=True, added_mass_kg=1.0, half_size_m=half_size).validate()


def test_additional_mass_preserves_cumulative_state_reset_and_inertia() -> None:
    backend = _FakeBackend()
    owner = G1RuntimePayload(
        backend, 2,
        G1RuntimePayloadConfig(
            enabled=True, added_mass_kg=6.0, attachment_pos_m=[0.0, 1.0, 0.0],
            half_size_m=[0.1, 0.2, 0.3], attach_time_s=10.0,
            additional_mass_events=[[20.0, 2.0]],
        ), ctrl_dt=0.02,
    )
    info = {"steps": np.array([499, 499])}
    owner.init_info(info)
    owner.before_physics(info)
    assert backend.calls == []
    info["steps"][:] = [500, 499]
    owner.before_physics(info)
    np.testing.assert_array_equal(backend.calls[-1][0], [0])
    np.testing.assert_allclose(backend.calls[-1][2], [[8.0]])
    info["steps"][:] = [999, 500]
    owner.before_physics(info)
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [6.0, 6.0])
    info["steps"][:] = [1000, 999]
    owner.before_physics(info)
    np.testing.assert_array_equal(backend.calls[-1][0], [0])
    np.testing.assert_allclose(backend.calls[-1][2], [[10.0]])
    np.testing.assert_allclose(backend.calls[-1][3], [[[0.0, 0.8, 0.0]]])
    # Baseline 2kg + cumulative 8kg box: reduced mass 1.6kg, never 6+8kg.
    expected = np.diag([1 + 8 * 0.13 / 3 + 1.6, 2 + 8 * 0.10 / 3, 2 + 8 * 0.05 / 3 + 1.6])
    np.testing.assert_allclose(backend.calls[-1][4][0, 0], expected)
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [True, False])
    owner.reconcile_info(info)
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [8.0, 6.0])
    count = len(backend.calls)
    owner.before_physics(info)
    assert len(backend.calls) == count
    owner.restore_before_reset(info, np.array([0]))
    np.testing.assert_allclose(backend.calls[-1][2], [[2.0]])
    np.testing.assert_allclose(backend.calls[-1][4][0, 0], np.diag([1.0, 2.0, 2.0]))
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 6.0])
    info["steps"][:] = [500, 1000]
    owner.before_physics(info)
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [6.0, 8.0])


@pytest.mark.parametrize("events", [[[10.0, 2.0]], [[20.0, -2.0]], [[20.0, 2.0], [15.0, 1.0]], [[float("nan"), 2.0]]])
def test_additional_mass_rejects_invalid_schedule(events) -> None:
    with pytest.raises(ValueError, match="additional_mass_events"):
        G1RuntimePayloadConfig(enabled=True, added_mass_kg=6.0, additional_mass_events=events).validate()
