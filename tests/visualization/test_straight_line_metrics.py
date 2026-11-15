"""Straightness measures path shape, not heading alignment or dwell time."""

import numpy as np
import pytest

from unilab.visualization.straight_line_trial import measure_straight_line


QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def test_diagonal_straight_path_is_straight_despite_lateral_offset():
    path = np.array([[10.0, 20.0, 0.7], [11.0, 21.0, 0.7], [12.0, 22.0, 0.7]])
    result = measure_straight_line(path, QUAT)
    assert result["line_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["forward_distance_m"] == pytest.approx(2.0)
    assert result["max_lateral_deviation_m"] == pytest.approx(2.0)
    assert result["direction_error_deg"] == pytest.approx(45.0)


def test_bent_path_score_ignores_dwell_and_uneven_time_sampling():
    path = np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7], [1.0, 1.0, 0.7]])
    uneven = np.array([
        [0.0, 0.0, 0.7], [0.0, 0.0, 0.7], [0.1, 0.0, 0.7],
        [0.9, 0.0, 0.7], [1.0, 0.0, 0.7], [1.0, 0.0, 0.7], [1.0, 1.0, 0.7],
    ])
    original = measure_straight_line(path, QUAT)
    # Continuous unit L-shaped path has orthogonal RMS sqrt(1/24).
    assert original["line_rmse_m"] == pytest.approx(np.sqrt(1 / 24), abs=0.001)
    assert measure_straight_line(uneven, QUAT)["line_rmse_m"] == pytest.approx(original["line_rmse_m"], abs=1e-12)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = path.copy()
    rotated[:, :2] = path[:, :2] @ rotation.T + [8, -3]
    assert measure_straight_line(rotated, QUAT)["line_rmse_m"] == pytest.approx(original["line_rmse_m"], abs=1e-12)


def test_stationary_path_does_not_get_a_perfect_straightness_score():
    result = measure_straight_line(np.tile([4.0, -3.0, 0.7], (10, 1)), QUAT)
    assert result["line_rmse_m"] is None
    assert result["max_line_deviation_m"] is None
    assert result["path_length_m"] == 0.0
    assert result["forward_distance_m"] == 0.0
    assert result["direction_error_deg"] is None


@pytest.mark.parametrize("delta, expected", [([2.0, 0.0], 0.0), ([-2.0, 0.0], 180.0), ([0.0, 2.0], 90.0)])
def test_global_direction_preserves_forward_backward_and_sideways(delta, expected):
    path = np.zeros((3, 3))
    path[:, :2] = np.linspace([0.0, 0.0], delta, 3)
    assert measure_straight_line(path, QUAT)["direction_error_deg"] == pytest.approx(expected)


def test_global_direction_uses_entry_heading_and_is_rigid_transform_invariant():
    path = np.array([[0.0, 0.0, 0.7], [1.0, 1.0, 0.7], [2.0, 2.0, 0.7]])
    yaw = 0.4
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    path[:, :2] = path[:, :2] @ rotation.T + [8.0, -3.0]
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    assert measure_straight_line(path, quat)["direction_error_deg"] == pytest.approx(45.0)


def test_closed_path_has_no_net_forward_direction():
    path = np.array([[0.0, 0.0, 0.7], [2.0, 0.0, 0.7], [2.0, 1.0, 0.7], [0.0, 0.0, 0.7]])
    assert measure_straight_line(path, QUAT)["direction_error_deg"] is None
