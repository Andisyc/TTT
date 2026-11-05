"""Measurement boundaries, with FADA's pinned RMSE kernel as a test oracle."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from unilab.visualization.experiment_measurement import (
    ControlStepMeasurement,
    PhysicalMeasurement,
    velocity_metric_inputs,
)


def _state(step, *, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return PhysicalMeasurement(
        run_id="synthetic-measurement", env_index=0, episode_id=0,
        control_step=step, sim_time_s=step * 0.02,
        base_quat_wxyz=np.array(quat), linear_velocity_world=np.array(linear),
        angular_velocity_world=np.array(angular),
    )


def _step(before, after, **changes):
    measurement = ControlStepMeasurement(
        before=before, after=after, ctrl_dt=0.02,
        velocity_command_body=np.array([1.0, 0.0, 0.2]),
        policy_action=np.array([0.1, 0.2]), environment_action=np.array([0.3, 0.4]),
        policy_joint_names=("left", "right"), environment_joint_names=("right", "left"),
        terminated=False, truncated=False,
    )
    return replace(measurement, **changes)


def _upstream_rmse_pre_fall(sq_arr, trimmed_fall_step=None):
    # Apache-2.0, LeCAR-Lab/FADA@290b3a9c5c0ebe2e9851b61350617544be05e9bc,
    # compute_mocap_velocity_metrics.py:584-590. Unchanged kernel body; the
    # closure's trimmed_fall_step is an explicit argument for this offline oracle.
    end = trimmed_fall_step if trimmed_fall_step is not None else len(sq_arr)
    sub = sq_arr[:end]
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite)))


def test_aligned_inputs_feed_pinned_fada_kernel_without_silent_failure_trimming():
    start = _state(0)
    middle = _state(1, linear=(4.0, 4.0, 0.0), angular=(0.0, 0.0, 1.2))
    end = _state(2, linear=(1.0, 0.0, 0.0), angular=(0.0, 0.0, 0.2))
    steps = [_step(start, middle), _step(middle, end, terminated=True)]
    values = velocity_metric_inputs(steps, linear_frame="body_xy", sample_phase="completed")
    np.testing.assert_allclose(values.timestamps_s, [0.02, 0.04])
    np.testing.assert_array_equal(values.terminated, [False, True])
    np.testing.assert_array_equal(values.truncated, [False, False])
    # Exact upstream reductions at lines 580-583; NOT mean over XY dimensions.
    lin_sq_per_step = np.sum(np.square(values.command_linear_xy - values.actual_linear_xy), axis=1)
    ang_sq_per_step = np.square(values.command_angular_z - values.actual_angular_body_z)
    assert _upstream_rmse_pre_fall(lin_sq_per_step) == pytest.approx(np.sqrt(12.5))
    assert _upstream_rmse_pre_fall(ang_sq_per_step) == pytest.approx(np.sqrt(0.5))
    # No automatic adoption of the upstream pre-fall cutoff: both rows survived.
    query = velocity_metric_inputs(steps, linear_frame="body_xy", sample_phase="query")
    np.testing.assert_allclose(query.timestamps_s, [0.0, 0.02])
    np.testing.assert_allclose(query.actual_linear_xy, [[0, 0], [4, 4]])


@pytest.mark.parametrize("sign", [-1, 1])
def test_world_to_metric_frames_use_physical_body_not_world_z(sign):
    quat = np.array([np.sqrt(3.0) / 2.0, 0.0, 0.5, 0.0]) * sign  # pitch 60 degrees
    sample = _state(0, linear=(1.0, 0.0, 0.0), angular=(1.0, 0.0, 0.0), quat=quat)
    steps = [_step(sample, replace(sample, control_step=1, sim_time_s=0.02))]
    body = velocity_metric_inputs(steps, linear_frame="body_xy", sample_phase="query")
    yaw = velocity_metric_inputs(steps, linear_frame="yaw_frame_xy", sample_phase="query")
    np.testing.assert_allclose(body.actual_linear_xy, [[0.5, 0.0]], atol=1e-14)
    np.testing.assert_allclose(yaw.actual_linear_xy, [[1.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(body.actual_angular_body_z, [np.sqrt(3.0) / 2.0])
    np.testing.assert_allclose(yaw.actual_angular_body_z, [np.sqrt(3.0) / 2.0])


def test_yaw_rotates_world_forward_into_body_lateral():
    half = np.sqrt(0.5)
    sample = _state(0, linear=(1, 0, 0), quat=(half, 0, 0, half))
    values = velocity_metric_inputs(
        [_step(sample, replace(sample, control_step=1, sim_time_s=0.02))],
        linear_frame="body_xy", sample_phase="query",
    )
    np.testing.assert_allclose(values.actual_linear_xy, [[0.0, -1.0]], atol=1e-14)


@pytest.mark.parametrize("changes, message", [
    ({"episode_id": 1}, "same run"),
    ({"run_id": "other-run"}, "same run"),
    ({"env_index": 1}, "same run"),
    ({"control_step": 0}, "immediately following"),
    ({"sim_time_s": 0.04}, "timestamps"),
])
def test_transition_rejects_reset_or_misaligned_end_state(changes, message):
    with pytest.raises(ValueError, match=message):
        _step(_state(0), replace(_state(1), **changes))


@pytest.mark.parametrize("changes, message", [
    ({"linear_velocity_world": np.array([np.nan, 0, 0])}, "finite"),
    ({"angular_velocity_world": np.array([0, np.inf, 0])}, "finite"),
    ({"base_quat_wxyz": np.array([2, 0, 0, 0])}, "unit quaternion"),
    ({"base_quat_wxyz": np.array([1, 0, 0])}, "shape"),
    ({"control_step": -1}, "non-negative integer"),
    ({"sim_time_s": np.nan}, "sim_time"),
])
def test_invalid_truth_is_rejected_not_filtered(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(_state(0), **changes)


def test_window_rejects_cross_episode_gaps_and_continuation_after_terminal():
    first = _step(_state(0), _state(1))
    next_episode = _step(replace(_state(1), episode_id=1), replace(_state(2), episode_id=1))
    for following in (next_episode, _step(_state(2), _state(3))):
        with pytest.raises(ValueError, match="contiguous single-episode"):
            velocity_metric_inputs([first, following], linear_frame="body_xy", sample_phase="completed")
    for flag in ("terminated", "truncated"):
        terminal = replace(first, **{flag: True})
        with pytest.raises(ValueError, match="terminal transition"):
            velocity_metric_inputs([terminal, _step(_state(1), _state(2))], linear_frame="body_xy", sample_phase="completed")


def test_measurements_own_arrays_and_keep_action_orders_separate():
    quaternion = np.array([1.0, 0, 0, 0])
    sample = replace(_state(0), base_quat_wxyz=quaternion)
    command = np.array([0.4, 0, 0])
    actions = np.array([0.1, 0.2])
    step = _step(sample, _state(1), velocity_command_body=command, policy_action=actions)
    quaternion[:] = 0
    command[:] = 99
    actions[:] = 99
    np.testing.assert_array_equal(sample.base_quat_wxyz, [1, 0, 0, 0])
    np.testing.assert_allclose(step.velocity_command_body, [0.4, 0, 0])
    np.testing.assert_allclose(step.policy_action, [0.1, 0.2])
    assert step.policy_joint_names == ("left", "right")
    assert step.environment_joint_names == ("right", "left")
    with pytest.raises(ValueError, match="read-only"):
        sample.linear_velocity_world[0] = 10
    values = velocity_metric_inputs([step], linear_frame="body_xy", sample_phase="completed")
    with pytest.raises(ValueError, match="read-only"):
        values.actual_linear_xy[0, 0] = 10


@pytest.mark.parametrize("kwargs, message", [
    ({"linear_frame": "world_xy", "sample_phase": "query"}, "linear_frame"),
    ({"linear_frame": "body_xy", "sample_phase": "after_reset"}, "sample_phase"),
])
def test_metric_convention_must_be_explicit_and_supported(kwargs, message):
    with pytest.raises(ValueError, match=message):
        velocity_metric_inputs([_step(_state(0), _state(1))], **kwargs)
