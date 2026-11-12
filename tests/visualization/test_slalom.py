"""Independent command integration and actual-state slalom acceptance criteria."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.slalom import SlalomCommandProgram, SlalomConfig, SlalomEvaluator


def _integrate_twist(pose, command, dt):
    """Exact SE(2) exponential, independent of the program's arc fitting."""
    vx, vy, omega = command
    angle = omega * dt
    if abs(omega) < 1e-12:
        local = np.array([vx * dt, vy * dt])
    else:
        local = np.array([
            (vx * np.sin(angle) + vy * (np.cos(angle) - 1.0)) / omega,
            (vx * (1.0 - np.cos(angle)) + vy * np.sin(angle)) / omega,
        ])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pose[:2] += np.array([[c, -s], [s, c]]) @ local
    pose[2] += angle


@pytest.mark.parametrize("pole_count", [4, 5])
@pytest.mark.parametrize("ctrl_dt", [0.02, 0.03])
def test_independently_integrated_commands_follow_alternating_route(pole_count, ctrl_dt):
    config = SlalomConfig(pole_count=pole_count)
    program = SlalomCommandProgram.from_config(config, ctrl_dt=ctrl_dt, prediction_horizon=6)
    pose = np.zeros(3)
    previous = None
    crossings = []
    next_pole = 0
    for step in range(program.total_control_steps):
        command = program.command_at_control_step(step)
        if step % program.segment_steps:
            np.testing.assert_array_equal(command, previous)
        previous = command
        old = pose.copy()
        _integrate_twist(pose, command, ctrl_dt)
        if next_pole < pole_count and old[0] < program.pole_centers[next_pole, 0] <= pose[0]:
            fraction = (program.pole_centers[next_pole, 0] - old[0]) / (pose[0] - old[0])
            crossings.append(old[1] + fraction * (pose[1] - old[1]))
            next_pole += 1
        if (step + 1) % program.segment_steps == 0:
            index = (step + 1) // program.segment_steps
            np.testing.assert_allclose(pose[:2], program.reference_points[index], atol=2e-10)
    assert len(crossings) == pole_count
    np.testing.assert_allclose(crossings, [1.0 if i % 2 == 0 else 0.0 for i in range(pole_count)], atol=0.003)
    expected_y = 0.0 if pole_count == 5 else 1.0
    np.testing.assert_allclose(pose, [program.finish_x_m, expected_y, 0.0], atol=2e-10)
    assert program.finish_x_m == pytest.approx(7.0 if pole_count == 5 else 6.0)
    assert program.duration_s == pytest.approx(program.total_control_steps * ctrl_dt)
    assert program.segment_steps >= 6
    assert np.all(program.commands[:, 0] > 0.0)
    np.testing.assert_array_equal(program.commands[:, 1], 0.0)
    assert np.any(program.commands[:, 2] > 0.0)
    assert np.any(program.commands[:, 2] < 0.0)
    assert np.max(np.abs(program.commands[:, 2])) < 0.8


def test_program_finishes_once_and_never_restarts_or_exposes_mutable_table():
    program = SlalomCommandProgram.from_config(SlalomConfig(), ctrl_dt=0.02)
    assert program.segment_steps == 40
    for step in (program.total_control_steps, program.total_control_steps + 1,
                 2 * program.total_control_steps, 10**9):
        np.testing.assert_array_equal(program.command_at_control_step(step), np.zeros(3))
        assert program.segment_index_at_control_step(step) == len(program.commands) - 1
        assert program.time_at_control_step(step) == pytest.approx(step * program.ctrl_dt)
    assert program.command_at_control_step(program.total_control_steps - 1)[0] > 0.0
    command = program.command_at_control_step(0)
    command[:] = 123.0
    assert program.command_at_control_step(0)[0] != 123.0
    assert not program.commands.flags.writeable
    assert not program.reference_points.flags.writeable
    assert not program.pole_centers.flags.writeable
    for invalid in (-1, True, 1.2):
        with pytest.raises(ValueError):
            program.command_at_control_step(invalid)


@pytest.mark.parametrize("field,value", [
    ("pole_count", 3), ("pole_count", True), ("enabled", 1),
    ("spacing_m", 0.0), ("speed_m_s", np.nan), ("pole_mass_kg", -1.0),
    ("pole_friction", -0.1), ("settle_s", -1.0), ("tip_angle_deg", 90.0),
    ("tip_hold_s", 0.0), ("first_pole_x_m", 0.5), ("pass_clearance_m", 0.5),
])
def test_invalid_course_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(SlalomConfig(), **{field: value})


def test_unrealizable_geometry_and_short_hold_rejected():
    with pytest.raises(ValueError, match="tangent angle"):
        SlalomCommandProgram.from_config(SlalomConfig(amplitude_m=5.0), ctrl_dt=0.02)
    with pytest.raises(ValueError, match="prediction_horizon"):
        SlalomCommandProgram.from_config(SlalomConfig(), ctrl_dt=0.02, prediction_horizon=41)
    for invalid in (0.0, np.inf, True):
        with pytest.raises(ValueError):
            SlalomCommandProgram.from_config(SlalomConfig(), ctrl_dt=invalid)
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError):
            SlalomCommandProgram.from_config(SlalomConfig(), ctrl_dt=0.02, prediction_horizon=invalid)


def test_environment_validation_reuses_command_owner_and_validates_final_stop():
    program = SlalomCommandProgram.from_config(SlalomConfig(), ctrl_dt=0.02)
    env = SimpleNamespace(commands=SimpleNamespace(
        resampling_time=0.0, heading_command=False,
        vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        dead_zone_enabled=False, small_xy_threshold=0.0,
    ))
    program.validate_environment(env, keyboard=False)
    with pytest.raises(ValueError, match="keyboard"):
        program.validate_environment(env, keyboard=True)
    env.commands.resampling_time = 1.0
    with pytest.raises(ValueError, match="resampling_time"):
        program.validate_environment(env, keyboard=False)
    env.commands.resampling_time = 0.0
    env.commands.vel_limit[0][0] = 0.01
    with pytest.raises(ValueError, match="vel_limit"):
        program.validate_environment(env, keyboard=False)


def _trial(config=None):
    config = SlalomConfig() if config is None else config
    centers = np.column_stack((config.first_pole_x_m + np.arange(config.pole_count) * config.spacing_m,
                               np.full(config.pole_count, config.amplitude_m)))
    evaluator = SlalomEvaluator(config, centers)
    positions = np.column_stack((centers, np.full(config.pole_count, config.pole_height_m / 2.0)))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (config.pole_count, 1))
    contacts = np.zeros(config.pole_count, dtype=bool)
    return evaluator, positions, quats, contacts


def _sample(trial, base=(0.0, 0.0), *, dt=0.02, finished=False, done=False):
    evaluator, positions, quats, contacts = trial
    evaluator.update(np.asarray(base), positions, quats, contacts, done, dt, finished)
    return evaluator.report()


def _pass_all_gates(trial):
    evaluator = trial[0]
    config = evaluator.config
    for index, center in enumerate(evaluator.pole_centers):
        _sample(trial, (center[0], 2.0 * config.amplitude_m if index % 2 == 0 else 0.0))


@pytest.mark.parametrize("pole_count", [4, 5])
def test_measured_gates_finish_and_full_settle_required_for_success(pole_count):
    trial = _trial(SlalomConfig(pole_count=pole_count))
    _pass_all_gates(trial)
    report = trial[0].report()
    assert report["passed_poles"] == pole_count
    assert not report["success"]
    finish = report["finish_x_m"]
    report = _sample(trial, (finish, 0.0), dt=10.0)
    assert not report["success"]
    assert report["settle_elapsed_s"] == 0.0
    for _ in range(99):
        report = _sample(trial, (finish, 0.0), finished=True)
    assert not report["success"]
    report = _sample(trial, (finish, 0.0), finished=True)
    assert report["success"]
    assert not report["failed"]
    assert report["settle_elapsed_s"] == pytest.approx(2.0)


def test_finished_command_clock_never_substitutes_for_actual_route():
    trial = _trial()
    report = _sample(trial, dt=1000.0, finished=True)
    assert report["route_finished"]
    assert report["passed_poles"] == 0
    assert not report["success"]
    assert not report["failed"]
    report = _sample(trial, (10.0, 0.0), finished=True)
    assert report["failed"]
    assert "skipped_gate" in report["failure_reasons"]
    assert not report["success"]


def test_wrong_side_and_bypass_fail_even_if_robot_reaches_finish():
    wrong = _trial()
    report = _sample(wrong, (1.5, 0.0))
    assert report["failed"]
    assert "pole_1_wrong_side_or_clearance" in report["failure_reasons"]
    bypass = _trial()
    for x in bypass[0].pole_centers[:, 0]:
        report = _sample(bypass, (x, 2.0))
    report = _sample(bypass, (7.0, 2.0), dt=3.0, finished=True)
    assert report["failed"]
    assert "pole_2_wrong_side_or_clearance" in report["failure_reasons"]
    assert not report["success"]


def test_gate_side_uses_interpolated_crossing_not_endpoint():
    trial = _trial()
    _sample(trial, (1.4, 0.3))
    report = _sample(trial, (1.6, 1.2))
    assert report["failed"]
    assert report["passed_poles"] == 0
    good = _trial()
    _sample(good, (1.4, 0.9))
    report = _sample(good, (1.6, 1.1))
    assert not report["failed"]
    assert report["passed_poles"] == 1
    report = _sample(good, (1.4, 1.0))
    assert not report["failed"]
    report = _sample(good, (1.6, 1.0))
    assert "out_of_order_gate" in report["failure_reasons"]


def test_tilt_hold_is_continuous_uses_body_z_and_latches():
    trial = _trial()
    quats = trial[2]
    angle = np.deg2rad(40.0)
    tipped = 3.0 * np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
    quats[0] = tipped
    assert not _sample(trial, dt=0.1)["pole_fallen"][0]
    quats[0] = [1.0, 0.0, 0.0, 0.0]
    assert not _sample(trial, dt=0.1)["pole_fallen"][0]
    quats[0] = tipped
    assert not _sample(trial, dt=0.1)["pole_fallen"][0]
    report = _sample(trial, dt=0.1)
    assert report["pole_fallen"][0]
    assert report["failed"]
    quats[0] = [1.0, 0.0, 0.0, 0.0]
    assert _sample(trial)["pole_fallen"][0]
    yaw_only = _trial()
    yaw_only[2][0] = [0.0, 0.0, 0.0, 2.0]
    assert not _sample(yaw_only, dt=1.0)["failed"]


def test_horizontal_pole_displacement_latches_but_vertical_motion_does_not():
    trial = _trial()
    trial[1][0, 2] += 1.0
    assert not _sample(trial)["failed"]
    trial[1][0, 0] += 0.11
    report = _sample(trial)
    assert report["pole_displaced"][0]
    assert "pole_1_displaced" in report["failure_reasons"]
    trial[1][0, 0] -= 0.11
    assert _sample(trial)["pole_displaced"][0]


def test_contacts_are_recorded_separately_and_do_not_fail_a_successful_trial():
    trial = _trial()
    trial[3][2] = True
    _sample(trial)
    _sample(trial)
    trial[3][:] = False
    _pass_all_gates(trial)
    report = _sample(trial, (7.0, 0.0), dt=2.0, finished=True)
    assert report["success"]
    assert not report["failed"]
    assert report["pole_contact"] == [False, False, True, False, False]
    assert report["contact_samples"] == [0, 0, 2, 0, 0]
    report["pole_contact"][2] = False
    assert trial[0].report()["pole_contact"][2]


def test_robot_done_and_physical_failure_during_settling_prevent_success():
    trial = _trial()
    _pass_all_gates(trial)
    assert not _sample(trial, (7.0, 0.0), dt=1.0, finished=True)["success"]
    report = _sample(trial, (7.0, 0.0), dt=1.0, finished=True, done=True)
    assert report["failed"]
    assert "robot_done" in report["failure_reasons"]
    assert not report["success"]
    moved = _trial()
    _pass_all_gates(moved)
    _sample(moved, (7.0, 0.0), dt=1.0, finished=True)
    moved[1][1, 1] += 0.2
    report = _sample(moved, (7.0, 0.0), dt=1.0, finished=True)
    assert report["failed"]
    assert not report["success"]


def test_settle_resets_when_actual_robot_retreats_before_finish():
    trial = _trial()
    _pass_all_gates(trial)
    _sample(trial, (7.0, 0.0), dt=1.0, finished=True)
    assert _sample(trial, (6.9, 0.0), finished=True)["settle_elapsed_s"] == 0.0
    assert not _sample(trial, (7.0, 0.0), dt=1.0, finished=True)["success"]
    assert _sample(trial, (7.0, 0.0), dt=1.0, finished=True)["success"]


def test_evaluator_rejects_invalid_shapes_and_quaternions():
    trial = _trial()
    evaluator, positions, quats, contacts = trial
    with pytest.raises(ValueError, match="base_xy"):
        evaluator.update(np.zeros(3), positions, quats, contacts, False, 0.02, False)
    with pytest.raises(ValueError, match="pole_positions"):
        evaluator.update(np.zeros(2), positions[None], quats, contacts, False, 0.02, False)
    with pytest.raises(ValueError, match="contacts"):
        evaluator.update(np.zeros(2), positions, quats, contacts.astype(int), False, 0.02, False)
    quats[0] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        _sample(trial)
