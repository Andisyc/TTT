"""Figure-eight turn direction and closed-path command integration."""

import numpy as np
import pytest

from unilab.visualization.demo_commands import (
    DoubleCircleCommandConfig,
    DoubleCircleCommandProgram,
    FigureEightCommandConfig,
    FigureEightCommandProgram,
)


@pytest.mark.parametrize("width,height", [(4.0, 2.0), (6.0, 3.0), (4.0, 1.0), (0.4, 0.2)])
def test_figure_eight_turns_without_reversals_and_closes(width, height):
    program = FigureEightCommandProgram.from_config(
        FigureEightCommandConfig(width_m=width, height_m=height), ctrl_dt=0.02,
    )
    commands = program.commands
    half = len(commands) // 2
    assert np.all(commands[:half, 2] < 0.0)
    assert np.all(commands[half:, 2] > 0.0)
    assert np.all(commands[:, 0] > 0.0)
    np.testing.assert_array_equal(commands[:, 1], 0.0)

    turns = commands[:, 2] * program.segment_duration_s
    headings = np.concatenate(([0.0], np.cumsum(turns)))
    mean_headings = (headings[:-1] + headings[1:]) / 2.0
    distances = commands[:, 0] * program.segment_duration_s * np.sinc(turns / (2.0 * np.pi))
    positions = np.vstack((np.zeros(2), np.cumsum(
        distances[:, None] * np.column_stack((np.cos(mean_headings), np.sin(mean_headings))),
        axis=0,
    )))
    np.testing.assert_allclose(positions, program.reference_points, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(positions[half], 0.0, atol=1e-12)
    np.testing.assert_allclose(positions[-1], 0.0, atol=1e-12)
    assert abs(headings[-1]) < 1e-12
    assert program.segment_steps == 40
    assert program.segment_index_at_control_step(program.total_control_steps) == 0


def test_double_circle_holds_unequal_durations_and_repeats_at_cycle_boundary():
    program = DoubleCircleCommandProgram.from_config(
        DoubleCircleCommandConfig(0.3, 0.3, 1.01, 1.07), ctrl_dt=0.02,
    )
    assert program.phase_steps == (51, 54)
    assert program.total_control_steps == 105
    assert program.duration_s == pytest.approx(2.1)
    for step in (0, 50, 105):
        np.testing.assert_allclose(program.command_at_control_step(step), [0.3, 0.0, -0.3])
    for step in (51, 104):
        np.testing.assert_allclose(program.command_at_control_step(step), [0.3, 0.0, 0.3])
    np.testing.assert_allclose(
        np.linalg.norm(program.reference_points[:129] - [0.0, -1.0], axis=1), 1.0,
    )
    np.testing.assert_allclose(
        np.linalg.norm(program.reference_points[128:] - [0.0, 1.0], axis=1), 1.0,
    )
    np.testing.assert_array_equal(program.reference_points[[0, 128, -1]], 0.0)
