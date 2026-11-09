"""Independent numeric checks for source teacher feature semantics."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fada_oracle.environment.features import actor_observation, pack_privileged, sample_commands


def _actor_inputs() -> dict:
    return {
        "gyro": torch.tensor([[4.0, -8.0, 12.0]]),
        "upvector": torch.tensor([[0.2, -0.4, 0.8]]),
        "dof_pos": torch.arange(2.0, 59.0, 2.0).unsqueeze(0),
        "dof_vel": torch.arange(20.0, 581.0, 20.0).unsqueeze(0),
        "default_joint_pos": torch.arange(1.0, 30.0),
        "last_actions": -torch.arange(1.5, 30.0).unsqueeze(0),
        "commands": torch.tensor([[0.7, -0.9, 1.3]]),
        "gait_phase": torch.tensor([[0.2, 0.8]]),
        "noise_config": {
            "level": 0.4, "scale_gyro": 2.0, "scale_gravity": 0.2,
            "scale_joint_angle": 0.6, "scale_joint_vel": 4.0,
        },
    }


_CLEAN_OBSERVATION = [
    1, -2, 3, -0.2, 0.4, -0.8,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5,
    -10.5, -11.5, -12.5, -13.5, -14.5, -15.5, -16.5, -17.5,
    -18.5, -19.5, -20.5, -21.5, -22.5, -23.5, -24.5, -25.5,
    -26.5, -27.5, -28.5, -29.5,
    0.7, -0.9, 1.3, 0.2, 0.8,
]


def test_actor_source_order_scales_world_upvector_sign_and_raw_actions_phase() -> None:
    actual = actor_observation(**_actor_inputs(), add_noise=False)
    assert actual.shape == (1, 98)
    torch.testing.assert_close(actual, torch.tensor([_CLEAN_OBSERVATION]), rtol=1e-6, atol=1e-6)


def test_actor_noise_is_uniform_in_sensor_units_before_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    # A uniform [0, 1) draw of 0.75 is a source uniform [-1, 1) draw of +0.5.
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.75))
    actual = actor_observation(**_actor_inputs())
    expected = torch.tensor([[
        1.1, -1.9, 3.1, -0.24, 0.36, -0.84,
        1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12,
        11.12, 12.12, 13.12, 14.12, 15.12, 16.12, 17.12, 18.12, 19.12,
        20.12, 21.12, 22.12, 23.12, 24.12, 25.12, 26.12, 27.12, 28.12, 29.12,
        1.04, 2.04, 3.04, 4.04, 5.04, 6.04, 7.04, 8.04, 9.04, 10.04,
        11.04, 12.04, 13.04, 14.04, 15.04, 16.04, 17.04, 18.04, 19.04,
        20.04, 21.04, 22.04, 23.04, 24.04, 25.04, 26.04, 27.04, 28.04, 29.04,
        -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5,
        -10.5, -11.5, -12.5, -13.5, -14.5, -15.5, -16.5, -17.5,
        -18.5, -19.5, -20.5, -21.5, -22.5, -23.5, -24.5, -25.5,
        -26.5, -27.5, -28.5, -29.5,
        0.7, -0.9, 1.3, 0.2, 0.8,
    ]])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("level", [0.0, -0.5])
def test_nonpositive_source_noise_level_preserves_clean_features(level: float) -> None:
    inputs = _actor_inputs()
    inputs["noise_config"] = SimpleNamespace(**{**inputs["noise_config"], "level": level})
    actual = actor_observation(**inputs)
    torch.testing.assert_close(actual, torch.tensor([_CLEAN_OBSERVATION]), rtol=1e-6, atol=1e-6)


def test_actor_batched_defaults_and_inputs_preserve_rows_without_mutation() -> None:
    inputs = _actor_inputs()
    inputs["noise_config"] = SimpleNamespace(**inputs["noise_config"])
    for name, value in tuple(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[name] = value.repeat(2, 1)
    original = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    inputs["default_joint_pos"][1] += 0.5
    original["default_joint_pos"][1] += 0.5
    expected = torch.tensor([_CLEAN_OBSERVATION, _CLEAN_OBSERVATION])
    expected[1, 6:35] = torch.tensor([
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
        20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5,
    ])
    torch.testing.assert_close(actor_observation(**inputs, add_noise=False), expected, rtol=1e-6, atol=1e-6)
    for name, value in original.items():
        torch.testing.assert_close(inputs[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("name,width", [
    ("gyro", 2), ("upvector", 2), ("dof_pos", 28), ("dof_vel", 28),
    ("last_actions", 28), ("commands", 4), ("gait_phase", 1),
])
def test_actor_rejects_wrong_feature_width(name: str, width: int) -> None:
    inputs = _actor_inputs()
    inputs[name] = torch.ones(1, width)
    with pytest.raises(ValueError, match=name):
        actor_observation(**inputs, add_noise=False)


# Hand-written source slice table, independent of the production module constants.
_SLICES = [
    ["base_linear_velocity", 0, 3],
    ["foot_contact_resultants", 3, 9],
    ["foot_contact_flags", 9, 11],
    ["terrain_heights", 11, 20],
    ["root_clearance", 20, 21],
    ["kp_scale", 21, 50],
    ["kd_scale", 50, 79],
    ["normalized_torque", 79, 108],
    ["ground_friction", 108, 109],
    ["base_com_shift", 109, 112],
    ["added_base_mass", 112, 113],
    ["body_mass_scale", 113, 144],
    ["dof_position_bias", 144, 173],
    ["torque_rfi", 173, 202],
    ["control_delay", 202, 203],
    ["push_interval", 203, 204],
    ["push_velocity", 204, 205],
]


def _privileged_inputs() -> tuple[dict[str, torch.Tensor], dict]:
    ranges = {
        "base_linear_velocity": (1, 4), "foot_contact_resultants": (4, 10),
        "foot_contact_flags": (10, 12), "terrain_heights": (12, 21),
        "root_clearance": (21, 22), "kp_scale": (22, 51), "kd_scale": (51, 80),
        "normalized_torque": (80, 109), "ground_friction": (109, 110),
        "base_com_shift": (110, 113), "added_base_mass": (113, 114),
        "body_mass_scale": (114, 145), "dof_position_bias": (145, 174),
        "torque_rfi": (174, 203), "control_delay": (203, 204),
        "push_interval": (204, 205), "push_velocity": (205, 206),
    }
    # Reverse mapping insertion order: only checkpoint slice order is authoritative.
    fields = {
        name: torch.arange(float(start), float(stop)).unsqueeze(0)
        for name, (start, stop) in reversed(tuple(ranges.items()))
    }
    return fields, {"privileged_field_slices": deepcopy(_SLICES)}


def test_privileged_pack_preserves_all_nonzero_actual_values_in_metadata_order() -> None:
    fields, metadata = _privileged_inputs()
    expected = torch.tensor([[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
        181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
        201, 202, 203, 204, 205,
    ]], dtype=torch.float32)
    actual = pack_privileged(fields, metadata)
    assert actual.shape == (1, 205)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = {name: value.repeat(2, 1) for name, value in fields.items()}
    changed["normalized_torque"][1, 7] = -0.375
    expected_batch = expected.repeat(2, 1)
    expected_batch[1, 86] = -0.375
    torch.testing.assert_close(pack_privileged(changed, metadata), expected_batch, rtol=0, atol=0)


@pytest.mark.parametrize("field", [entry[0] for entry in _SLICES])
def test_privileged_pack_rejects_every_missing_field(field: str) -> None:
    fields, metadata = _privileged_inputs()
    del fields[field]
    with pytest.raises(ValueError, match=f"missing required privileged fields: {field}"):
        pack_privileged(fields, metadata)


@pytest.mark.parametrize("failure", [
    "absent", "unknown_field", "wrong_order", "gap", "overlap", "wrong_width",
    "wrong_total", "duplicate", "extra", "missing", "noninteger_offset",
])
def test_privileged_pack_rejects_non_source_schema(failure: str) -> None:
    fields, metadata = _privileged_inputs()
    slices = metadata["privileged_field_slices"]
    if failure == "absent":
        metadata.clear()
    elif failure == "unknown_field":
        slices[0][0] = "invented_velocity"
    elif failure == "wrong_order":
        slices[0][0], slices[1][0] = slices[1][0], slices[0][0]
    elif failure == "gap":
        slices[1][1] = 4
    elif failure == "overlap":
        slices[1][1] = 2
    elif failure == "wrong_width":
        slices[0][2], slices[1][1] = 4, 4
    elif failure == "wrong_total":
        slices[-1][2] = 206
    elif failure == "duplicate":
        slices[1][0] = slices[0][0]
    elif failure == "extra":
        slices.append(["extra", 205, 206])
    elif failure == "missing":
        slices.pop()
    else:
        slices[0][1] = 0.0
    with pytest.raises(ValueError, match="privileged_field_slices"):
        pack_privileged(fields, metadata)


@pytest.mark.parametrize("failure", ["width", "batch", "nonfinite", "rank"])
def test_privileged_pack_rejects_invalid_actual_field(failure: str) -> None:
    fields, metadata = _privileged_inputs()
    if failure == "width":
        fields["body_mass_scale"] = torch.ones(1, 30)
    elif failure == "batch":
        fields["kp_scale"] = torch.ones(2, 29)
    elif failure == "nonfinite":
        fields["torque_rfi"][0, 0] = float("nan")
    else:
        fields["root_clearance"] = torch.tensor([0.75])
    with pytest.raises(ValueError):
        pack_privileged(fields, metadata)


def test_commands_preserve_disjoint_standing_straight_and_general_source_distribution():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        commands = sample_commands(
            8192, {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.0, 1.0]},
            standing=0.2, straight=0.12, small_xy_threshold=0.0, device="cpu",
        )
    standing = (commands == 0).all(dim=1)
    straight = (commands[:, 0] > 0) & (commands[:, 1:] == 0).all(dim=1)
    assert 0.18 < float(standing.float().mean()) < 0.22
    assert 0.10 < float(straight.float().mean()) < 0.14
    assert bool((commands.abs() <= 1).all())
