"""Cold-path source identity and official-config coverage; never step a model."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest

from fada_oracle.environment import build_experiment_config, materialize_scene



def _source_hash(scene: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene.parent.rglob("*.xml")):
        if path.is_file():
            digest.update(path.relative_to(scene.parent).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@pytest.fixture(scope="module")
def source_scene(tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("source-scene")
    joint_stems = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
        "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
        "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
        "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
        "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
        "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ]
    robot = ET.Element("mujoco", model="configuration-test-g1")
    ET.SubElement(robot, "compiler", angle="radian", autolimits="true")
    ET.SubElement(robot, "option", timestep=str(0.02 / 3))
    world = ET.SubElement(robot, "worldbody")
    pelvis = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.754")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    ET.SubElement(pelvis, "inertial", pos="0 0 0", mass="4.5", diaginertia="0.1 0.2 0.3")
    ET.SubElement(pelvis, "geom", type="sphere", size="0.02")
    actuators = ET.SubElement(robot, "actuator")
    for index, stem in enumerate(joint_stems):
        body_name = "torso_link" if stem == "waist_pitch" else stem + "_link"
        body = ET.SubElement(pelvis, "body", name=body_name, pos=f"0 0 {0.03 * (index + 1)}")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(0.3 + 0.01 * index), diaginertia="0.01 0.02 0.03")
        ET.SubElement(body, "geom", type="sphere", size="0.01")
        ET.SubElement(body, "joint", name=stem + "_joint", axis="0 1 0", range=f"{-0.5 - 0.01 * index} {1.5 + 0.01 * index}", armature=str(0.01 + index * 0.001), frictionloss=str(0.1 + index * 0.01))
        limit = 20 + index
        ET.SubElement(actuators, "position", name=stem + "_joint", joint=stem + "_joint", kp=str(40.0 + index), kv=str(2.0 + index * 0.01), forcerange=f"{-limit} {limit}")
    keyframes = ET.SubElement(robot, "keyframe")
    defaults = [index * 0.001 for index in range(29)]
    ET.SubElement(keyframes, "key", name="stand", qpos=" ".join(map(str, [0, 0, 0.754, 1, 0, 0, 0, *defaults])))
    ET.ElementTree(robot).write(source / "g1.xml", encoding="utf-8")
    scene = source / "scene.xml"
    scene.write_text('<mujoco><include file="g1.xml"/><worldbody><geom name="floor" type="plane" size="0 0 0.05"/></worldbody></mujoco>')
    (source / "auxiliary.bin").write_bytes(b"non-mesh asset preservation fixture")
    (source / "historical-empty.xml").write_bytes(b"")
    (source / "unused.xml").write_text('<mujoco><actuator><position name="unused" kp="1" kv="2"/></actuator></mujoco>')
    return scene


@pytest.fixture(scope="module")
def source_metadata(source_scene) -> dict:
    robot = ET.parse(source_scene.parent / "g1.xml")
    positions = robot.findall(".//actuator/position")
    return {
        "asset_sha256": _source_hash(source_scene),
        "body_names": ["world", *[body.attrib["name"] for body in robot.findall(".//body")]],
        "actuated_joint_names": [actuator.attrib["joint"] for actuator in positions],
        "default_joint_pos": [index * 0.001 for index in range(29)],
        "kp": [float(actuator.attrib["kp"]) for actuator in positions],
        "kd": [float(actuator.attrib["kv"]) for actuator in positions],
        "sim_dt": 0.02 / 3,
        "ctrl_dt": 0.02,
        "action_scale": 0.75,
        "environment_config": {
            "scene": {"model_file": str(source_scene), "fragment_files": [], "terrain": None},
            "sim_dt": 0.02 / 3, "ctrl_dt": 0.02, "max_episode_seconds": 20.0,
            "control_config": {"action_scale": 0.75, "simulate_action_latency": False},
            "reward_config": {"max_tilt_deg": 65.0, "min_base_height": 0.3, "gait_frequency": 1.0},
            "commands": {"vel_limit": [[-0.8, -0.4, -0.6], [1.2, 0.5, 0.7]], "resampling_time": 8.0, "rel_standing_envs": 0.25},
            "domain_rand": {
                "randomize_reset_pose": True, "randomize_base_mass": True,
                "added_mass_range": [-1.5, 1.5], "randomize_ground_friction": True,
                "ground_friction_multiplier_range": [0.8, 1.2],
                "randomize_kp": True, "kp_multiplier_range": [0.9, 1.1],
                "randomize_kd": True, "kd_multiplier_range": [0.85, 1.15],
                "randomize_body_mass": False, "random_com": False,
                "randomize_gravity": False, "randomize_dof_armature": False,
                "push_robots": False, "randomize_dof_position_bias": False,
                "randomize_control_delay": False, "torque_rfi_fraction": 0.0,
                "actuator_strength": {"enabled": False},
            },
        },
    }


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, source_scene, source_metadata):
    return materialize_scene(source_scene, tmp_path_factory.mktemp("source-assets") / "owned", source_metadata)




def test_unreferenced_empty_and_position_xml_remain_untouched(prepared, source_scene, source_metadata):
    for name in ("historical-empty.xml", "unused.xml"):
        assert (prepared.scene_file.parent / name).read_bytes() == (source_scene.parent / name).read_bytes()
    assert (prepared.scene_file.parent / "historical-empty.xml").stat().st_size == 0
    assert ET.parse(prepared.scene_file.parent / "unused.xml").find(".//actuator/position") is not None
    assert _source_hash(source_scene) == source_metadata["asset_sha256"] == prepared.source_sha256
    provenance = json.loads(prepared.provenance_file.read_text())
    assert provenance["transformed_files"] == ["g1.xml"]


def test_source_copy_retains_identity_assets_and_only_converts_position_actuators(prepared, source_scene, source_metadata):
    assert _source_hash(source_scene) == source_metadata["asset_sha256"] == prepared.source_sha256
    assert _source_hash(prepared.scene_file) == prepared.transformed_sha256
    assert prepared.transformed_sha256 != prepared.source_sha256
    provenance = json.loads(prepared.provenance_file.read_text())
    assert provenance["source_sha256"] == prepared.source_sha256
    assert provenance["transformed_sha256"] == prepared.transformed_sha256
    original = ET.parse(source_scene.parent / "g1.xml")
    copied = ET.parse(prepared.scene_file.parent / "g1.xml")
    positions = original.findall(".//actuator/position")
    motors = copied.findall(".//actuator/motor")
    assert len(positions) == len(motors) == 29
    assert not copied.findall(".//actuator/position")
    for position, motor in zip(positions, motors, strict=True):
        assert motor.attrib == {key: value for key, value in position.attrib.items() if key not in {"kp", "kv"}}
    assert [node.attrib for node in original.findall(".//inertial")] == [node.attrib for node in copied.findall(".//inertial")]
    for path in source_scene.parent.rglob("*"):
        if path.is_file() and path.suffix != ".xml":
            target = prepared.scene_file.parent / path.relative_to(source_scene.parent)
            assert target.stat().st_size == path.stat().st_size > 0
            assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(path.read_bytes()).digest()
    assert materialize_scene(source_scene, prepared.scene_file.parent, source_metadata) == prepared


@pytest.mark.parametrize("target", ["self", "descendant", "ancestor", "alias"])
def test_asset_copy_rejects_recursive_or_aliased_targets(tmp_path, source_metadata, target):
    source = tmp_path / "source"
    source.mkdir()
    scene = source / "scene.xml"
    scene.write_text('<mujoco><actuator><position name="joint" kp="1" kv="1"/></actuator></mujoco>')
    metadata = {"asset_sha256": _source_hash(scene)}
    destination = {"self": source, "descendant": source / "copy", "ancestor": tmp_path}.get(target)
    if target == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(source, target_is_directory=True)
    before = scene.read_bytes()
    with pytest.raises(ValueError, match="disjoint"):
        materialize_scene(scene, destination, metadata)
    assert scene.read_bytes() == before


def test_asset_copy_rejects_hash_mismatch_without_creating_target(tmp_path, source_scene, source_metadata):
    metadata = dict(source_metadata, asset_sha256="0" * 64)
    destination = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="asset_sha256"):
        materialize_scene(source_scene, destination, metadata)
    assert not destination.exists()




@pytest.mark.parametrize("backend,device,num_envs", [("classic", "cpu", 2), ("classic", "cuda:0", 1), ("warp", "cpu", 64), ("warp", "mps", 64), ("isaacsim", "cuda", 64)])
def test_backend_contract_rejects_fallbacks(tmp_path, backend, device, num_envs):
    with pytest.raises(ValueError):
        build_experiment_config({}, tmp_path / "unused.xml", backend=backend, device=device, num_envs=num_envs, seed=0, run_dir=tmp_path)


@pytest.mark.parametrize("field", ["randomize_body_mass", "random_com", "randomize_gravity", "randomize_dof_armature", "push_robots", "randomize_dof_position_bias", "randomize_control_delay"])
def test_unsupported_enabled_source_randomization_is_rejected(source_metadata, tmp_path, field):
    changed = deepcopy(source_metadata)
    changed["environment_config"]["domain_rand"][field] = True
    with pytest.raises(ValueError, match="Unsupported enabled source domain randomization"):
        build_experiment_config(changed, tmp_path / "unused.xml", backend="classic", device="cpu", num_envs=1, seed=0, run_dir=tmp_path)


@pytest.mark.parametrize("field,value", [("kp", [1.0]*28), ("kd", [float("nan")]*29), ("default_joint_pos", []), ("ctrl_dt", 0.03), ("action_scale", 0.5)])
def test_invalid_or_inconsistent_controls_are_rejected(source_metadata, tmp_path, field, value):
    changed = deepcopy(source_metadata)
    changed[field] = value
    with pytest.raises(ValueError, match=field):
        build_experiment_config(changed, tmp_path / "unused.xml", backend="classic", device="cpu", num_envs=1, seed=0, run_dir=tmp_path)
