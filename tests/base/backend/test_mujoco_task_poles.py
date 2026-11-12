from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg, ScenePoleCfg, TerrainSceneCfg
from unilab.dr.types import ResetRandomizationPayload


@contextmanager
def _backend(scene, *, num_envs=1, add_body_sensors=False):
    backend = MuJoCoBackend(
        scene, num_envs, 0.005, base_name="pelvis", num_threads=1,
        add_body_sensors=add_body_sensors,
    )
    try:
        backend.materialize()
        yield backend
    finally:
        backend.close()


def _xml_fingerprint(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*.xml")}


@pytest.fixture
def toy_scene(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text('''<mujoco>
      <option timestep="0.005"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 .1"/>
        <body name="pelvis" pos="-3 0 .5">
          <freejoint name="root"/>
          <geom name="robot_collision" type="sphere" size=".12" mass="8"/>
          <body name="robot_child" pos="0 0 .2">
            <joint name="hinge" type="hinge"/>
            <geom type="sphere" size=".05" mass=".2"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="motor" joint="hinge" ctrlrange="-1 1"/></actuator>
      <keyframe><key name="stand" qpos="-3 0 .5 1 0 0 0 .25"/></keyframe>
    </mujoco>''', encoding="utf-8")
    return path


@pytest.mark.parametrize("kwargs", [
    {"name": ""}, {"position": (0, 0)}, {"position": (0, np.nan, 0)},
    {"radius": 0}, {"height": -1}, {"mass": np.inf}, {"friction": -1},
])
def test_invalid_cold_pole_configuration(kwargs):
    values = dict(name="pole", position=(1.5, .5, .4))
    values.update(kwargs)
    with pytest.raises(ValueError):
        ScenePoleCfg(**values)


def test_pole_scene_rejects_ambiguous_composition():
    pole = ScenePoleCfg("pole", (1.5, .5, .4))
    for kwargs in ({"fragment_files": ["task.xml"]}, {"terrain": TerrainSceneCfg()}):
        with pytest.raises(ValueError, match="fragments or terrain"):
            SceneCfg("robot.xml", task_poles=[pole], **kwargs)
    with pytest.raises(ValueError, match="unique"):
        SceneCfg("robot.xml", task_poles=[pole, pole])


@pytest.mark.parametrize("add_body_sensors", [False, True])
def test_g1_poles_preserve_robot_prefix_and_canonical_fingerprint(add_body_sensors):
    source = ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    before = _xml_fingerprint(source.parent)
    poles = [ScenePoleCfg(f"slalom_pole_{i}", (1.5 + i, .5, .4)) for i in range(5)]
    with _backend(SceneCfg(str(source), task_poles=poles), add_body_sensors=add_body_sensors) as backend:
        model = backend.model
        assert model.nq == original.nq + 35
        assert model.nv == original.nv + 30
        assert model.nbody == original.nbody + 5
        assert model.nu == original.nu == 29
        assert backend.num_dof_vel == 29
        assert backend.get_dof_pos().shape == backend.get_dof_vel().shape == (1, 29)
        assert backend.get_joint_range().shape == (29, 2)
        np.testing.assert_array_equal(backend.get_robot_body_ids(), np.arange(original.nbody))
        np.testing.assert_array_equal(backend.get_dof_armature(), original.dof_armature)
        for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, original.nbody),
                            (mujoco.mjtObj.mjOBJ_JOINT, original.njnt),
                            (mujoco.mjtObj.mjOBJ_ACTUATOR, original.nu)):
            assert [mujoco.mj_id2name(model, kind, i) for i in range(count)] == [
                mujoco.mj_id2name(original, kind, i) for i in range(count)]
        np.testing.assert_array_equal(model.jnt_qposadr[:original.njnt], original.jnt_qposadr)
        np.testing.assert_array_equal(model.jnt_dofadr[:original.njnt], original.jnt_dofadr)
        np.testing.assert_array_equal(model.key_qpos[:, :original.nq], original.key_qpos)
        np.testing.assert_array_equal(model.key_qvel[:, :original.nv], original.key_qvel)
        expected_props = np.asarray([(*pole.position, 1, 0, 0, 0) for pole in poles]).ravel()
        np.testing.assert_array_equal(model.key_qpos[0, original.nq:], expected_props)
        np.testing.assert_array_equal(model.key_qvel[:, original.nv:], 0)
        np.testing.assert_array_equal(backend.get_keyframe_actuator_positions("stand"),
                                      original.key_qpos[0, original.jnt_qposadr[original.actuator_trnid[:, 0]]])
        assert backend.get_init_qvel().shape == (model.nv,)
        assert len(backend.get_body_names()) == model.nbody
        assert backend.get_body_mass().shape == (model.nbody,)
        for pole in poles:
            body = backend.get_body_id(pole.name)
            geom = backend.get_geom_id(f"{pole.name}_geom")
            joint = int(model.body_jntadr[body])
            assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
            assert model.geom_contype[geom] and model.geom_conaffinity[geom]
            np.testing.assert_allclose(model.body_mass[body], pole.mass)
            np.testing.assert_allclose(model.geom_size[geom, :2], [pole.radius, pole.height / 2])
        artifacts = backend.get_scene_artifacts()
        assert artifacts.model_file == str(source)
        assert artifacts.visual_model_file == backend._model_file
        private = Path(artifacts.artifacts_dir)
        assert private != source.parent and private.is_dir()
        visual = mujoco.MjModel.from_xml_path(artifacts.visual_model_file)
        assert visual.nbody == model.nbody
        assert any((private / "assets").iterdir())
        assert _xml_fingerprint(source.parent) == before
    assert not private.exists()
    assert _xml_fingerprint(source.parent) == before


def test_real_pole_pose_contacts_are_current_and_reset_per_row(toy_scene):
    scene = dict(model_file=str(toy_scene), task_poles=[
        dict(name="pole", position=(0, 0, .4)),
        dict(name="other_pole", position=(2, 0, .4)),
    ])
    with _backend(scene, num_envs=2) as backend:
        qpos = np.tile(backend.get_keyframe_qpos("stand"), (2, 1))
        qvel = np.zeros((2, backend.nv))
        # Row 0 intersects the robot's child body, not its root sphere.
        qpos[0, :3] = [0, 0, .95]
        # Move the child into the cylinder while keeping root sphere clear.
        qpos[0, 3:7] = [0, 1, 0, 0]
        backend.set_state(np.arange(2), qpos, qvel)
        positions, quats, contact = backend.get_task_pole_state()
        assert positions.shape == (2, 2, 3) and quats.shape == (2, 2, 4)
        assert contact.dtype == np.bool_
        assert contact[0, 0] and not contact[0, 1]
        assert not contact[1].any()  # Floor contact alone is not robot contact.
        np.testing.assert_allclose(positions[:, 0], [[0, 0, .4]] * 2)
        # A tilted, displaced prop is read from physical frame sensors.
        pole_joint = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_free")
        address = int(backend.model.jnt_qposadr[pole_joint])
        qpos[1, address:address + 7] = [1, 1, .2, np.sqrt(.5), 0, np.sqrt(.5), 0]
        backend.set_state(np.asarray([1]), qpos[1:], qvel[1:])
        positions, quats, contact = backend.get_task_pole_state()
        np.testing.assert_allclose(positions[1, 0], [1, 1, .2])
        np.testing.assert_allclose(quats[1, 0], [np.sqrt(.5), 0, np.sqrt(.5), 0])
        assert contact[0, 0]  # Partial reset preserves the untouched row.
        backend.reset_physics_rows_to_default(np.asarray([0]))
        positions, quats, contact = backend.get_task_pole_state()
        assert not contact.any()  # Not latched after separation/reset.
        np.testing.assert_allclose(positions[0, 0], [0, 0, .4])
        np.testing.assert_allclose(positions[1, 0], [1, 1, .2])
        snapshot = backend.capture_rollout_state()
        backend.reset_physics_rows_to_default(np.asarray([1]))
        backend.restore_rollout_state(snapshot)
        np.testing.assert_allclose(backend.get_task_pole_state()[0][1, 0], [1, 1, .2])


def test_robot_collision_imparts_real_momentum_to_pole(toy_scene):
    scene = SceneCfg(str(toy_scene), task_poles=[ScenePoleCfg("pole", (0, 0, .4))])
    with _backend(scene) as backend:
        qpos = backend.get_keyframe_qpos("stand")[None]
        qpos[0, :3] = [-.18, 0, .4]
        qvel = backend.get_init_qvel()[None]
        qvel[0, 0] = 2.0
        backend.set_state(np.asarray([0]), qpos, qvel)
        assert backend.get_task_pole_state()[2][0, 0]
        backend.step(np.zeros((1, backend.num_actuators)), nsteps=2)
        joint = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_free")
        velocity_address = backend.model.jnt_dofadr[joint]
        full_velocity = backend.get_physics_state()[0, 1 + backend.nq:1 + backend.nq + backend.nv]
        assert full_velocity[velocity_address] > 0.0


def test_free_pole_moves_and_topples_under_real_force(toy_scene):
    with _backend(SceneCfg(str(toy_scene), task_poles=[ScenePoleCfg("pole", (0, 0, .4))])) as backend:
        body_id = np.asarray([backend.get_body_id("pole")], dtype=np.int32)
        ctrl = np.zeros((1, backend.num_actuators))
        initial = backend.get_task_pole_state()[0].copy()
        backend.apply_body_force(body_id, np.asarray([[[100., 0., 0.]]]))
        backend.step(ctrl, nsteps=20)
        backend.step(ctrl, nsteps=100)
        positions, quats, contact = backend.get_task_pole_state()
        assert np.linalg.norm(positions[0, 0, :2] - initial[0, 0, :2]) > .1
        assert 1 - 2 * (quats[0, 0, 1] ** 2 + quats[0, 0, 2] ** 2) < np.cos(np.deg2rad(30))
        assert not contact.any()
        # Post-step sensor values describe the current qpos, not the prior substep.
        joint = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_free")
        start = backend.model.jnt_qposadr[joint]
        np.testing.assert_allclose(positions[0, 0], backend._qpos_view[0, start:start + 3])
        backend.reset_physics_rows_to_default(np.asarray([0]))
        np.testing.assert_allclose(backend.get_task_pole_state()[0][0, 0], [0, 0, .4])
        np.testing.assert_allclose(backend.get_task_pole_state()[1][0, 0], [1, 0, 0, 0])


def test_robot_armature_reset_does_not_randomize_prop_dofs(toy_scene):
    with _backend(SceneCfg(str(toy_scene), task_poles=[ScenePoleCfg("pole", (0, 0, .4))])) as backend:
        armature = np.full((1, 7), .2)
        backend.set_state(np.asarray([0]), backend.get_keyframe_qpos("stand")[None],
                          backend.get_init_qvel()[None],
                          ResetRandomizationPayload(dof_armature=armature))
        assert backend.get_dof_armature().shape == (7,)
        assert backend.get_dof_vel().shape == (1, 1)


def test_empty_poles_leave_model_unchanged(toy_scene):
    original = mujoco.MjModel.from_xml_path(str(toy_scene))
    with _backend(SceneCfg(str(toy_scene))) as backend:
        assert (backend.nq, backend.nv) == (original.nq, original.nv)
        assert backend.get_scene_artifacts().model_file == str(toy_scene)
        assert backend.get_scene_artifacts().artifacts_dir is None
        assert backend.get_task_pole_state()[0].shape == (1, 0, 3)
        assert backend.get_task_pole_state()[2].shape == (1, 0)
