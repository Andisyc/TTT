from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from unilab.algos.torch.distill.fada.observation import project_official_compact_observation
from unilab.algos.torch.distill.fada.official_deployment import OfficialFADADeploymentAdapter
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.footstand import Go2FootStandTask
from unilab.training.oracle.evaluation import PHYSICAL_COLUMNS, _base_measurements

mujoco = pytest.importorskip("mujoco")


def _world_angular_velocity(model, qpos, qvel):
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
    result = np.empty((len(qpos), 3))
    spatial_velocity = np.empty(6)
    for row in range(len(qpos)):
        data.qpos[:] = qpos[row]
        data.qvel[:] = qvel[row]
        mujoco.mj_forward(model, data)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial_velocity, 0,
        )
        result[row] = spatial_velocity[:3]
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonzero_angular_frames_through_backend_and_consumers(tmp_path, dtype):
    model_path = tmp_path / "free_root.xml"
    model_path.write_text(
        '<mujoco><worldbody><body name="root"><freejoint/>'
        '<geom type="box" size="0.1 0.1 0.1" mass="1"/>'
        '</body></worldbody></mujoco>',
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_path)), 3, 0.01,
        base_name="root", np_dtype=dtype, num_threads=1,
    )
    try:
        backend.materialize()
        rotations = Rotation.from_euler("xyz", [[0, 0, np.pi / 2], [0, np.pi / 2, 0], [0.3, -0.4, 0.5]])
        quat_xyzw = rotations.as_quat()
        qpos = np.tile(backend.model.qpos0, (3, 1))
        qpos[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
        qvel = np.array([
            [0.4, -0.3, 0.2, 1.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.0, 0.0, 1.0],
            [-0.2, 0.5, -0.4, 0.4, -0.7, 1.1],
        ])
        rows = np.arange(3)
        backend.set_state(rows, qpos, qvel)
        original_state = backend.get_physics_state().copy()
        expected_world = _world_angular_velocity(backend.model, qpos, qvel)
        world = backend.get_base_ang_vel()
        np.testing.assert_allclose(world, expected_world, rtol=2e-6, atol=2e-6)
        assert world.dtype == dtype
        np.testing.assert_array_equal(backend.get_physics_state(), original_state)

        env = SimpleNamespace(
            num_envs=3,
            get_base_pos=backend.get_base_pos,
            get_base_quat=backend.get_base_quat,
            get_base_lin_vel=backend.get_base_lin_vel,
            get_base_ang_vel=backend.get_base_ang_vel,
            get_dof_pos=lambda: np.zeros((3, 29)),
            get_dof_vel=lambda: np.zeros((3, 29)),
        )
        adapter = object.__new__(OfficialFADADeploymentAdapter)
        adapter.env = env
        adapter.device = torch.device("cpu")
        adapter._world_gravity = np.broadcast_to([0.0, 0.0, -1.0], (3, 3))
        physical = adapter.observation()
        compact = project_official_compact_observation(
            physical,
            preprocessing={
                "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
                "noise_enabled": False,
                "term_scale": {
                    "base_ang_vel": 0.25, "dof_pos": 1.0,
                    "dof_vel": 0.05, "projected_gravity": 1.0,
                },
            },
            default_joint_pos=torch.zeros(29),
            source_indices=torch.arange(29),
        )
        np.testing.assert_allclose(compact[:, :3].numpy(), qvel[:, 3:6] * 0.25, rtol=2e-6, atol=2e-6)

        measurements, valid = _base_measurements(env)
        assert valid.all()
        world_columns = [PHYSICAL_COLUMNS.index(f"w{axis}_world_rad_s") for axis in "xyz"]
        body_columns = [PHYSICAL_COLUMNS.index(f"w{axis}_body_rad_s") for axis in "xyz"]
        np.testing.assert_allclose(measurements[:, world_columns], expected_world, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(measurements[:, body_columns], qvel[:, 3:6], rtol=2e-6, atol=2e-6)

        footstand = cast(Any, object.__new__(Go2FootStandTask))
        footstand._backend = backend
        context = RewardContext(
            info={}, linvel=qvel[:, :3], gyro=qvel[:, 3:6],
            dof_pos=np.zeros((3, 12)), dof_vel=np.zeros((3, 12)),
        )
        np.testing.assert_allclose(
            footstand._cost_stay_still(context),
            np.sum(qvel[:, :2] ** 2, axis=1) + qvel[:, 5] ** 2,
            rtol=2e-6, atol=2e-6,
        )

        snapshot = backend.capture_rollout_state()
        changed_qpos = qpos[::-1].copy()
        changed_qvel = -qvel
        backend.set_state(rows, changed_qpos, changed_qvel)
        np.testing.assert_allclose(
            backend.get_base_ang_vel(),
            _world_angular_velocity(backend.model, changed_qpos, changed_qvel),
            rtol=2e-6, atol=2e-6,
        )
        backend.restore_rollout_state(snapshot)
        np.testing.assert_allclose(backend.get_base_ang_vel(), expected_world, rtol=2e-6, atol=2e-6)
        backend.reset_physics_rows_to_default(rows)
        np.testing.assert_allclose(backend.get_base_ang_vel(), 0.0, atol=2e-6)
        backend.restore_rollout_state(snapshot)
        np.testing.assert_allclose(backend.get_base_ang_vel(), expected_world, rtol=2e-6, atol=2e-6)
    finally:
        backend.close()


def test_fixed_root_angular_velocity_stays_zero(tmp_path):
    model_path = tmp_path / "fixed_root.xml"
    model_path.write_text(
        '<mujoco><worldbody><body name="root" euler="0.2 0.3 0.4">'
        '<geom type="box" size="0.1 0.1 0.1" mass="1"/>'
        '</body></worldbody></mujoco>',
    )
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), 1, 0.01, base_name="root", num_threads=1)
    try:
        backend.materialize()
        np.testing.assert_allclose(backend.get_base_ang_vel(), [[0.0, 0.0, 0.0]], atol=0.0)
    finally:
        backend.close()
