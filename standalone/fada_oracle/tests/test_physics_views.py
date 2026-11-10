"""Exercise native-double and Warp-float randomization without physics stepping."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp


@pytest.fixture(scope="module")
def physics_owner():
    project = Path(__file__).resolve().parents[1]
    if not (project / ".deps/FADA/.git").is_dir():
        pytest.skip("Run fada-oracle setup to prepare the pinned official source")
    from fada_oracle.upstream import import_upstream
    import_upstream(project)
    from holosoma.config_types.simulator import MujocoBackend
    from holosoma.utils.simulator_config import SimulatorType, set_simulator_type_enum
    set_simulator_type_enum(SimulatorType.MUJOCO)
    from fada_oracle.environment.runtime import PhysicsViews
    return PhysicsViews, MujocoBackend


@pytest.mark.parametrize("warp_storage", [False, True], ids=["native-float64", "warp-float32"])
def test_randomization_preserves_backend_precision_and_resets_nominal_values(physics_owner, warp_storage):
    PhysicsViews, MujocoBackend = physics_owner
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 0.1" friction="0.8 0.01 0.001"/>
        <body name="pelvis" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="4" pos="0.01 0.02 0.03"/>
          <body name="left_foot" pos="0.2 0 -0.3">
            <joint name="left"/>
            <geom type="sphere" size="0.05" mass="1"/>
          </body>
          <body name="right_foot" pos="-0.2 0 -0.3">
            <joint name="right"/>
            <geom type="sphere" size="0.05" mass="2"/>
          </body>
        </body>
      </worldbody>
      <sensor>
        <contact name="left_foot_net_contact" geom1="floor" body2="left_foot" data="found force" num="1" reduce="netforce"/>
        <contact name="right_foot_net_contact" geom1="floor" body2="right_foot" data="found force" num="1" reduce="netforce"/>
      </sensor>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    count = 4 if warp_storage else 1
    dtype = torch.float32 if warp_storage else torch.float64
    backend = MujocoBackend.WARP if warp_storage else MujocoBackend.CLASSIC
    simulator = SimpleNamespace(
        root_model=model, root_data=data, device="cpu", num_envs=count,
        simulator_config=SimpleNamespace(mujoco_backend=backend),
        dof_names=["left", "right"], dof_qvel_addrs=[6, 7],
    )
    nominal_mass = torch.tensor(model.body_mass.copy(), dtype=dtype)
    nominal_friction = torch.tensor(model.geom_friction.copy(), dtype=dtype)
    nominal_com = torch.tensor(model.body_ipos.copy(), dtype=dtype)
    if warp_storage:
        # Real Warp allocations on CPU reproduce CUDA's float32 storage contract;
        # the carrier exposes only the fields consumed by the real owner.
        def worlds(value):
            return wp.array(np.repeat(np.asarray(value)[None], count, axis=0), dtype=wp.float32, device="cpu")
        simulator.backend = SimpleNamespace(
            mjw_model=SimpleNamespace(body_mass=worlds(model.body_mass), geom_friction=worlds(model.geom_friction), body_ipos=worlds(model.body_ipos)),
            mjw_data=SimpleNamespace(sensordata=worlds(data.sensordata), qfrc_actuator=worlds(data.qfrc_actuator), xpos=worlds(data.xpos), qacc_warmstart=worlds(data.qacc_warmstart), time=wp.zeros(count, dtype=wp.float32, device="cpu")),
        )
    metadata = {
        "body_names": ["world", "pelvis", "left_foot", "right_foot"],
        "actuated_joint_names": ["left", "right"],
        "environment_config": {"sensor": {}},
    }
    views = PhysicsViews(simulator, metadata)
    data.qpos[2] = 0.754
    data.qvel[:] = 0.125
    data.qacc_warmstart[:] = 0.25
    data.time = 3.25
    native_qpos, native_qvel = data.qpos.copy(), data.qvel.copy()
    selected = torch.tensor([1, 3] if warp_storage else [0])
    expected_mass = nominal_mass.repeat(count, 1)
    expected_friction = nominal_friction.repeat(count, 1, 1)
    expected_mass[selected, 1] += 0.25
    expected_friction[selected, views.floor_id, 0] *= 1.5
    settings = {"randomize_base_mass": True, "added_mass_range": [0.25, 0.25], "randomize_ground_friction": True, "ground_friction_multiplier_range": [1.5, 1.5]}
    with torch.random.fork_rng(devices=[]):
        for _ in range(2):
            views.randomize(selected, settings)
            torch.testing.assert_close(views.mass, expected_mass, rtol=0, atol=0)
            torch.testing.assert_close(views.friction, expected_friction, rtol=0, atol=0)
            np.testing.assert_array_equal(data.qpos, native_qpos)
            np.testing.assert_array_equal(data.qvel, native_qvel)
            np.testing.assert_array_equal(data.qacc_warmstart, np.full(model.nv, 0.25))
            assert data.time == 3.25
    torch.testing.assert_close(views.nominal_mass, nominal_mass, rtol=0, atol=0)
    torch.testing.assert_close(views.nominal_friction, nominal_friction, rtol=0, atol=0)
    torch.testing.assert_close(views.nominal_com, nominal_com, rtol=0, atol=0)
    views.randomize(selected, {"randomize_base_mass": False, "randomize_ground_friction": False})
    torch.testing.assert_close(views.mass, nominal_mass.repeat(count, 1), rtol=0, atol=0)
    torch.testing.assert_close(views.friction, nominal_friction.repeat(count, 1, 1), rtol=0, atol=0)


def test_reset_preserves_mujoco_local_angular_velocity_at_nonzero_yaw(physics_owner, monkeypatch):
    from fada_oracle.environment.runtime import OracleEnvironment
    from holosoma.simulator.mujoco.tensor_views import MujocoRootStateView

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom type="sphere" size="0.1" mass="4"/><site name="imu"/>
    </body></worldbody><sensor><gyro name="gyro" site="imu"/></sensor></mujoco>
    """)
    data = mujoco.MjData(model)
    root = MujocoRootStateView(
        data.qpos, data.qvel, slice(0, 3), slice(3, 7), slice(0, 3), slice(3, 6), 1,
    )
    env = object.__new__(OracleEnvironment)
    env.device = "cpu"
    env.base_init_state = torch.tensor([0, 0, 0.754, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0])
    env.oracle_metadata = {"environment_config": {
        "domain_rand": {"randomize_reset_pose": True},
        "reset_base_qvel_limit": 0.5, "standing_reset_base_qvel_limit": 0.0,
    }}
    env.command_manager = SimpleNamespace(commands=torch.ones(1, 3))
    env.simulator = SimpleNamespace(robot_root_states=root)
    env.need_to_refresh_envs = torch.zeros(1, dtype=torch.bool)
    velocity = torch.tensor([[0.1, -0.2, 0.3, -0.3, 0.4, -0.1]])

    def source_draw(target, low=0, high=1, *, generator=None):
        if target.numel() == 0:
            return target
        if target.shape == (1, 6):
            return target.copy_(velocity)
        return target.fill_(torch.pi / 2 if low == -torch.pi else 0)

    monkeypatch.setattr(torch.Tensor, "uniform_", source_draw)
    env._reset_root_states(torch.tensor([0]))
    mujoco.mj_forward(model, data)
    torch.testing.assert_close(torch.from_numpy(data.qvel).float(), velocity[0], rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(data.sensordata).float(), velocity[0, 3:], rtol=1e-6, atol=1e-6)
