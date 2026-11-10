"""Source-faithful observations and state over the official MuJoCo implementation.

The simulator and stepping algorithm remain Holosoma's; this module owns only
source assets, state bindings, resets and physics snapshot transfer.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from holosoma.config_types.simulator import MujocoBackend
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.simulator.mujoco.mujoco import MuJoCo
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager

from .features import actor_observation, pack_privileged


class OracleSceneManager(MujocoSceneManager):
    """The supplied scene, including its floor and sensors, owns the world."""

    def add_robot(self, terrain_state, robot_config, xml_filter=None, prefix=""):
        scene = Path(robot_config.asset.asset_root) / robot_config.asset.xml_file
        self.robot_model_path = str(scene)
        self.robot_config = robot_config
        self.world_spec = mujoco.MjSpec.from_file(str(scene))
        self.world_spec.option.timestep = self.sim_dt
        self.robot_prefix = ""

    def __init__(self, simulator_config):
        super().__init__(simulator_config)
        self.sim_dt = 1.0 / simulator_config.sim.fps


class OracleMuJoCo(MuJoCo):
    scene_manager_class = OracleSceneManager

    def __init__(self, tyro_config, terrain_manager, device):
        super().__init__(tyro_config, terrain_manager, device)
        self._source_forward_graph = None
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            import warp as wp

            wp.init()
            warp_device = wp.get_device(device)
            stream = wp.stream_to_torch(warp_device.stream)
            stream.wait_stream(torch.cuda.current_stream(device))
            torch.cuda.set_stream(stream)
            self._source_torch_stream = stream

    def _setup_scene(self):
        # Do not let the generic scene builder add a second floor or rewrite the
        # source collision masks. All geometry already belongs to the supplied XML.
        terrain = self.terrain_manager.get_state("locomotion_terrain")
        self.scene_manager.add_robot(terrain, self.robot_config)

    def prepare_manager_fields(self, **managers):
        from holosoma.simulator.mujoco.fields import collect_required_fields, prepare_fields

        required = set(collect_required_fields(*managers.values()))
        required.update(("body_mass", "geom_friction"))
        prepare_fields(self, sorted(required))
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            import mujoco_warp as mjw
            import warp as wp

            # Field expansion replaces Warp allocations. The graph captured by
            # the original constructor still points at the pre-expansion arrays.
            with wp.ScopedDevice(self.backend.mjw_device):
                with wp.ScopedCapture() as capture:
                    mjw.step(self.backend.mjw_model, self.backend.mjw_data)
                self.backend.step_graph = capture.graph
                with wp.ScopedCapture() as capture:
                    mjw.forward(self.backend.mjw_model, self.backend.mjw_data)
                self._source_forward_graph = capture.graph

    def forward_source_state(self):
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            import warp as wp

            if self._source_forward_graph is None:
                raise RuntimeError("Source forward graph must be prepared before state synchronization")
            with wp.ScopedDevice(self.backend.mjw_device):
                wp.capture_launch(self._source_forward_graph)
        else:
            mujoco.mj_forward(self.root_model, self.root_data)

    def set_dof_state_tensor(self, env_ids=None, dof_states=None):
        if env_ids is not None and len(env_ids) == 0:
            return
        super().set_dof_state_tensor(env_ids, dof_states)
        # Reset and snapshot restore write root state before DOFs. Publish the
        # resulting derived state before the official refresh/observation path.
        # ClassicBackend already calls mj_forward in its setter.
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            self.forward_source_state()


class PhysicsViews:
    """Bind backend-specific storage once; hot observations never inspect XML."""

    def __init__(self, simulator, metadata):
        self.simulator = simulator
        self.model = simulator.root_model
        self.data = simulator.root_data
        self.device = torch.device(simulator.device)
        self.num_envs = simulator.num_envs
        self.warp = simulator.simulator_config.mujoco_backend == MujocoBackend.WARP
        model = self.model
        if [model.body(i).name for i in range(model.nbody)] != metadata["body_names"]:
            raise ValueError("Source scene body order differs from the Oracle checkpoint")
        if simulator.dof_names != metadata["actuated_joint_names"]:
            raise ValueError("Source scene joint order differs from the Oracle checkpoint")
        self.base_id = model.body("pelvis").id
        self.floor_id = model.geom("floor").id
        self.floor_z = float(model.geom_pos[self.floor_id, 2])
        self.dof_ids = torch.tensor(simulator.dof_qvel_addrs, device=self.device)
        self.sensors = {}
        names = metadata["environment_config"]["sensor"]
        for name in (*names.values(), "left_foot_net_contact", "right_foot_net_contact"):
            sensor = model.sensor(name)
            address = int(sensor.adr[0])
            width = int(sensor.dim[0])
            self.sensors[name] = slice(address, address + width)
        if self.warp:
            import warp as wp

            backend = simulator.backend
            self.sensor_data = wp.to_torch(backend.mjw_data.sensordata)
            self.torques = wp.to_torch(backend.mjw_data.qfrc_actuator)
            self.xpos = wp.to_torch(backend.mjw_data.xpos)
            self.mass = wp.to_torch(backend.mjw_model.body_mass)
            self.friction = wp.to_torch(backend.mjw_model.geom_friction)
            self.com = wp.to_torch(backend.mjw_model.body_ipos)
            self.warmstart = wp.to_torch(backend.mjw_data.qacc_warmstart)
            self.clock = wp.to_torch(backend.mjw_data.time)
        else:
            self.sensor_data = torch.from_numpy(self.data.sensordata).unsqueeze(0)
            self.torques = torch.from_numpy(self.data.qfrc_actuator).unsqueeze(0)
            self.xpos = torch.from_numpy(self.data.xpos).unsqueeze(0)
            self.mass = torch.from_numpy(model.body_mass).unsqueeze(0)
            self.friction = torch.from_numpy(model.geom_friction).unsqueeze(0)
            self.com = torch.from_numpy(model.body_ipos).unsqueeze(0)
            self.warmstart = torch.from_numpy(self.data.qacc_warmstart).unsqueeze(0)
            self.clock = None
            # mj_setConst uses qpos0 and mutates its MjData scratch argument.
            # Never pass the active state that reset/snapshot restore just wrote.
            self._constant_data = mujoco.MjData(model)
        # Native MuJoCo fields are float64; expanded Warp fields are float32.
        # Capture independent nominal values in the owning buffer's format once.
        self.nominal_mass = torch.tensor(model.body_mass, dtype=self.mass.dtype, device=self.mass.device)
        self.nominal_friction = torch.tensor(model.geom_friction, dtype=self.friction.dtype, device=self.friction.device)
        self.nominal_com = torch.tensor(model.body_ipos, dtype=self.com.dtype, device=self.com.device)

    def sensor(self, name):
        return self.sensor_data[:, self.sensors[name]].to(dtype=torch.float32)

    def randomize(self, env_ids, settings):
        count = env_ids.numel()
        if count == 0:
            return
        # Start from the captured nominal model, never the previous episode's DR.
        self.mass[env_ids] = self.nominal_mass
        self.friction[env_ids] = self.nominal_friction
        if settings["randomize_base_mass"]:
            low, high = settings["added_mass_range"]
            delta = torch.empty(count, device=self.device).uniform_(low, high)
            self.mass[env_ids, self.base_id] += delta
        if settings["randomize_ground_friction"]:
            low, high = settings["ground_friction_multiplier_range"]
            factor = torch.empty(count, device=self.device).uniform_(low, high)
            self.friction[env_ids, self.floor_id, 0] *= factor
        if not self.warp:
            mujoco.mj_setConst(self.model, self._constant_data)

    def privileged(self, env):
        metadata = env.oracle_metadata
        settings = metadata["environment_config"]["domain_rand"]
        batch, device = self.num_envs, self.device
        left, right = self.sensor("left_foot_net_contact"), self.sensor("right_foot_net_contact")
        kp, kd = env.action_manager.get_term("joint_control").get_pd_scale_tensors()
        nominal_mass = self.nominal_mass.to(dtype=torch.float32)
        mass = self.mass.to(dtype=torch.float32).expand(batch, -1)
        safe_mass = torch.where(nominal_mass == 0, 1, nominal_mass)
        mass_scale = torch.where(nominal_mass == 0, 1, mass / safe_mass)
        limits = env.torque_limits.clamp_min(1e-6)
        fields = {
            "base_linear_velocity": self.sensor(metadata["environment_config"]["sensor"]["local_linvel"]),
            "foot_contact_resultants": torch.cat((left[:, 1:4], right[:, 1:4]), dim=1),
            "foot_contact_flags": torch.cat((left[:, :1] > 0, right[:, :1] > 0), dim=1).float(),
            # The accepted source schema itself defines these nine flat-terrain values as zero.
            "terrain_heights": torch.zeros((batch, 9), device=device),
            "root_clearance": self.xpos[:, self.base_id, 2:3].float() - self.floor_z,
            "kp_scale": kp,
            "kd_scale": kd,
            "normalized_torque": self.torques[:, self.dof_ids].float() / limits,
            "ground_friction": self.friction[:, self.floor_id, 0:1].float().expand(batch, -1),
            "base_com_shift": (self.com[:, self.base_id] - self.nominal_com[self.base_id]).float().expand(batch, -1),
            "added_base_mass": (mass[:, self.base_id:self.base_id + 1] - nominal_mass[self.base_id]),
            "body_mass_scale": mass_scale,
            # These modes are rejected when enabled by config.py, not silently approximated.
            "dof_position_bias": torch.zeros((batch, 29), device=device),
            "torque_rfi": torch.zeros((batch, 29), device=device),
            "control_delay": torch.zeros((batch, 1), device=device),
            "push_interval": torch.full((batch, 1), float(settings["fada_push_interval_seconds"]), device=device),
            "push_velocity": torch.full((batch, 1), float(settings["fada_max_push_velocity"]), device=device),
        }
        return pack_privileged(fields, metadata)

    def capture(self):
        return {
            "body_mass": self.mass.detach().clone(),
            "geom_friction": self.friction.detach().clone(),
            "qacc_warmstart": self.warmstart.detach().clone(),
            "time": self.clock.detach().clone() if self.warp else torch.tensor([self.data.time], dtype=torch.float64),
        }

    def restore(self, state):
        for name, target in (("body_mass", self.mass), ("geom_friction", self.friction),
                             ("qacc_warmstart", self.warmstart)):
            value = state[name]
            if value.shape != target.shape:
                raise ValueError(f"Source physics snapshot shape differs for {name}")
            target.copy_(value)
        if self.warp:
            self.clock.copy_(state["time"])
        else:
            self.data.time = float(state["time"][0])
            mujoco.mj_setConst(self.model, self._constant_data)


class OracleEnvironment(LeggedRobotLocomotionManager):
    def __init__(self, tyro_config, *, device, metadata):
        self.oracle_metadata = metadata
        self._source_views = None
        self._closed = False
        super().__init__(tyro_config, device=device)
        self.source_views()

    def source_views(self):
        if self._source_views is None:
            self._source_views = PhysicsViews(self.simulator, self.oracle_metadata)
        return self._source_views

    def oracle_observation(self):
        sensor_names = self.oracle_metadata["environment_config"]["sensor"]
        gait = self.command_manager.get_state("locomotion_gait")
        return actor_observation(
            gyro=self.source_views().sensor(sensor_names["gyro"]),
            upvector=self.source_views().sensor(sensor_names["upvector"]),
            dof_pos=self.simulator.dof_pos[:],
            dof_vel=self.simulator.dof_vel[:],
            default_joint_pos=self.default_dof_pos,
            last_actions=self.action_manager.action,
            commands=self.command_manager.commands[:, :3],
            gait_phase=gait.phase,
            noise_config=self.oracle_metadata["environment_config"]["noise_config"],
        )

    def _reset_dofs(self, env_ids, target_state=None):
        if target_state is not None:
            return super()._reset_dofs(env_ids, target_state)
        self.simulator.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.simulator.dof_vel[env_ids] = 0

    def _reset_root_states(self, env_ids, target_root_states=None):
        if target_root_states is not None:
            return super()._reset_root_states(env_ids, target_root_states)
        count = env_ids.numel()
        state = self.base_init_state.expand(count, -1).clone()
        settings = self.oracle_metadata["environment_config"]
        if settings["domain_rand"]["randomize_reset_pose"]:
            state[:, :2] += torch.empty((count, 2), device=self.device).uniform_(-0.5, 0.5)
            yaw = torch.empty(count, device=self.device).uniform_(-torch.pi, torch.pi)
            state[:, 3:5] = 0
            state[:, 5] = torch.sin(yaw / 2)
            state[:, 6] = torch.cos(yaw / 2)
        state[:, 7:13] = torch.empty((count, 6), device=self.device).uniform_(
            -float(settings["reset_base_qvel_limit"]), float(settings["reset_base_qvel_limit"]),
        )
        standing = (self.command_manager.commands[env_ids, :3] == 0).all(dim=1)
        stand_limit = float(settings["standing_reset_base_qvel_limit"])
        state[standing, 7:13] = torch.empty((int(standing.sum()), 6), device=self.device).uniform_(
            -stand_limit, stand_limit,
        )
        self.simulator.robot_root_states[env_ids] = state
        self.need_to_refresh_envs[env_ids] = True

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(env_ids)
        self._refresh_envs_after_reset(env_ids)
        self._compute_observations()
        return self.obs_buf_dict

    def _refresh_envs_after_reset(self, env_ids):
        if env_ids.numel():
            super()._refresh_envs_after_reset(env_ids)

    def capture_fada_physics_state(self):
        return self.source_views().capture()

    def restore_fada_physics_state(self, state):
        self.source_views().restore(state)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.simulator.viewer is not None:
            self.simulator.viewer.close()
        self._source_views = None
        # Retire the entire official simulator owner, including its Warp graphs/views.
        self.simulator = None
