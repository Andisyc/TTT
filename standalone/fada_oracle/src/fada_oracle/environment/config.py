"""Map checkpoint controls and the compiled source model onto official FADA config."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.experiment import ExperimentConfig

_TERMS = "fada_oracle.environment.terms:"
_RANDOMIZATION = "holosoma.managers.randomization.terms.locomotion:"


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _vector(metadata: dict, name: str, width: int, *, positive: bool = False) -> list[float]:
    values = metadata.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise ValueError(f"{name} must have {width} entries in actuator order")
    return [_number(value, name, positive=positive) for value in values]


def _range(value, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range")
    result = [_number(item, name, positive=positive) for item in value]
    if result[0] > result[1]:
        raise ValueError(f"{name} range is reversed")
    return result


def _validate_source(metadata: dict) -> tuple[dict, dict, list[str], list[float], list[float], list[float]]:
    source = metadata.get("environment_config")
    if not isinstance(source, dict) or not source:
        raise ValueError("environment_config must contain the source environment configuration")
    joints = metadata.get("actuated_joint_names")
    if (
        not isinstance(joints, (list, tuple)) or len(joints) != 29
        or any(not isinstance(name, str) or not name for name in joints)
        or len(set(joints)) != 29
    ):
        raise ValueError("actuated_joint_names must contain 29 unique ordered joint names")
    nominal = _vector(metadata, "default_joint_pos", 29)
    kp = _vector(metadata, "kp", 29, positive=True)
    kd = _vector(metadata, "kd", 29, positive=True)
    for name in ("sim_dt", "ctrl_dt"):
        value = _number(metadata.get(name), name, positive=True)
        if not math.isclose(value, _number(source.get(name), f"environment_config.{name}", positive=True), rel_tol=1e-12):
            raise ValueError(f"{name} differs between deployment metadata and environment_config")
    scale = _number(metadata.get("action_scale"), "action_scale", positive=True)
    if scale != _number(source.get("control_config", {}).get("action_scale"), "control_config.action_scale", positive=True):
        raise ValueError("action_scale differs between metadata and source control_config")
    dr = source.get("domain_rand")
    if not isinstance(dr, dict):
        raise ValueError("environment_config.domain_rand is required")
    supported = {"randomize_reset_pose", "randomize_base_mass", "randomize_ground_friction", "randomize_kp", "randomize_kd"}
    for name, value in dr.items():
        if (name.startswith("randomize_") or name in {"random_com", "push_robots"}) and name not in supported and value:
            raise ValueError(f"Unsupported enabled source domain randomization: {name}")
    if dr.get("torque_rfi_fraction", 0) != 0:
        raise ValueError("Unsupported enabled source domain randomization: torque_rfi_fraction")
    strength = dr.get("actuator_strength", {})
    if any(strength.get(name, False) for name in ("enabled", "curriculum_enabled", "group_curriculum_enabled")):
        raise ValueError("Unsupported enabled source actuator-strength randomization")
    if source.get("control_config", {}).get("simulate_action_latency", False):
        raise ValueError("Unsupported source simulate_action_latency")
    if source.get("scene", {}).get("fragment_files") or source.get("scene", {}).get("terrain") is not None:
        raise ValueError("Only the complete source flat scene is supported")
    return source, dr, list(joints), nominal, kp, kd


def build_experiment_config(
    metadata: dict, scene_file: Path, *, backend: str, device: str,
    num_envs: int, seed: int, run_dir: Path,
) -> ExperimentConfig:
    """Build official frozen configuration without creating or stepping a runtime.

    The caller first loads the pinned official checkout with ``import_upstream``.
    Metadata belongs to the runtime explicitly, not to extra frozen config fields.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if backend == "classic":
        if device != "cpu" or num_envs != 1:
            raise ValueError("Official classic MuJoCo requires device='cpu' and num_envs=1")
    elif backend == "warp":
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("Official Warp requires a CUDA device; there is no CPU fallback")
    else:
        raise ValueError("backend must be 'classic' or 'warp'")
    source, dr, joints, nominal, kp, kd = _validate_source(metadata)
    sim_dt, ctrl_dt = float(metadata["sim_dt"]), float(metadata["ctrl_dt"])
    fps, decimation = round(1.0 / sim_dt), round(ctrl_dt / sim_dt)
    if fps <= 0 or decimation <= 0 or not math.isclose(1.0 / fps, sim_dt, rel_tol=1e-12) or not math.isclose(decimation * sim_dt, ctrl_dt, rel_tol=1e-12):
        raise ValueError("Source timing must map exactly to official integer fps/control_decimation")
    episode_seconds = _number(source.get("max_episode_seconds"), "max_episode_seconds", positive=True)
    reward_source = source.get("reward_config", {})
    max_tilt = _number(reward_source.get("max_tilt_deg"), "max_tilt_deg", positive=True)
    min_height = _number(reward_source.get("min_base_height"), "min_base_height", positive=True)
    frequency = _number(reward_source.get("gait_frequency"), "gait_frequency", positive=True)
    commands = source.get("commands", {})
    velocities = commands.get("vel_limit")
    if not isinstance(velocities, (list, tuple)) or len(velocities) != 2 or any(len(row) != 3 for row in velocities):
        raise ValueError("commands.vel_limit must contain lower/upper three-dimensional bounds")
    ranges = {name: _range([velocities[0][i], velocities[1][i]], f"commands.{name}") for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw"))}
    resampling = _number(commands.get("resampling_time"), "commands.resampling_time", positive=True)
    standing = _number(commands.get("rel_standing_envs"), "commands.rel_standing_envs")
    if not 0 <= standing <= 1:
        raise ValueError("commands.rel_standing_envs must be in [0, 1]")
    if commands.get("heading_command", False):
        raise ValueError("Source heading-command mode is not supported")
    for flag in ("dead_zone_enabled", "random_height_during_walking", "observe_height_command"):
        if commands.get(flag, False):
            raise ValueError(f"Unsupported enabled source command mode: {flag}")
    if commands.get("rel_transition_envs", 0) != 0:
        raise ValueError("Source transition-command sampling is not supported")
    straight = _number(commands.get("rel_straight_envs", 0), "commands.rel_straight_envs")
    small_xy = _number(commands.get("small_xy_threshold", 0), "commands.small_xy_threshold")
    if not 0 <= straight <= 1 - standing or small_xy < 0:
        raise ValueError("Source command probabilities or small-XY threshold are invalid")
    if straight and max(ranges["lin_vel_x"][0], small_xy) >= ranges["lin_vel_x"][1]:
        raise ValueError("Straight commands require a positive forward sampling interval")

    import mujoco
    from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
    from holosoma.config_types.curriculum import CurriculumManagerCfg
    from holosoma.config_types.logger import DisabledLoggerConfig
    from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg
    from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
    from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
    from holosoma.config_values import simulator, terrain
    from holosoma.config_values.loco.g1.experiment import g1_29dof_oracle

    scene_file = Path(scene_file).resolve(strict=True)
    model = mujoco.MjModel.from_xml_path(str(scene_file))
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)]
    if bodies != list(metadata.get("body_names", [])):
        raise ValueError("Compiled source body order differs from checkpoint metadata")
    joint_ids = model.actuator_trnid[:, 0].tolist()
    actual_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in joint_ids]
    if actual_joints != joints:
        raise ValueError("Compiled source actuator order differs from checkpoint metadata")
    if any(not model.jnt_limited[index] for index in joint_ids):
        raise ValueError("Every source actuated joint must have finite position limits")
    if not model.actuator_forcelimited.all():
        raise ValueError("Every source actuator must retain its force limit")
    force_ranges = model.actuator_forcerange.tolist()
    if any(lo >= 0 or hi <= 0 or not math.isclose(-lo, hi) for lo, hi in force_ranges):
        raise ValueError("Official symmetric torque clipping requires symmetric source force limits")
    if any(model.actuator_biastype) or any(model.actuator_dyntype) or any(model.actuator_gainprm[:, 0] != 1):
        raise ValueError("Prepared scene must contain direct torque motors, not position actuators")
    dof_ids = [int(model.jnt_dofadr[index]) for index in joint_ids]
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand < 0:
        raise ValueError("Source model must contain its original 'stand' keyframe")
    root_qpos = model.key_qpos[stand, :7].tolist()
    base = g1_29dof_oracle
    robot = replace(
        base.robot, num_bodies=model.nbody - 1, body_names=bodies[1:],
        policy_obs_dim=98, critic_obs_dim=303,
        algo_obs_dim_dict={"actor_obs": 98, "critic_obs": 303, "oracle_obs": 98, "oracle_priv": 205},
        dof_names=joints, actions_dim=29, dof_obs_size=29,
        key_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
        foot_body_name="ankle_roll_link", foot_height_name="ankle_roll_link",
        dof_pos_lower_limit_list=[float(model.jnt_range[index, 0]) for index in joint_ids],
        dof_pos_upper_limit_list=[float(model.jnt_range[index, 1]) for index in joint_ids],
        dof_effort_limit_list=[float(hi) for _, hi in force_ranges],
        dof_armature_list=[float(model.dof_armature[index]) for index in dof_ids],
        dof_joint_friction_list=[float(model.dof_frictionloss[index]) for index in dof_ids],
        init_state=replace(base.robot.init_state, pos=root_qpos[:3], rot=root_qpos[4:7] + root_qpos[3:4], lin_vel=[0.0]*3, ang_vel=[0.0]*3, default_joint_angles=dict(zip(joints, nominal, strict=True))),
        control=replace(base.robot.control, control_type="P", stiffness=dict(zip(joints, kp, strict=True)), damping=dict(zip(joints, kd, strict=True)), action_scale=float(metadata["action_scale"]), clip_actions=False, clip_torques=True, action_scales_by_effort_limit_over_p_gain=False),
        asset=replace(base.robot.asset, asset_root=str(scene_file.parent), xml_file=scene_file.name, mesh_root=str(scene_file.parent / "assets")),
        randomize_link_body_names=["pelvis"],
    )
    simulator_base = simulator.mujoco if backend == "classic" else simulator.mjwarp
    simulator_cfg = replace(simulator_base, _target_="fada_oracle.environment.runtime.OracleMuJoCo", config=replace(simulator_base.config, sim=replace(simulator_base.config.sim, fps=fps, control_decimation=decimation, max_episode_length_s=episode_seconds), robot_mjcf_filter=replace(simulator_base.config.robot_mjcf_filter, enable=False), bridge=replace(simulator_base.config.bridge, enabled=False), policy_sync_zmq_port=-1))
    observation = ObservationManagerCfg(groups={
        "oracle_obs": ObsGroupCfg(terms={"oracle_obs": ObsTermCfg(func=_TERMS + "oracle_obs")}),
        "oracle_priv": ObsGroupCfg(terms={"oracle_priv": ObsTermCfg(func=_TERMS + "oracle_priv")}),
    }, clip_observations=float("inf"))
    command_setup = {
        "locomotion_gait": CommandTermCfg(func=_TERMS + "SourceGait", params={"gait_frequency": frequency}),
        "locomotion_command": CommandTermCfg(func=_TERMS + "SourceCommand", params={"command_ranges": ranges, "stand_prob": standing}),
    }
    command_hooks = {name: CommandTermCfg(func=term.func) for name, term in command_setup.items()}
    command = CommandManagerCfg(params={"locomotion_command_resampling_time": resampling}, setup_terms=command_setup, reset_terms=command_hooks, step_terms=command_hooks)
    gains = {name: _range(dr.get(f"{name}_multiplier_range"), f"{name}_multiplier_range", positive=True) if dr.get(f"randomize_{name}", False) else [1.0, 1.0] for name in ("kp", "kd")}
    if dr.get("randomize_base_mass", False):
        _range(dr.get("added_mass_range"), "added_mass_range")
    if dr.get("randomize_ground_friction", False):
        _range(dr.get("ground_friction_multiplier_range"), "ground_friction_multiplier_range", positive=True)
    physics = RandomizationTermCfg(func=_TERMS + "source_physics")
    randomization = RandomizationManagerCfg(setup_terms={
        "action_delay": RandomizationTermCfg(func=_RANDOMIZATION + "setup_action_delay_buffers", params={"ctrl_delay_step_range": [0, 0], "enabled": False}),
        "source_physics": physics,
        "actuator_randomizer_state": RandomizationTermCfg(func=_RANDOMIZATION + "ActuatorRandomizerState", params={"kp_range": gains["kp"], "kd_range": gains["kd"], "enable_pd_gain": bool(dr.get("randomize_kp") or dr.get("randomize_kd")), "enable_rfi_lim": False}),
    }, reset_terms={
        "source_physics": physics,
        "actuator_randomizer_state": RandomizationTermCfg(func=_RANDOMIZATION + "ActuatorRandomizerState"),
        "randomize_dof_state": RandomizationTermCfg(func=_RANDOMIZATION + "randomize_dof_state", params={"joint_pos_scale_range": [1.0, 1.0], "joint_pos_bias_range": [0.0, 0.0], "joint_vel_range": [0.0, 0.0], "randomize_dof_pos_bias": False}),
    })
    termination = TerminationManagerCfg(terms={
        "source_failure": TerminationTermCfg(func=_TERMS + "source_failure", params={"max_tilt_deg": max_tilt, "min_base_height": min_height}),
        "timeout": base.termination.terms["timeout"],
    })
    curriculum = CurriculumManagerCfg(params=dict(base.curriculum.params), setup_terms={"average_episode_tracker": base.curriculum.setup_terms["average_episode_tracker"]})
    return replace(base, env_class="fada_oracle.environment.runtime.OracleEnvironment", training=replace(base.training, num_envs=num_envs, seed=seed, headless=True, export_onnx=False), simulator=simulator_cfg, robot=robot, terrain=terrain.terrain_locomotion_plane, observation=observation, command=command, randomization=randomization, termination=termination, curriculum=curriculum, logger=DisabledLoggerConfig(base_dir=str(Path(run_dir).resolve())))
