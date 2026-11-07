"""One nominal Oracle rollout: stand, straight, left and right in four env rows.

CSV rows contain actions applied over [(step-1)*dt, step*dt] and the resulting
physical state at step*dt. Each case stops recording at its first termination;
no reset episodes or padded samples are joined. These are acceptance diagnostics,
not an OAT protocol, Etilt definition, or an automatic policy-quality verdict.
All rows start at the task keyframe with a shared nominal gait period and phase
pair [0, pi]. Policy command/phase columns are PRE-step raw actor inputs;
physical columns are POST-step measurements. Protocol v2 fixes angular frames.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from unilab.algos.torch.distill.collection.transition import set_transition_input_rows
from unilab.envs.common.rotation import np_quat_apply_inverse
from unilab.training import ensure_registries
from unilab.training.oracle.checkpoint import read_oracle_checkpoint
from unilab.training.oracle.playback import build_oracle_play_environment
from unilab.training.oracle.teacher import MotionQualityOracleTeacher
from unilab.training.seed import apply_training_seed

CASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("straight", (0.6, 0.0, 0.0)),
    ("left", (0.6, 0.0, 0.3)),
    ("right", (0.6, 0.0, -0.3)),
)
PHYSICAL_COLUMNS = (
    "x_world_m", "y_world_m", "z_world_m", "quat_w", "quat_x", "quat_y", "quat_z",
    "vx_world_m_s", "vy_world_m_s", "vz_world_m_s",
    "wx_world_rad_s", "wy_world_rad_s", "wz_world_rad_s",
    "vx_body_m_s", "vy_body_m_s", "vz_body_m_s",
    "wx_body_rad_s", "wy_body_rad_s", "wz_body_rad_s",
    "roll_rad", "pitch_rad", "yaw_rad",
)


def _base_measurements(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read world-frame velocities and explicitly derive body-frame columns."""
    position = env.get_base_pos()
    quaternion = env.get_base_quat()
    linear_world = env.get_base_lin_vel()
    angular_world = env.get_base_ang_vel()
    physical = np.concatenate((position, quaternion, linear_world, angular_world), axis=1)
    valid = np.isfinite(physical).all(axis=1)
    values = np.full((env.num_envs, len(PHYSICAL_COLUMNS)), np.nan)
    if valid.any():
        quat = quaternion[valid]
        values[valid] = np.concatenate((
            position[valid], quat, linear_world[valid],
            angular_world[valid],
            np_quat_apply_inverse(quat, linear_world[valid]),
            np_quat_apply_inverse(quat, angular_world[valid]),
            Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("xyz"),
        ), axis=1)
    return values, valid


def evaluate_motion_quality_oracle(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    duration_s: float = 10.0,
    seed: int = 1,
    device: str | None = None,
    record_video: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint once; refuse to overwrite an existing result directory."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    apply_training_seed(seed)
    ensure_registries()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = read_oracle_checkpoint(checkpoint_path)
    commands = np.asarray([command for _, command in CASES], dtype=np.float32)
    limits = np.asarray(payload["metadata"]["environment_config"]["commands"]["vel_limit"])
    if np.any(commands < limits[0]) or np.any(commands > limits[1]):
        raise ValueError("acceptance commands exceed the checkpoint's training command range")
    # Nominal acceptance is independent of whether a saved play profile was enabled.
    patch = {
        "noise_config": {"level": 0.0},
        "runtime_payload": {"enabled": False},
        "reset_base_qvel_limit": 0.0, "standing_reset_base_qvel_limit": 0.0,
        "commands": {"rel_standing_envs": 0.0, "heading_command": False,
                     "resampling_time": max(duration_s + 1.0, 10.0)},
        "domain_rand": {"randomize_reset_pose": False, "randomize_kp": False,
                        "randomize_kd": False, "randomize_ground_friction": False,
                        "randomize_base_mass": False},
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = None
    rows: list[dict[str, Any]] = []
    snapshots: list[np.ndarray] = []
    outcomes: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path), "checkpoint_metadata": payload["metadata"],
        "acceptance_protocol": "nominal_fixed_command_v2",
        "seed": seed, "requested_duration_s": duration_s,
        "cases": [{"name": name, "command": list(command)} for name, command in CASES],
        "online_updates": False, "payload_enabled": False, "record_video": record_video,
        "pose_source": "public env.get_base_* (base, not the actor torso gyro)",
        "quaternion_order": "wxyz; base orientation in world",
        "angular_velocity_source": "public env.get_base_ang_vel in world frame; inverse-rotate once to body",
        "command_frame": "base/body vx, vy in m/s and yaw rate in rad/s",
        "row_timing": "action over [(step-1)*ctrl_dt, step*ctrl_dt]; post-step state",
        "termination_rule": "first terminated/truncated/invalid-physics row ends that case",
        "padding": False, "policy_quality_accepted": None,
        "video_case_order": [name for name, _ in CASES],
        "video_note": "terminated cases remain visible but have no further CSV samples",
    }
    action_columns = [f"action_{i}" for i in range(payload["metadata"]["network_config"]["action_dim"])]
    fieldnames = ["case", "step", "time_s", "wall_elapsed_s", "command_vx_m_s",
                  "command_vy_m_s", "command_wz_rad_s", "policy_command_vx_m_s",
                  "policy_command_vy_m_s", "policy_command_wz_rad_s",
                  "policy_gait_phase_left_rad", "policy_gait_phase_right_rad",
                  "policy_gait_period_s", "physical_valid", "terminated",
                  "truncated", *PHYSICAL_COLUMNS, *action_columns]
    active = np.ones(len(CASES), dtype=bool)
    execution_status, error = "finished", None
    step = 0
    ctrl_dt = 0.0
    observation_slices: dict[str, slice] = {}
    offset = 0
    for name, width in payload["metadata"]["observation_layout"]:
        observation_slices[name] = slice(offset, offset + int(width))
        offset += int(width)
    try:
        env, cfg = build_oracle_play_environment(
            payload, num_envs=len(CASES), device=device, env_patch=patch,
        )
        ctrl_dt = float(env.cfg.ctrl_dt)
        steps = math.ceil(duration_s / ctrl_dt)
        teacher = MotionQualityOracleTeacher(payload, env, str(cfg.training.device))
        env.set_autoreset(False)
        env.init_state()
        env.reset(np.arange(len(CASES), dtype=np.int32))
        # Only the acceptance state changes; training reset/randomization stays untouched.
        initial = env.state
        initial.info["gait_phase"][:] = np.asarray(
            [0.0, np.pi], dtype=initial.info["gait_phase"].dtype,
        )
        initial.info["mq_gait_period"].fill(1.0 / float(env.cfg.reward_config.gait_frequency))
        initial_state = {
            "base_pos_world_m": env.get_base_pos(),
            "base_quat_wxyz": env.get_base_quat(),
            "joint_pos_rad": env.get_dof_pos(),
            "joint_vel_rad_s": env.get_dof_vel(),
            "base_lin_vel_world_m_s": env.get_base_lin_vel(),
            "base_ang_vel_body_rad_s": np_quat_apply_inverse(
                env.get_base_quat(), env.get_base_ang_vel(),
            ),
        }
        for name, values in initial_state.items():
            if not np.allclose(values, values[:1], rtol=0.0, atol=1e-7):
                raise ValueError(f"acceptance cases require identical initial state: {name}")
        obs, info = set_transition_input_rows(env, command_info_key="commands", command_rows=commands)
        metadata.update({
            "effective_config": OmegaConf.to_container(cfg, resolve=True),
            "effective_environment": env.get_motion_quality_deployment_profile()["environment_config"],
            "device": str(cfg.training.device), "ctrl_dt": ctrl_dt, "planned_steps": steps,
            "planned_duration_s": steps * ctrl_dt,
            "action_joint_names": payload["metadata"]["actuated_joint_names"],
            "initial_base_pos_world_m": env.get_base_pos().tolist(),
            "initial_base_quat_wxyz": env.get_base_quat().tolist(),
            "initial_conditions": {name: values.tolist() for name, values in initial_state.items()},
            "initial_gait_phase_rad": info["gait_phase"].tolist(),
            "gait_period_s": info["mq_gait_period"].tolist(),
        })
        guarded = np.zeros(len(CASES), dtype=bool)

        def measure_guard(_state: Any, indices: np.ndarray | None) -> None:
            guarded.fill(False)
            if indices is not None:
                guarded[indices] = True

        env.set_step_measurement_callback(measure_guard)
        started = time.monotonic()
        print(f"[Oracle eval] {steps} steps, ctrl_dt={ctrl_dt:g}s, four fixed cases -> {output}", flush=True)
        for step in range(1, steps + 1):
            # Refresh BEFORE querying the policy; never feed the previous random command.
            commanded = commands.copy()
            commanded[~active] = 0.0
            obs, info = set_transition_input_rows(
                env, command_info_key="commands", command_rows=commanded,
            )
            policy_commands = obs["obs"][:, observation_slices["command"]].copy()
            policy_phase = obs["obs"][:, observation_slices["gait_phase"]].copy()
            policy_period = info["mq_gait_period"].copy()
            if not np.allclose(policy_commands[active], commands[active], rtol=0.0, atol=1e-6):
                raise RuntimeError("actor observations do not contain the requested fixed commands")
            actions = teacher.actions_from_env_observation(obs, info)
            actions[~active] = 0.0
            state = env.step(actions)
            values, valid = _base_measurements(env)
            valid &= ~guarded
            for i in np.flatnonzero(active):
                name = CASES[i][0]
                terminated, truncated = bool(state.terminated[i]), bool(state.truncated[i])
                row = {
                    "case": name, "step": step, "time_s": step * ctrl_dt,
                    "wall_elapsed_s": time.monotonic() - started,
                    "command_vx_m_s": float(commands[i, 0]), "command_vy_m_s": float(commands[i, 1]),
                    "command_wz_rad_s": float(commands[i, 2]), "physical_valid": bool(valid[i]),
                    "policy_command_vx_m_s": float(policy_commands[i, 0]),
                    "policy_command_vy_m_s": float(policy_commands[i, 1]),
                    "policy_command_wz_rad_s": float(policy_commands[i, 2]),
                    "policy_gait_phase_left_rad": float(policy_phase[i, 0]),
                    "policy_gait_phase_right_rad": float(policy_phase[i, 1]),
                    "policy_gait_period_s": float(policy_period[i]),
                    "terminated": terminated, "truncated": truncated,
                    **dict(zip(PHYSICAL_COLUMNS, values[i].tolist() if valid[i] else [None] * len(PHYSICAL_COLUMNS))),
                    **dict(zip(action_columns, actions[i].tolist())),
                }
                rows.append(row)
                reason = ("invalid_physics" if not valid[i] else "terminated" if terminated
                          else "truncated" if truncated else "completed" if step == steps else None)
                if reason is not None:
                    outcomes[name] = {"reason": reason, "completed": reason == "completed",
                                      "step": step, "actual_duration_s": step * ctrl_dt,
                                      "terminated": terminated, "truncated": truncated}
                    active[i] = False
            if record_video:
                snapshots.append(env.get_physics_state_snapshot())
            if step == steps or not active.any() or step % max(1, round(1.0 / ctrl_dt)) == 0:
                print(f"[Oracle eval] step={step}/{steps} active={int(active.sum())}/4", flush=True)
            if not active.any():
                break
        if record_video and snapshots:
            from unilab.base.backend.mujoco.playback import render_mujoco_states_video

            render_mujoco_states_video(
                env=env, state_list=snapshots, output_video=output / "rollout.mp4",
                camera_kwargs={name: cfg.training[name] for name in (
                    "cam_distance", "cam_elevation", "cam_azimuth",
                )},
            )
    except BaseException as exc:
        execution_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for name, _ in CASES:
            if name not in outcomes:
                case_rows = [row for row in rows if row["case"] == name]
                outcomes[name] = {"reason": execution_status, "completed": False,
                                  "step": case_rows[-1]["step"] if case_rows else 0,
                                  "actual_duration_s": case_rows[-1]["time_s"] if case_rows else 0.0}
        summary = {"execution_status": execution_status, "error": error, "cases": outcomes,
                   "all_cases_completed": all(item["completed"] for item in outcomes.values()),
                   "policy_quality_accepted": None}
        try:
            with (output / "timeseries.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            (output / "outcome.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        finally:
            if env is not None:
                env.close()
    print(f"[Oracle eval] results: {output}", flush=True)
    return summary
