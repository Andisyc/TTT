"""HORA, generic distillation, and FADA playback session factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

from unilab.visualization.playback_policy_sessions import (
    _ensure_scripts_dir,
    _resolve_task_checkpoint_from_playback_cfg,
    select_torch_device,
)
from unilab.visualization.playback_sessions import (
    _HORA_DISTILL_CHECKPOINT_UNAVAILABLE,
    FADAPlaybackSession,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
)

LogFn = Callable[[str], None]

def _default_hora_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from train_hora_distill import (
        _apply_teacher_defaults,
        _build_play_env_cfg_override,
        _cfg_with_checkpoint_runtime,
        _format_stage2_play_checkpoint_error,
        _resolve_stage2_checkpoint_path,
        _student_policy,
    )

    from unilab.algos.torch.hora.distill import (
        build_student_actor_and_normalizer,
        load_distilled_checkpoint,
    )
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import create_env, get_log_root

    return {
        "apply_teacher_defaults": _apply_teacher_defaults,
        "build_play_env_cfg_override": _build_play_env_cfg_override,
        "build_student_actor_and_normalizer": build_student_actor_and_normalizer,
        "cfg_with_checkpoint_runtime": _cfg_with_checkpoint_runtime,
        "create_env": create_env,
        "format_stage2_play_checkpoint_error": _format_stage2_play_checkpoint_error,
        "get_log_root": get_log_root,
        "load_distilled_checkpoint": load_distilled_checkpoint,
        "resolve_stage2_checkpoint_path": _resolve_stage2_checkpoint_path,
        "student_policy": _student_policy,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
        "checkpoint_reader": torch.load,
    }


def create_hora_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an interactive playback session for HORA stage-2 student checkpoints."""

    resolved_deps = dict(_default_hora_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    load_path, load_path_dir = resolved_deps["resolve_stage2_checkpoint_path"](cfg)
    checkpoint_path = str(load_path) if load_path is not None else None
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if load_path is None or load_path_dir is None or not Path(load_path).exists():
            task_log_root = resolved_deps["get_log_root"](Path(root_dir), cfg) / str(
                cfg.training.task_name
            )
            log(
                resolved_deps["format_stage2_play_checkpoint_error"](
                    cfg,
                    task_log_root=task_log_root,
                    load_path=load_path,
                    load_path_dir=load_path_dir,
                )
            )
            log("WARNING: falling back to zero actions.")
            runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)
        else:
            log(f"Loading distilled checkpoint: {load_path}")
            checkpoint = resolved_deps["checkpoint_reader"](
                load_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
                    f"(found keys: {set(checkpoint.keys())})."
                )
            runtime_cfg = resolved_deps["cfg_with_checkpoint_runtime"](cfg, checkpoint)
    else:
        runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)

    env_cfg_override = resolved_deps["build_play_env_cfg_override"](runtime_cfg)
    create_env = resolved_deps["create_env"]
    try:
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=str(runtime_cfg.training.task_name),
        )
    except TypeError:
        if deps is None:
            raise
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
        )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    torch_device = torch.device(device_name)

    if playback_cfg.action_mode == "policy" and load_path is not None and Path(load_path).exists():
        actor, hist_normalizer = resolved_deps["build_student_actor_and_normalizer"](
            wrapped_env,
            runtime_cfg,
            device=torch_device,
        )
        resolved_deps["load_distilled_checkpoint"](
            actor,
            hist_normalizer,
            load_path,
            device=torch_device,
        )
        actor.eval()
        hist_normalizer.eval()
        student_policy = resolved_deps["student_policy"]

        def policy(obs: Any) -> Any:
            return student_policy(actor, hist_normalizer, obs, device=torch_device)

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def _apply_distill_playback_reset_contract(
    env_cfg_override: Mapping[str, Any] | None, task_name: str
) -> dict[str, Any] | None:
    """Force standing-only reset sampling for G1 distill playback owners."""

    task_key = str(task_name).lower().split("/", 1)[0].replace("-", "_")
    task_key = task_key.replace("_", "")
    if task_key not in {"g1walkflat", "g1walkheight"}:
        return dict(env_cfg_override) if env_cfg_override is not None else None
    merged = dict(env_cfg_override or {})
    commands_override = dict(merged.get("commands") or {})
    commands_override["rel_standing_envs"] = 1.0
    if "rel_transition_envs" in commands_override:
        commands_override["rel_transition_envs"] = 0.0
    merged["commands"] = commands_override
    if "standing_reset_base_qvel_limit" in merged:
        merged["standing_reset_base_qvel_limit"] = 0.0
    return merged


def _apply_keyboard_playback_reset_contract(
    env_cfg_override: Mapping[str, Any] | None,
    task_name: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Return deterministic standing reset overrides for keyboard-driven G1 playback."""

    # B1: OFF 路径只复制原配置, 保持训练和非键盘回放的 reset 分布不变.
    merged = dict(env_cfg_override or {})
    if not enabled:
        return merged

    task_key = str(task_name).lower().split("/", 1)[0].replace("-", "_")
    task_key = task_key.replace("_", "")
    if task_key not in {"g1walkflat", "g1walkheight"}:
        return merged

    # B2: ON 路径在 env 创建前原子化派生静止 command 与零根部速度 reset 合同.
    commands_override = dict(merged.get("commands") or {})
    commands_override["rel_standing_envs"] = 1.0
    commands_override["rel_transition_envs"] = 0.0
    merged["commands"] = commands_override
    merged["reset_base_qvel_limit"] = 0.0
    merged["standing_reset_base_qvel_limit"] = 0.0
    return merged


def _default_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from unilab.algos.torch.distill import load_distillation_student_policy
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()

    return {
        "build_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="distill",
        ).build_play_env_cfg_override(),
        "create_env": create_env,
        "load_student_policy": load_distillation_student_policy,
        "resolve_checkpoint": _resolve_task_checkpoint_from_playback_cfg,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
    }


def _default_fada_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from unilab.algos.torch.distill import load_fada_deployable_policy_checkpoint
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    return {
        "build_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="distill",
        ).build_play_env_cfg_override(),
        "create_env": create_env,
        "load_fada_policy": load_fada_deployable_policy_checkpoint,
        "resolve_checkpoint": _resolve_task_checkpoint_from_playback_cfg,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
    }


def create_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session for generic distillation student checkpoints."""

    resolved_deps = dict(_default_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    checkpoint = resolved_deps["resolve_checkpoint"](playback_cfg, cfg, root_dir)
    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    policy_obs_mode = playback_cfg.policy_obs_mode
    if policy_obs_mode == "auto":
        policy_obs_mode = "actor"

    create_env = resolved_deps["create_env"]
    task_name = str(getattr(cfg.training, "task_name", playback_cfg.task))
    build_env_cfg_override = resolved_deps.get("build_env_cfg_override")
    env_cfg_override = build_env_cfg_override(cfg) if build_env_cfg_override is not None else {}
    env_cfg_override = _apply_distill_playback_reset_contract(env_cfg_override, task_name)
    try:
        env = create_env(
            cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=task_name,
        )
    except TypeError:
        if deps is None:
            raise
        try:
            env = create_env(
                cfg,
                num_envs=int(playback_cfg.num_envs),
                env_cfg_override=env_cfg_override,
            )
        except TypeError:
            env = create_env(cfg, num_envs=int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if checkpoint is None or not Path(checkpoint).exists():
            log(
                "WARNING: no generic distillation student checkpoint found - "
                "falling back to zero actions."
            )
        else:
            log(f"Loading distillation student checkpoint: {checkpoint}")
            loaded_student = resolved_deps.get("load_student_policy")
            if loaded_student is None:
                from unilab.algos.torch.distill import load_distillation_student_policy

                loaded_student = load_distillation_student_policy
            from .playback_distill_policy import load_distill_playback_policy

            policy = load_distill_playback_policy(
                checkpoint=checkpoint,
                cfg=cfg,
                env=env,
                device=device_name,
                load_student_policy=loaded_student,
                log=log,
            )

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def create_fada_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[FADAPlaybackSession, str, str | None]:
    """Create a stateful FADA Planner-IDM playback session."""

    # B1: policy 模式先严格恢复 checkpoint, 避免加载失败后遗留已创建的环境资源.
    from omegaconf import OmegaConf
    experiment_recording = None
    experiment_node = OmegaConf.select(cfg, "experiment_recording", default=None)
    if experiment_node is not None and bool(experiment_node.enabled):
        from unilab.visualization.experiment_recording import ExperimentRecordingConfig

        experiment_recording = ExperimentRecordingConfig(**OmegaConf.to_container(experiment_node, resolve=True))
        if playback_cfg.action_mode != "policy" or int(playback_cfg.num_envs) != 1:
            raise ValueError("experiment recording requires single-env policy playback")

    online_node = getattr(cfg, "online_adaptation", None)
    online_enabled = bool(getattr(online_node, "enabled", False))
    online = None
    if online_enabled:
        from unilab.algos.torch.distill.fada.online_controller import (
            FADAOnlineAdaptationConfig,
        )

        online = FADAOnlineAdaptationConfig(**(
            OmegaConf.to_container(online_node, resolve=True)
            if OmegaConf.is_config(online_node) else vars(online_node)
        ))
        if playback_cfg.action_mode != "policy":
            raise ValueError("FADA online adaptation requires policy action mode")
        if int(playback_cfg.num_envs) != 1:
            raise ValueError("FADA online adaptation requires exactly one env")

    resolved_deps = dict(_default_fada_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    checkpoint = resolved_deps["resolve_checkpoint"](playback_cfg, cfg, root_dir)
    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    policy_obs_mode = playback_cfg.policy_obs_mode
    if policy_obs_mode == "auto":
        policy_obs_mode = "actor"
    controller = None
    architecture = None
    official_profile = None
    official_adapter = None
    oracle_source_profile = None
    if playback_cfg.action_mode == "policy":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "FADA policy playback requires training.play_checkpoint_path or a resolvable run."
            )
        from unilab.algos.torch.distill.fada_playback import FADAPlaybackController

        loaded = resolved_deps["load_fada_policy"](checkpoint, device=device_name)
        from unilab.algos.torch.distill.fada.checkpoint import LoadedOfficialFADAStudentPolicy
        if isinstance(loaded, LoadedOfficialFADAStudentPolicy):
            from unilab.algos.torch.distill.fada.official_deployment import load_official_playback_profile
            from unilab.algos.torch.distill.fada.runtime_policy import get_fada_runtime_spec
            architecture = get_fada_runtime_spec(loaded.policy)
            # Bind the actual source control contract before creating any env.
            official_profile = load_official_playback_profile(
                cfg, root_dir=root_dir, source_identity=loaded.checkpoint["source_identity"],
                embedded_profile=loaded.checkpoint.get("control_profile"),
            )
            from unilab.algos.torch.distill.fada.oracle_deployment import oracle_student_source_profile
            oracle_source_profile = oracle_student_source_profile(
                loaded.checkpoint, observation_contract=architecture.observation_contract,
            )
        else:
            architecture = loaded.policy.config
            from unilab.algos.torch.distill.fada.oracle_deployment import oracle_student_source_profile
            oracle_source_profile = oracle_student_source_profile(
                loaded.checkpoint, observation_contract=architecture.observation_contract,
            )

        if online is not None:
            from unilab.algos.torch.distill.fada.observation import (
                FADA_G1_STATE_OBSERVATION_CONTRACT,
            )

            if official_profile is None and architecture.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
                raise ValueError("Online snapshot adaptation requires g1_fada_state_v2")
        elif official_profile is None:
            controller = FADAPlaybackController(loaded.policy, device=device_name)
        log(f"Loading FADA Planner-IDM checkpoint: {checkpoint}")
        completed_name = (
            "completed_steps" if "completed_steps" in loaded.checkpoint else "completed_iterations"
        )
        log(
            "FADA checkpoint diagnostics: "
            f"{completed_name}={int(loaded.checkpoint[completed_name])}, "
            f"obs_dim={architecture.obs_dim}, action_dim={architecture.action_dim}, "
            f"command_dim={architecture.command_dim}, history={architecture.history_length}, "
            f"horizon={architecture.prediction_horizon}"
        )

    # B2: 通过 distill task config 构造环境, 键盘模式先收敛为静止零速 reset, 再核对 policy IO.
    create_env = resolved_deps["create_env"]
    task_name = str(getattr(cfg.training, "task_name", playback_cfg.task))
    if official_profile is not None and task_name != "G1WalkFlat":
        raise ValueError("official Locomotion playback requires task=g1_walk_flat/mujoco")
    if oracle_source_profile is not None:
        from unilab.algos.torch.distill.fada.oracle_deployment import build_oracle_student_play_env_cfg
        if task_name != "G1WalkFlat":
            raise ValueError("Oracle student playback requires task=g1_walk_flat/mujoco")
        env_cfg_override = build_oracle_student_play_env_cfg(
            oracle_source_profile, cfg, root_dir=root_dir,
        )
    else:
        build_env_cfg_override = resolved_deps.get("build_env_cfg_override")
        env_cfg_override = build_env_cfg_override(cfg) if build_env_cfg_override is not None else {}
    if official_profile is not None:
        from unilab.algos.torch.distill.fada.official_deployment import bind_official_playback_control

        env_cfg_override = bind_official_playback_control(env_cfg_override, official_profile)
    env_cfg_override = _apply_keyboard_playback_reset_contract(
        env_cfg_override,
        task_name,
        enabled=playback_cfg.keyboard,
    )
    if official_profile is not None:
        env_cfg_override["sim_dt"] = official_profile.sim_dt
    if playback_cfg.keyboard:
        log("Keyboard playback reset: standing command with zero base velocity.")
    demo = OmegaConf.select(cfg, "demo.figure_eight", default=None)
    recording_duration = OmegaConf.select(cfg, "video_recording.duration_s", default=None)
    slalom_node = OmegaConf.select(cfg, "demo.slalom", default=None)
    slalom_program = None
    if slalom_node is not None and bool(slalom_node.enabled):
        if task_name != "G1WalkFlat" or playback_cfg.keyboard or online_enabled:
            raise ValueError("Slalom requires frozen, non-keyboard G1WalkFlat playback")
        if demo is not None and bool(demo.enabled):
            raise ValueError("Slalom and figure-eight cannot both own velocity commands")
        if bool(OmegaConf.select(cfg, "env.runtime_payload.enabled", default=False)):
            raise ValueError("The nominal slalom Demo requires runtime_payload.enabled=false")
        from dataclasses import replace

        from unilab.assets import ASSETS_ROOT_PATH
        from unilab.base.scene import SceneCfg, ScenePoleCfg
        from unilab.visualization.slalom import SlalomCommandProgram, SlalomConfig

        slalom_cfg = SlalomConfig(**cast(dict[str, Any], OmegaConf.to_container(slalom_node, resolve=True)))
        slalom_program = SlalomCommandProgram.from_config(
            slalom_cfg, ctrl_dt=float(env_cfg_override.get("ctrl_dt", 0.02)),
            prediction_horizon=(architecture.prediction_horizon if architecture is not None else 1),
        )
        poles = [
            ScenePoleCfg(
                name=f"slalom_pole_{i+1}", position=(float(x), float(y), slalom_cfg.pole_height_m/2),
                radius=slalom_cfg.pole_radius_m, height=slalom_cfg.pole_height_m,
                mass=slalom_cfg.pole_mass_kg, friction=slalom_cfg.pole_friction,
            )
            for i, (x, y) in enumerate(slalom_program.pole_centers)
        ]
        scene = env_cfg_override.get("scene")
        if isinstance(scene, SceneCfg):
            if scene.task_poles:
                raise ValueError("Slalom cannot replace an existing task-pole layout")
            env_cfg_override["scene"] = replace(scene, task_poles=poles)
        else:
            scene = dict(scene) if scene is not None else {
                "model_file": str(ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml"),
            }
            if scene.get("task_poles"):
                raise ValueError("Slalom cannot replace an existing task-pole layout")
            env_cfg_override["scene"] = {**scene, "task_poles": poles}
        env_cfg_override["max_episode_seconds"] = max(
            float(env_cfg_override.get("max_episode_seconds", 0)),
            slalom_program.duration_s + slalom_cfg.settle_s + 1,
            float(recording_duration or 0) + 1,
        )
    if (demo is not None and bool(demo.enabled) and recording_duration is not None
            and bool(OmegaConf.select(cfg, "video_recording.enabled", default=False))):
        # A long route recording must not restart its phase at the task's old timeout.
        env_cfg_override["max_episode_seconds"] = max(
            float(env_cfg_override.get("max_episode_seconds", 0.0)),
            float(recording_duration) + 1.0,
        )
    env = create_env(
        cfg,
        num_envs=int(playback_cfg.num_envs),
        env_cfg_override=env_cfg_override,
        sim_backend="mujoco",
        task_name=task_name,
    )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")
    wrapped_env = None
    recorder = None
    try:
        if oracle_source_profile is not None:
            from unilab.algos.torch.distill.fada.oracle_deployment import validate_oracle_student_environment
            validate_oracle_student_environment(oracle_source_profile, env)
            log(f"Oracle student source controls: sim_dt={env.cfg.sim_dt}, ctrl_dt={env.cfg.ctrl_dt}")
        command_program = slalom_program
        demo = getattr(getattr(cfg, "demo", None), "figure_eight", None)
        if demo is not None and bool(getattr(demo, "enabled", False)):
            from unilab.visualization.demo_commands import (
                FigureEightCommandConfig,
                FigureEightCommandProgram,
            )

            command_program = FigureEightCommandProgram.from_config(
                FigureEightCommandConfig(
                    width_m=float(demo.width_m), height_m=float(demo.height_m),
                    speed_m_s=float(demo.speed_m_s),
                    segment_duration_s=float(demo.segment_duration_s),
                    forward_scale=float(getattr(demo, "forward_scale", 1.0)),
                    left_yaw_scale=float(getattr(demo, "left_yaw_scale", 1.0)),
                    right_yaw_scale=float(getattr(demo, "right_yaw_scale", 1.0)),
                    time_scale=float(getattr(demo, "time_scale", 1.0)),
                ),
                ctrl_dt=float(env.cfg.ctrl_dt),
                prediction_horizon=(architecture.prediction_horizon if architecture is not None else 1),
                window_size=(online.window_size if online is not None else 0),
                update_interval=(online.update_interval if online is not None else 0),
            )
            command_program.validate_environment(
                env.cfg, keyboard=playback_cfg.keyboard or bool(getattr(
                    getattr(cfg, "interactive", None), "keyboard", False,
                )),
            )
            log("Figure-eight demo: open-loop, segment-held commands, repeating closed reference route.")
        if slalom_program is not None:
            slalom_program.validate_environment(env.cfg, keyboard=playback_cfg.keyboard)
            log(f"Slalom demo: {len(slalom_program.pole_centers)} poles, open-loop single pass, "
                f"walking duration={slalom_program.duration_s:g}s; no pose feedback.")
        footprint_recorder = None
        footprints = getattr(getattr(cfg, "demo", None), "footprints", None)
        if footprints is not None and bool(getattr(footprints, "enabled", False)):
            from unilab.visualization.footprints import FootprintRecorder

            footprint_recorder = FootprintRecorder(
                num_envs=int(playback_cfg.num_envs), ctrl_dt=float(env.cfg.ctrl_dt),
                max_records=footprints.max_records,
            )
        wrapped_env = resolved_deps["wrapper_cls"](
            env, device=device_name, policy_obs_mode=policy_obs_mode
        )
        if official_profile is not None:
            from unilab.algos.torch.distill.fada.official_deployment import OfficialFADADeploymentAdapter
            from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
            official_adapter = OfficialFADADeploymentAdapter(env, official_profile, device=device_name)
            controller = OfficialFADAPlaybackController(
                loaded.policy, profile=official_profile,
                target_joint_names=official_adapter.target_joint_names, device=device_name,
            )
        elif architecture is not None:
            from unilab.algos.torch.distill.fada_observation import (
                raw_observation_dim_for_fada_contract,
            )

            observed_obs_dim = int(wrapped_env.num_obs)
            observed_action_dim = int(env.action_space.shape[0])
            expected_raw_obs_dim = raw_observation_dim_for_fada_contract(
                architecture.observation_contract,
                policy_observation_dim=architecture.obs_dim,
            )
            if (observed_obs_dim, observed_action_dim) != (
                expected_raw_obs_dim, architecture.action_dim,
            ):
                raise ValueError(
                    "FADA checkpoint/playback IO mismatch: "
                    f"checkpoint_raw=({expected_raw_obs_dim}, {architecture.action_dim}) "
                    f"environment=({observed_obs_dim}, {observed_action_dim})"
                )

        if online is not None:
            from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig
            from unilab.algos.torch.distill.fada.online_controller import (
                OnlineAdaptationController,
            )

            lora_config = FADALoRAConfig(
                rank=online.lora_rank, alpha=online.lora_alpha, dropout=online.lora_dropout,
            )
            controller = OnlineAdaptationController(
                loaded.policy,
                lora_config,
                window_size=online.window_size,
                warmup_steps=online.warmup_steps,
                gamma=online.gamma,
                sigma=online.sigma,
                update_interval=online.update_interval,
                lr=online.learning_rate,
                max_grad_norm=online.max_grad_norm,
                device=device_name,
                mode=online.mode,
                **({"playback": controller} if official_profile is not None else {}),
            )
            if online.mode == "fada_ttt":
                method_details = (
                    "objective=inverse_dynamics, input=realized_future, "
                    "target=saved_policy_action, reference=none, anchor=disabled, "
                    "reduction=mean_batch_action, gamma=not_applicable, sigma=0, "
                )
            else:
                objective = (
                    "self_supervised_correction" if online.mode == "self_supervised_correction"
                    else "expert_trajectory_difference"
                )
                reference = "current_lora" if online.mode == "trajectory_lora" else "frozen_base"
                method_details = (
                    f"objective={objective}, reference={reference}, "
                    f"gamma={online.gamma}, sigma={online.sigma}, "
                )
            log(
                f"Online adaptation enabled: mode={online.mode}, {method_details}"
                f"prediction_horizon={architecture.prediction_horizon}, "
                f"window_size={online.window_size}, "
                f"warmup_steps={online.warmup_steps}, update_interval={online.update_interval}, "
                f"learning_rate={online.learning_rate}, "
                f"max_grad_norm={online.max_grad_norm}, lora_rank={lora_config.rank}, "
                f"lora_alpha={lora_config.alpha}, lora_dropout={lora_config.dropout}"
            )

        log(f"Policy obs mode: {policy_obs_mode}")
        log(f"Action mode: {playback_cfg.action_mode}")
        session = FADAPlaybackSession(
            env=env,
            wrapped_env=wrapped_env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            controller=controller,
            num_envs=playback_cfg.num_envs,
            official_adapter=official_adapter,
            command_program=command_program,
            footprint_recorder=footprint_recorder,
        )
        if experiment_recording is not None:
            from unilab.visualization.experiment_recording import ExperimentRecorder, experiment_run_metadata

            recorder = ExperimentRecorder(
                env, experiment_recording,
                experiment_run_metadata(cfg, env, checkpoint_path, root_dir, official_adapter),
            )
            session.bind_experiment_recorder(recorder)
            log(f"Raw experiment records: {recorder.output_dir}")
        return session, policy_obs_mode, checkpoint_path
    except BaseException as exc:
        try:
            if recorder is not None:
                recorder.close("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        finally:
            if wrapped_env is not None:
                wrapped_env.close()
            else:
                env.close()
        raise
