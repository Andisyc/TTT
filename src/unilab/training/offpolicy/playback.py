"""Checkpoint loading, export, and playback for off-policy policies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parents[4]

from unilab.training import (
    create_env,
    log_playback_plan,
)
from unilab.training import (
    resolve_checkpoint_path as resolve_checkpoint_path_common,
)
from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

from .factory import build_offpolicy_env_cfg_override


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    xpu = getattr(torch_module, "xpu", None)
    xpu_is_available = getattr(xpu, "is_available", None)
    if callable(xpu_is_available) and xpu_is_available():
        return "xpu"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint_path(
    root_dir: Path, algo_log_name: str, task: str, load_run: str | int
) -> tuple[str | None, str | None]:
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path_common(
        Path(root_dir) / "logs" / algo_log_name / task,
        load_run,
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def extract_reset_obs(reset_result):
    """Extract obs_dict from env.reset(...) using the current (obs_dict, info_dict) contract."""
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs_out, _ = reset_result
            return obs_out
    raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")


def resolve_play_obs_dim(obs_groups_spec: dict[str, int]) -> int:
    obs_dim, _ = resolve_play_obs_dims(obs_groups_spec)
    return obs_dim


def resolve_play_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    from unilab.base.observations import get_obs_dims

    obs_dim, critic_obs_dim = get_obs_dims(obs_groups_spec)
    return int(obs_dim), int(critic_obs_dim)


def extract_play_obs(obs_dict):
    from unilab.base.observations import split_obs_dict

    obs_out, _ = split_obs_dict(obs_dict)
    return obs_out


def resolve_play_actor_spec(
    algo_name: str,
    cfg: DictConfig,
    *,
    obs_dim: int,
    critic_obs_dim: int,
) -> tuple[str, dict[str, Any]]:
    """Resolve the actor implementation and model kwargs used by off-policy play."""
    if algo_name != "sac":
        return algo_name, {}

    from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime

    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    if custom_runtime is None:
        return "sac", {}

    actor_algo_type = str(custom_runtime.algo_type or algo_name)
    actor_kwargs = custom_runtime.build_model_kwargs(
        obs_dim=int(obs_dim),
        critic_obs_dim=int(critic_obs_dim),
    )
    return actor_algo_type, actor_kwargs


def play_offpolicy(algo_name: str, cfg: DictConfig) -> str | None:
    """Play pipeline for off-policy algorithms."""
    if OmegaConf.select(cfg, "oracle.identity") == "motion_quality_oracle_v1":
        from unilab.training.oracle.playback import play_oracle

        return play_oracle(cfg)
    import numpy as np
    import torch

    from unilab.algos.torch.common.actor_factory import build_actor
    from unilab.algos.torch.offpolicy.worker import (
        offpolicy_actor_requires_priv_info,
        resolve_offpolicy_actor_priv_info,
    )

    load_path, load_path_dir = resolve_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None

    cfg = (
        resolve_sim2sim_config(
            load_path_dir,
            cfg,
            algo_name=algo_name,
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )
        or cfg
    )

    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    device = default_device(torch, cfg.training.device)
    print(f"Using device for play: {device}")

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.training.play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        algo_name,
        cfg,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    actor_requires_priv_info = offpolicy_actor_requires_priv_info(actor_algo_type)

    normalizer = None
    if algo_name == "sac":
        actor = build_actor(
            actor_algo_type,
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
            **actor_kwargs,
        )
    elif algo_name == "td3":
        import torch

        from unilab.algos.torch.fast_td3.learner import EmpiricalNormalization, TD3Actor

        actor = TD3Actor(
            obs_dim,
            action_dim,
            cfg.training.play_env_num,
            cfg.algo.algo_params.init_scale,
            cfg.algo.actor_hidden_dim,
            cfg.algo.algo_params.log_std_min,
            cfg.algo.algo_params.log_std_max,
            torch.device(device),
        )
        if cfg.algo.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    elif algo_name == "flashsac":
        actor = build_actor(
            "flashsac",
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
            actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
            actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        )
        if cfg.algo.obs_normalization:
            from unilab.algos.torch.common.normalization import EmpiricalNormalization

            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor = cast(Any, actor)
    actor.eval()

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    with policy_load_dim_guard(env_obs_dim=obs_dim, env_action_dim=action_dim, algo_name=algo_name):
        if algo_name in ("sac", "flashsac"):
            actor.load_state_dict(checkpoint["actor"])
            if normalizer and checkpoint.get("obs_normalizer"):
                normalizer.load_state_dict(checkpoint["obs_normalizer"])
                normalizer.eval()
        else:
            actor_state = {
                k: v for k, v in checkpoint["actor"].items() if k not in ("noise_scales",)
            }
            actor.load_state_dict(actor_state, strict=False)
            if normalizer and checkpoint.get("obs_normalizer"):
                normalizer.load_state_dict(checkpoint["obs_normalizer"])
                normalizer.eval()

    # Export actor to ONNX
    if load_path_dir is not None and bool(getattr(cfg.training, "export_onnx", True)):
        onnx_path = os.path.join(load_path_dir, "policy.onnx")
        dummy_input = torch.randn(1, obs_dim, device=device)
        dummy_priv_info = (
            torch.zeros(
                (1, int(actor_kwargs["priv_info_dim"])),
                device=device,
                dtype=dummy_input.dtype,
            )
            if actor_requires_priv_info
            else None
        )
        with torch.inference_mode():
            if normalizer:
                dummy_input = normalizer(dummy_input, update=False)
            if algo_name in ("sac", "flashsac"):
                export_module = actor.as_export_module()
                output_names = ["action"]
            else:
                export_module = actor
                output_names = ["action"]
            export_args = (
                (dummy_input, dummy_priv_info) if dummy_priv_info is not None else (dummy_input,)
            )
            input_names = ["obs", "priv_info"] if dummy_priv_info is not None else ["obs"]
            torch.onnx.export(
                export_module,
                export_args,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
            )
        print(f"Exported actor ONNX to {onnx_path}")

        # Verify ONNX output matches PyTorch
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        verify_input = torch.randn(1, obs_dim, device=device)
        verify_priv_info = (
            torch.zeros((1, int(actor_kwargs["priv_info_dim"])), device=device)
            if actor_requires_priv_info
            else None
        )
        with torch.inference_mode():
            onnx_feed = normalizer(verify_input, update=False) if normalizer else verify_input
            pt_output = (
                export_module(onnx_feed, verify_priv_info)
                if verify_priv_info is not None
                else export_module(onnx_feed)
            )
            if isinstance(pt_output, tuple):
                pt_output = pt_output[0]
            pt_np = pt_output.cpu().numpy()
        onnx_inputs = {"obs": onnx_feed.cpu().numpy().astype(np.float32)}
        if verify_priv_info is not None:
            onnx_inputs["priv_info"] = verify_priv_info.cpu().numpy().astype(np.float32)
        onnx_output = sess.run(None, onnx_inputs)[0]
        max_diff = np.max(np.abs(pt_np - onnx_output))
        mean_diff = np.mean(np.abs(pt_np - onnx_output))
        print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
        if max_diff > 1e-4:
            print("WARNING: ONNX output diverges from PyTorch!")
        else:
            print("ONNX export verified OK.")
    elif load_path_dir is not None:
        print("Skipping ONNX export because training.export_onnx=false.")

    if env.state is None:
        env.init_state()

    current_priv_info: np.ndarray | None = None

    def _resolve_play_priv_info(obs_dict: dict[str, np.ndarray], info: dict | None) -> np.ndarray:
        if not actor_requires_priv_info:
            raise ValueError("Privileged play info was requested for a non-privileged actor.")
        from unilab.base.observations import split_obs_dict

        actor_obs_np, critic_np = split_obs_dict(obs_dict)
        priv_info = resolve_offpolicy_actor_priv_info(
            algo_type=actor_algo_type,
            obs_np=np.asarray(actor_obs_np, dtype=np.float32),
            critic_np=np.asarray(critic_np, dtype=np.float32),
            info=info,
        )
        if priv_info is None:
            raise ValueError(f"{actor_algo_type} play step is missing privileged info.")
        return priv_info

    def _extract_reset_play_obs(reset_result) -> np.ndarray:
        nonlocal current_priv_info
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")
        obs_out, info_out = reset_result
        if actor_requires_priv_info:
            current_priv_info = _resolve_play_priv_info(obs_out, info_out)
        return np.asarray(extract_play_obs(obs_out), dtype=np.float32)

    def _policy_step(obs_np: np.ndarray) -> np.ndarray:
        nonlocal current_priv_info
        obs_torch = torch.from_numpy(obs_np).to(device)
        if normalizer:
            obs_torch = normalizer(obs_torch, update=False)
        if actor_requires_priv_info:
            if current_priv_info is None:
                raise ValueError(f"{actor_algo_type} play step is missing privileged info.")
            priv_info_torch = torch.from_numpy(current_priv_info).to(device)
            actions_np = (
                actor.explore(
                    obs_torch,
                    priv_info_torch,
                    deterministic=True,
                )
                .cpu()
                .numpy()
            )
        elif algo_name in ("sac", "flashsac"):
            actions_np = actor.explore(obs_torch, deterministic=True).cpu().numpy()
        else:
            actions_np = actor(obs_torch).cpu().numpy()
        state = env.step(actions_np)
        if actor_requires_priv_info:
            current_priv_info = _resolve_play_priv_info(state.obs, state.info)
        return np.asarray(extract_play_obs(state.obs), dtype=np.float32)

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=os.path.join(load_path_dir, "play_video.mp4") if load_path_dir else None,
            initialize=lambda: _extract_reset_play_obs(
                env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
            ),
            step=_policy_step,
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
            },
            on_plan=log_playback_plan,
        )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} ...")
    print("Done.")
    return play_video_path
