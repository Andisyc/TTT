"""Composition of the pinned author's trainer; no independent learning loop."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from .oracle import OracleAlgorithm, read_checkpoint
from .upstream import UPSTREAM_REVISION, import_upstream


@dataclass(frozen=True)
class RuntimeFactory:
    """Spawn-safe immutable paths only: never carry an env, tensor or CUDA owner."""

    project_root: str
    oracle_checkpoint: str
    prepared_scene: str
    backend: str

    def __call__(self, cfg, *, runtime_log_dir, experiment_override_args=None):
        if experiment_override_args:
            raise ValueError("Use the standalone source configuration rather than Holosoma experiment overrides")
        upstream = import_upstream(self.project_root)
        upstream.trainer.set_seed(int(cfg.seed))
        from holosoma.config_types.env import get_tyro_env_config
        from holosoma.utils.simulator_config import SimulatorType, set_simulator_type_enum
        from .environment.config import build_experiment_config

        metadata = read_checkpoint(self.oracle_checkpoint)["metadata"]
        config = build_experiment_config(
            metadata, Path(self.prepared_scene), backend=self.backend,
            device=str(cfg.device), num_envs=int(cfg.num_envs), seed=int(cfg.seed),
            run_dir=Path(runtime_log_dir),
        )
        # The stock string dispatcher recognizes only its built-in class paths;
        # the source-specific subclass still uses the official MuJoCo backend.
        set_simulator_type_enum(SimulatorType.MUJOCO)
        from .environment.runtime import OracleEnvironment

        env = OracleEnvironment(get_tyro_env_config(config), device=str(cfg.device), metadata=metadata)
        env.transformer_command_profile = cfg.command_profile
        try:
            algorithm = OracleAlgorithm(self.oracle_checkpoint, device=str(cfg.device))
            expert = upstream.trainer.ExpertPolicyWrapper(algorithm)
            return env, None, expert, str(cfg.device), config.to_serializable_dict()
        except BaseException:
            env.close()
            raise


def build_training_config(upstream, checkpoint: Path, output: Path, arguments: list[str]):
    parser = upstream.config_module.build_arg_parser()
    namespace = parser.parse_args([
        "--expert-checkpoint", str(checkpoint), "--output-dir", str(output), *arguments,
    ])
    cfg = upstream.config_module.config_from_args(namespace)
    if Path(cfg.expert_checkpoint).expanduser().resolve() != checkpoint:
        raise ValueError("Official arguments cannot replace the declared Oracle checkpoint")
    if cfg.resume_checkpoint is not None or cfg.resume_online_buffer is not None:
        raise ValueError("This independent route starts a new run; resume is not implemented")
    if not cfg.headless or cfg.simulator_override != "keep":
        raise ValueError("Use --backend for this headless official MuJoCo source")
    for name, width in (("obs_dim", 64), ("act_dim", 29), ("cmd_dim", 7)):
        supplied = getattr(cfg, name)
        if supplied is not None and supplied != width:
            raise ValueError(f"The source requires {name}={width}")
        setattr(cfg, name, width)
    for name, relative in (
        ("offline_cache_path", "cache/optimal.npz"),
        ("offline_suboptimal_cache_path", "cache/suboptimal.npz"),
        ("warmup_ckpt_path", "cache/warmup.pt"),
    ):
        setattr(cfg, name, str(output / relative))
    cfg.compact_obs_term_scale = dict.fromkeys(
        ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0,
    )
    cfg.validate()
    return cfg


def _validate_teacher_set(cfg, metadata):
    from holosoma.fada.planner_idm.offline_data_utils import list_model_checkpoints

    needs_suboptimal = (
        bool(cfg.use_generalized_idm) and float(cfg.suboptimal_data_ratio) > 0
    ) or bool(cfg.planner_suboptimal_enabled)
    if not needs_suboptimal:
        return
    candidates = list_model_checkpoints(Path(cfg.expert_checkpoint).parent)
    if len(candidates) < 2:
        raise ValueError(
            "The official suboptimal recipe needs intermediate Oracle checkpoints in the teacher directory; "
            "only the final checkpoint is available. Supply the original teacher directory. "
            "The route will not silently disable suboptimal collection."
        )
    # Validate on CPU before creating any simulator/CUDA owner. All candidates
    # may be selected by the author's reward-aware sampler later.
    for path in candidates:
        candidate = read_checkpoint(path)["metadata"]
        for key in ("network_config", "observation_layout", "privileged_field_slices",
                    "body_names", "actuated_joint_names", "asset_sha256",
                    "sim_dt", "ctrl_dt", "action_scale", "default_joint_pos", "kp", "kd"):
            if candidate[key] != metadata[key]:
                raise ValueError(f"Intermediate Oracle contract differs at {key}: {path}")


def train(*, project_root: Path, checkpoint: Path, scene: Path, output: Path,
          backend: str, official_arguments: list[str]):
    from .environment.assets import materialize_scene

    root = project_root.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"A new independent output directory is required: {output}")
    upstream = import_upstream(root)
    cfg = build_training_config(upstream, checkpoint, output, official_arguments)
    metadata = read_checkpoint(checkpoint)["metadata"]
    _validate_teacher_set(cfg, metadata)
    if backend == "classic" and (str(cfg.device) != "cpu" or cfg.num_envs != 1):
        raise ValueError("Classic official MuJoCo requires --device cpu --num-envs 1")
    if backend == "warp" and not str(cfg.device).startswith("cuda"):
        raise ValueError("Warp requires a CUDA device; no CPU fallback is allowed")
    prepared = materialize_scene(scene, output / "source_assets", metadata)
    factory = RuntimeFactory(str(root), str(checkpoint), str(prepared.scene_file), backend)
    env = worker = wandb_run = None
    try:
        env, _, expert, device, experiment = factory(cfg, runtime_log_dir=output)
        worker = upstream.trainer.StrictLabelWorkerClient(
            cfg=cfg, runtime_log_dir=output, runtime_factory=factory,
            timeout_s=cfg.strict_label_worker_timeout_s,
        )
        wandb_run = upstream.trainer.init_wandb_run(cfg, run_dir=output)
        trainer = _assemble_trainer(
            upstream, cfg, expert_policy=expert, experiment_config=experiment,
            strict_label_worker=worker, wandb_run=wandb_run,
        )
        (output / "config_snapshot.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        (output / "source_context.json").write_text(json.dumps({
            "upstream_revision": UPSTREAM_REVISION, "backend": backend,
            "oracle_checkpoint": str(checkpoint), "oracle_metadata": metadata,
            "source_asset_sha256": prepared.source_sha256,
            "prepared_asset_sha256": prepared.transformed_sha256,
        }, indent=2))
        warmup = trainer.warmup(env)
        logs = trainer.run_loop(env, n_iters=cfg.dagger_iters, checkpoint_dir=output / "checkpoints")
        trainer.dump_training_log(output / "training_log.json", logs=logs)
        final = trainer.save_student_checkpoint(output / "model_final.pt", extra={
            "stage": "final", "num_iters": cfg.dagger_iters,
        })
        if cfg.save_online_buffer:
            trainer.online_buffer.save_npz(output / "online_buffer.npz")
        result = {"final_checkpoint": str(final), "completed_iterations": cfg.dagger_iters, "warmup": warmup}
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if wandb_run is not None:
                    wandb_run.finish()


def _assemble_trainer(upstream, cfg, *, expert_policy, experiment_config,
                      strict_label_worker=None, strict_label_env=None, wandb_run=None):
    """One constructor path shared by the live entry and offline trajectory proof."""
    import torch

    policy = upstream.build_student_policy(cfg, obs_dim=64, act_dim=29, cmd_dim=7).to(cfg.device)
    planner_optimizer = torch.optim.AdamW(policy.planner_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    idm_optimizer = torch.optim.AdamW(policy.idm_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    replay_args = dict(
        capacity=cfg.replay_capacity, obs_dim=64, act_dim=29, cmd_dim=7,
        history_len=cfg.history_len, pred_horizon=cfg.pred_horizon, require_future_obs_targets=True,
    )
    return upstream.trainer.FADATrainer(
        cfg=cfg, student_model=policy, expert_policy=expert_policy,
        offline_buffer=upstream.ReplayBuffer(**replay_args, growable=True),
        suboptimal_offline_buffer=upstream.ReplayBuffer(**replay_args, growable=True),
        online_buffer=upstream.ReplayBuffer(**replay_args, eviction_policy=cfg.online_buffer_eviction_policy),
        optimizer=planner_optimizer, idm_optimizer=idm_optimizer,
        lr_scheduler=None, idm_lr_scheduler=None, device=torch.device(cfg.device),
        obs_dim=64, act_dim=29, cmd_dim=7, resolved_expert_checkpoint=str(cfg.expert_checkpoint),
        compact_obs_source_checkpoint=str(cfg.expert_checkpoint), experiment_config_payload=experiment_config,
        strict_label_worker=strict_label_worker, strict_label_env=strict_label_env, wandb_run=wandb_run,
    )
