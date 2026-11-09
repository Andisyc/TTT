"""Original FADA trainer with UniLab source bindings; no local learning loop."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.collection_contract import OFFICIAL_FADA_SOURCE_REVISION
from unilab.algos.torch.distill.fada.oracle_source import build_oracle_fada_source
from unilab.algos.torch.distill.fada.upstream_environment import UpstreamEnvironment
from unilab.algos.torch.distill.fada.upstream_oracle import UpstreamOracleAlgorithm
from unilab.training.oracle.checkpoint import read_oracle_checkpoint


def import_upstream(root: str):
    # The existing reader validates the pinned checkout and imports lazily. It does
    # not instantiate the legacy collection runtime or its IPC resources.
    from unilab.algos.torch.distill.fada.official_source_runtime import _import_upstream
    try:
        return _import_upstream(root)
    except ImportError as error:
        raise ImportError(
            "original FADA runtime dependencies are required in an isolated environment; "
            "the current UniLab environment was not modified"
        ) from error


def _source(checkpoint: str, *, device: str, num_envs: int, scene_model_file: str | None):
    raw, _, profile = build_oracle_fada_source(
        checkpoint, num_envs=num_envs, device=device, scene_model_file=scene_model_file,
    )
    try:
        env = UpstreamEnvironment(raw, device=device)
        algorithm = UpstreamOracleAlgorithm(checkpoint, raw, device)
        return env, algorithm, profile
    except BaseException:
        raw.close()
        raise


def _trainer_class(upstream):
    class UniLabSourceTrainer(upstream.FADATrainer):
        def _capture_env_snapshot(self, env):
            return env.capture()

        def _restore_env_snapshot(self, env, snapshot):
            env.restore(snapshot)

        def save_student_checkpoint(self, path, *, extra=None):
            # Preserve the author's payload and optimizer state, adding only the
            # source contract needed to consume it without the Oracle files.
            target = Path(path)
            temporary = target.with_suffix(target.suffix + ".tmp")
            super().save_student_checkpoint(temporary, extra=extra)
            payload = torch.load(temporary, map_location="cpu", weights_only=True)
            payload["unilab_upstream_source"] = self.unilab_source
            payload["rollout_step_count"] = int(self.rollout_step_count)
            torch.save(payload, temporary)
            temporary.replace(target)
            return target
    return UniLabSourceTrainer


def _shadow_entry(conn, *, root, checkpoint, cfg, scene_model_file):
    env = None
    try:
        _, upstream, _, _, _, _, _ = import_upstream(root)
        upstream.set_seed(cfg.seed)
        env, algorithm, _ = _source(
            checkpoint, device=cfg.device, num_envs=cfg.num_envs, scene_model_file=scene_model_file,
        )
        env.reset_all()
        runtime = object.__new__(_trainer_class(upstream))
        runtime.cfg = cfg
        runtime.expert_policy = upstream.ExpertPolicyWrapper(algorithm)
        runtime.device = torch.device(cfg.device)
        runtime.obs_dim, runtime.act_dim = 64, 29
        runtime.compact_obs_term_scale = cfg.compact_obs_term_scale
        runtime.compact_obs_term_noise = cfg.compact_obs_term_noise
        runtime._use_teacher_aligned_labels = True
        runtime.strict_label_env = env
        runtime.strict_label_worker = None
        conn.send({"ready": True})
        while True:
            request = conn.recv()
            if request["op"] == "close":
                break
            if request["op"] != "compute":
                raise ValueError("unknown original FADA shadow operation")
            env.restore(request["snapshot"])
            anchor = {name: value.to(runtime.device) for name, value in request["anchor_obs"].items()}
            # No autoreset or curriculum mutation may escape the shadow branch.
            with env.env.preserve_rollout_state():
                chunk, future, valid = runtime._compute_strict_expert_chunk_labels(env, anchor)
            conn.send({"ok": True, "expert_chunk": chunk.cpu(),
                       "expert_future_obs_chunk": future.cpu(), "strict_label_valid": valid.cpu()})
    except EOFError:
        pass
    except BaseException:
        import traceback
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            conn.close()


def _shadow_client(upstream, *, root, checkpoint, cfg, scene_model_file):
    class SourceShadowClient(upstream.StrictLabelWorkerClient):
        def __init__(self):
            context = mp.get_context("spawn")
            self._conn, child = context.Pipe(duplex=True)
            self._timeout_s = float(cfg.strict_label_worker_timeout_s)
            self._process = context.Process(
                target=_shadow_entry,
                kwargs={"conn": child, "root": root, "checkpoint": checkpoint,
                        "cfg": cfg, "scene_model_file": scene_model_file},
            )
            self._process.start()
            child.close()
            try:
                if not self._conn.poll(self._timeout_s):
                    raise TimeoutError("UniLab source shadow initialization timed out")
                response = self._conn.recv()
                if response != {"ready": True}:
                    raise RuntimeError(f"UniLab source shadow initialization failed: {response}")
            except BaseException:
                self.close()
                raise

        def close(self):
            try:
                if self._process.is_alive():
                    try:
                        self._conn.send({"op": "close"})
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join()
            finally:
                self._conn.close()
                self._process.close()
    return SourceShadowClient()


def source_metadata(checkpoint: str, profile: dict[str, Any]) -> dict[str, Any]:
    payload = read_oracle_checkpoint(checkpoint)
    meta = payload["metadata"]
    cfg = meta["environment_config"]
    profile = {**profile, "student_observation_contract": "official_compact_64_v1"}
    count = 29
    scale = meta["action_scale"]
    scales = [float(scale)] * count if isinstance(scale, (float, int)) else list(scale)
    identity = {
        "upstream_revision": OFFICIAL_FADA_SOURCE_REVISION,
        "expert_checkpoint": str(Path(checkpoint).resolve()),
        "expert_checkpoint_sha256": profile["teacher_identity"]["checkpoint_sha256"],
        "experiment_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    control = {
        "source_identity": identity, "joint_names": meta["actuated_joint_names"],
        "default_joint_pos": meta["default_joint_pos"], "action_scale": scales,
        "stiffness": meta["kp"], "damping": meta["kd"], "clip_actions": None,
        "sim_dt": meta["sim_dt"], "ctrl_dt": meta["ctrl_dt"],
        "gait_frequency": cfg["reward_config"]["gait_frequency"],
        "gait_clock_mode": cfg["gait_clock_mode"],
        "phase_init": None,
    }
    return {"version": 1, "source_profile": profile, "source_identity": identity, "control": control}


def run_upstream_training(*, upstream_root: str, oracle_checkpoint: str,
                          output_dir: str, overrides: list[str],
                          scene_model_file: str | None = None) -> dict[str, Any]:
    config_type, upstream, build_student, replay_class, _, _, _ = import_upstream(upstream_root)
    config_module = importlib.import_module(config_type.__module__)

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"independent training requires a new output directory: {output}")
    checkpoint = str(Path(oracle_checkpoint).expanduser().resolve())
    arguments = ["--expert-checkpoint", checkpoint, "--output-dir", str(output), *overrides]
    cfg = config_module.config_from_args(config_module.build_arg_parser().parse_args(arguments))
    if str(Path(cfg.expert_checkpoint).expanduser().resolve()) != checkpoint:
        raise ValueError("upstream overrides cannot change the declared Oracle checkpoint")
    if cfg.simulator_override != "keep" or not cfg.headless:
        raise ValueError("the UniLab source preserves its MuJoCo backend and runs headless")
    for name, expected in (("obs_dim", 64), ("act_dim", 29), ("cmd_dim", 7)):
        if getattr(cfg, name) not in (None, expected):
            raise ValueError(f"the UniLab source requires {name}={expected}")
    if cfg.resume_checkpoint is not None or cfg.resume_online_buffer is not None:
        raise ValueError("use the author's resume entry for existing runs; this entry starts an isolated source run")
    # All caches belong to this route, regardless of defaults derived from the teacher directory.
    for name, relative in (("offline_cache_path", "cache/optimal.npz"),
                           ("offline_suboptimal_cache_path", "cache/suboptimal.npz"),
                           ("warmup_ckpt_path", "cache/warmup.pt")):
        setattr(cfg, name, str(output / relative))
    cfg.obs_dim, cfg.act_dim, cfg.cmd_dim = 64, 29, 7
    cfg.compact_obs_term_scale = dict.fromkeys(
        ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0,
    )
    cfg.validate()
    upstream.set_seed(cfg.seed)
    env = worker = trainer = wandb_run = None
    try:
        env, algorithm, profile = _source(
            checkpoint, device=cfg.device, num_envs=cfg.num_envs, scene_model_file=scene_model_file,
        )
        metadata = source_metadata(checkpoint, profile)
        device = torch.device(cfg.device)
        policy = build_student(cfg, obs_dim=64, act_dim=29, cmd_dim=7).to(device)
        planner_optimizer = torch.optim.AdamW(policy.planner_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        idm_optimizer = torch.optim.AdamW(policy.idm_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        common = dict(capacity=cfg.replay_capacity, obs_dim=64, act_dim=29, cmd_dim=7,
                      history_len=cfg.history_len, pred_horizon=cfg.pred_horizon,
                      require_future_obs_targets=True)
        offline = replay_class(**common, growable=True)
        suboptimal = replay_class(**common, growable=True)
        online = replay_class(**common, eviction_policy=cfg.online_buffer_eviction_policy)
        worker = _shadow_client(upstream, root=upstream_root, checkpoint=checkpoint,
                                cfg=cfg, scene_model_file=scene_model_file)
        output.mkdir(parents=True)
        wandb_run = upstream.init_wandb_run(cfg, run_dir=output)
        trainer = _trainer_class(upstream)(
            cfg=cfg, student_model=policy, expert_policy=upstream.ExpertPolicyWrapper(algorithm),
            offline_buffer=offline, suboptimal_offline_buffer=suboptimal, online_buffer=online,
            optimizer=planner_optimizer, idm_optimizer=idm_optimizer,
            lr_scheduler=None, idm_lr_scheduler=None, device=device,
            obs_dim=64, act_dim=29, cmd_dim=7, resolved_expert_checkpoint=checkpoint,
            compact_obs_source_checkpoint=checkpoint, experiment_config_payload=profile["environment_config"],
            strict_label_worker=worker, wandb_run=wandb_run,
        )
        trainer.unilab_source = metadata
        (output / "config_snapshot.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        warmup = trainer.warmup(env)
        logs = trainer.run_loop(env, n_iters=cfg.dagger_iters, checkpoint_dir=output / "checkpoints")
        trainer.dump_training_log(output / "training_log.json", logs=logs)
        final = trainer.save_student_checkpoint(output / "model_final.pt", extra={"stage": "final", "num_iters": cfg.dagger_iters})
        if cfg.save_online_buffer:
            online.save_npz(output / "online_buffer.npz")
        result = {"source_route": "upstream_unilab_oracle", "final_checkpoint": str(final),
                  "completed_iterations": cfg.dagger_iters, "warmup": warmup}
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
