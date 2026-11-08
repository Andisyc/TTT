"""Frozen Oracle figure-eight sessions; video encoding belongs to the recorder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.offpolicy.worker import resolve_offpolicy_actor_priv_info
from unilab.training import ensure_registries
from unilab.training.offpolicy.playback import extract_play_obs
from unilab.training.oracle.checkpoint import read_oracle_checkpoint
from unilab.training.oracle.playback import OracleInferencePolicy, build_oracle_play_environment
from unilab.visualization.demo_commands import FigureEightCommandConfig, FigureEightCommandProgram
from unilab.visualization.footprints import FootprintRecorder
from unilab.visualization.playback_sessions import OffPolicyPlaybackSession


class OracleRecordingSession(OffPolicyPlaybackSession):
    """Single-episode Oracle playback with command-before-inference timing."""

    def __init__(
        self,
        *,
        env: Any,
        policy: OracleInferencePolicy,
        checkpoint_path: Path,
        checkpoint_identity: dict[str, Any],
        command_program: FigureEightCommandProgram,
        footprint_recorder: FootprintRecorder | None = None,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("Oracle recording requires exactly one environment")
        super().__init__(
            env=env,
            device=str(next(policy.parameters()).device),
            action_mode="policy",
            actor=policy.actor,
            actor_algo_type="hora_sac",
            normalizer=policy.normalizer,
            num_envs=1,
            obs_extractor=extract_play_obs,
            priv_info_resolver=resolve_offpolicy_actor_priv_info,
        )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_identity = dict(checkpoint_identity)
        self.command_program = command_program
        self.command_control_steps = np.zeros(1, dtype=np.int64)
        self.footprint_recorder = footprint_recorder
        self.finished = False
        self.termination_reason: Literal["terminated", "truncated"] | None = None
        self._closed = False
        self.set_autoreset(False)

    def set_autoreset(self, enabled: bool) -> None:
        if enabled:
            raise ValueError("Oracle recording must stop at the first episode end")
        super().set_autoreset(False)

    def reset(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Oracle recording session is closed")
        super().reset()
        self.command_control_steps.fill(0)
        self.finished = False
        self.termination_reason = None
        if self.footprint_recorder is not None:
            self.footprint_recorder.reset(self.env.get_foot_contact())
        return self.set_external_command(self.command_program.command_at_control_step(0))

    @torch.inference_mode()
    def step_once(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Oracle recording session is closed")
        if self.obs is None:
            raise RuntimeError("Oracle recording session must be reset before stepping")
        if self.finished:
            raise RuntimeError("Oracle recording episode has ended; reset before stepping")
        self.set_external_command(self.command_program.command_at_control_step(self.step_count))
        observation = super().step_once()
        self.command_control_steps[0] = self.step_count
        state = self.env.state
        done = state.terminated | state.truncated
        self.finished = bool(done[0])
        if self.finished:
            self.termination_reason = "terminated" if state.terminated[0] else "truncated"
        if self.footprint_recorder is not None:
            self.footprint_recorder.observe(
                foot_positions=self.env.get_foot_pos(),
                contacts=self.env.get_foot_contact(),
                done=done,
                autoreset=False,
            )
        return observation

    def close(self) -> None:
        if not self._closed:
            self.env.close()
            self._closed = True


def create_oracle_recording_session(
    checkpoint_path: str | Path,
    *,
    figure_eight: FigureEightCommandConfig,
    device: str = "cpu",
    scene_model_file: str | Path | None = None,
    max_episode_seconds: float = 180.0,
    footprints: bool = True,
) -> OracleRecordingSession:
    """Restore checkpoint-owned physics and apply only nominal recording settings.

    The returned session is unreset. Call reset() before stepping, and close()
    in the caller's finally block. Failed construction closes the created env.
    """
    if not np.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
        raise ValueError("max_episode_seconds must be finite and positive")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = read_oracle_checkpoint(checkpoint_path)
    cfg = OmegaConf.create(payload["metadata"]["config"])
    if not isinstance(cfg, DictConfig):
        raise ValueError("Oracle checkpoint config must be a mapping")
    if scene_model_file is not None:
        OmegaConf.update(
            cfg, "env.scene.model_file",
            str(Path(scene_model_file).expanduser().resolve()), force_add=True,
        )
    ensure_registries()
    env, _ = build_oracle_play_environment(
        payload,
        cfg,
        num_envs=1,
        device=device,
        env_patch={
            "max_episode_seconds": float(max_episode_seconds),
            "reset_base_qvel_limit": 0.0,
            "standing_reset_base_qvel_limit": 0.0,
            "noise_config": {"level": 0.0},
            "runtime_payload": {"enabled": False},
            "commands": {
                "resampling_time": 0.0,
                "heading_command": False,
                "rel_standing_envs": 0.0,
                "rel_transition_envs": 0.0,
                "rel_straight_envs": 0.0,
            },
            "domain_rand": {
                "randomize_reset_pose": False,
                "randomize_kp": False,
                "randomize_kd": False,
                "randomize_ground_friction": False,
                "randomize_base_mass": False,
                "randomize_body_mass": False,
                "random_com": False,
                "randomize_gravity": False,
                "randomize_dof_armature": False,
                "randomize_dof_position_bias": False,
                "randomize_control_delay": False,
                "torque_rfi_fraction": 0.0,
                "push_robots": False,
                "actuator_strength": {
                    "enabled": False,
                    "curriculum_enabled": False,
                    "group_curriculum_enabled": False,
                },
            },
        },
    )
    try:
        policy = OracleInferencePolicy(payload, env, device)
        program = FigureEightCommandProgram.from_config(
            figure_eight, ctrl_dt=float(env.cfg.ctrl_dt),
        )
        program.validate_environment(env.cfg, keyboard=False)
        footprint_recorder = (
            FootprintRecorder(num_envs=1, ctrl_dt=float(env.cfg.ctrl_dt))
            if footprints else None
        )
        return OracleRecordingSession(
            env=env,
            policy=policy,
            checkpoint_path=checkpoint_path,
            checkpoint_identity={
                "format": payload["format"],
                "identity": payload["metadata"]["identity"],
                "iteration": payload["metadata"]["iteration"],
            },
            command_program=program,
            footprint_recorder=footprint_recorder,
        )
    except BaseException:
        env.close()
        raise
