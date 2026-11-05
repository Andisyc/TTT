"""Independent privileged Oracle on the existing double-buffer SAC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime
from unilab.envs.locomotion.g1.fada_privileged import build_g1_fada_privileged_layout
from unilab.training.oracle.checkpoint import build_checkpoint_contract, save_oracle_checkpoint
from unilab.training.oracle.config import ORACLE_RUNTIME_IMPL, validate_oracle_config
from unilab.training.oracle.learner import MotionQualityOracleLearner


@dataclass(frozen=True)
class MotionQualityOracleRuntime(OffPolicyRuntime):
    """Select privileged SAC without inheriting historical FADA validation."""

    learner_cls: type[Any] | None = MotionQualityOracleLearner
    algo_type: str | None = "hora_sac"
    supports_symmetry: bool = False
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def validate_training_config(self, cfg: Any) -> None:
        validate_oracle_config(cfg)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        if critic_obs_dim <= obs_dim:
            raise ValueError("motion-quality Oracle requires a privileged critic tail")
        return {
            "priv_info_dim": critic_obs_dim - obs_dim,
            "priv_info_embed_dim": int(self.actor_cfg["priv_info_embed_dim"]),
            "priv_mlp_hidden_dims": tuple(self.actor_cfg["priv_mlp_hidden_dims"]),
            "priv_info_normalization": bool(self.actor_cfg["priv_info_normalization"]),
            "fixed_privileged_input": bool(self.actor_cfg["fixed_privileged_input"]),
        }

    def build_training_model_kwargs(
        self,
        *,
        cfg: Any,
        env: Any,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
    ) -> dict[str, Any]:
        interface = validate_oracle_config(cfg)
        if obs_dim != interface.actor_obs_dim or action_dim != interface.action_dim:
            raise ValueError("motion-quality Oracle requires 98D observations and 29D actions")
        if tuple(env.get_symmetry_obs_layouts()["obs"]) != interface.actor_obs_layout:
            raise ValueError("motion-quality Oracle observation ordering is incompatible")
        identity = env.get_fada_privileged_checkpoint_identity()
        layout = build_g1_fada_privileged_layout(identity.body_names)
        if (
            layout.schema != interface.privileged_schema
            or tuple(identity.field_slices) != layout.field_slices
            or critic_obs_dim != obs_dim + layout.width
        ):
            raise ValueError("motion-quality Oracle privileged layout is incompatible")
        if (
            len(identity.actuated_joint_names) != action_dim
            or len(set(identity.actuated_joint_names)) != action_dim
            or env.cfg.ctrl_dt != interface.ctrl_dt
            or env.cfg.control_config.action_scale != interface.action_scale
        ):
            raise ValueError("motion-quality Oracle action/control interface is incompatible")
        if self.actor_cfg != dict(cfg.algo.actor):
            raise ValueError("Oracle runtime actor configuration differs from training config")
        return {
            **self.build_model_kwargs(obs_dim=obs_dim, critic_obs_dim=critic_obs_dim),
            "obs_normalization": bool(cfg.algo.obs_normalization),
            "v_min": float(cfg.algo.value_support_min),
            "v_max": float(cfg.algo.value_support_max),
            "checkpoint_contract": build_checkpoint_contract(
                cfg, env, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim,
                action_dim=action_dim,
            ),
        }

    def build_checkpoint_saver(self, learner: Any) -> Any:
        return save_oracle_checkpoint


def resolve_motion_quality_oracle_runtime(
    rl_cfg: dict[str, Any],
) -> MotionQualityOracleRuntime | None:
    """Resolve only this experiment; shared factories own model construction."""
    if rl_cfg.get("runtime_impl") != ORACLE_RUNTIME_IMPL:
        return None
    actor_cfg = rl_cfg.get("actor")
    if not isinstance(actor_cfg, dict):
        raise ValueError("motion-quality Oracle requires an actor configuration mapping")
    return MotionQualityOracleRuntime(actor_cfg=dict(actor_cfg))
