"""Resolved config adaptation for training entrypoints."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.base.scene import SceneCfg
from unilab.training.reward import extract_reward_config


class BackendAdapter:
    """Build env/play overrides from the final composed config."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        root_dir: str | Path,
        algo_name: str | None = None,
        scene_materializer: Callable[..., str] = materialize_scene_visual_override,
    ) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.algo_name = algo_name
        self.scene_materializer = scene_materializer

    def build_task_env_cfg_override(self) -> dict[str, Any]:
        """Build env_cfg_override from the resolved reward + env sections."""
        env_cfg_override = extract_reward_config(self.cfg)
        env_cfg_override.update(self._to_plain_dict(getattr(self.cfg, "env", None)))
        if OmegaConf.select(self.cfg, "oracle.reward_profile") is not None:
            from unilab.envs.locomotion.g1.motion_quality_config import build_motion_quality_reward_config
            from unilab.training.oracle.config import validate_oracle_config

            validate_oracle_config(self.cfg)
            env_cfg_override["reward_config"] = build_motion_quality_reward_config(
                env_cfg_override["reward_config"]
            )
            env_cfg_override["post_step_forward_sensor"] = True

        return env_cfg_override

    def build_play_env_cfg_override(
        self, env_cfg_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply the play profile after optional checkpoint environment restoration."""
        env_cfg_override = (
            self.build_task_env_cfg_override()
            if env_cfg_override is None else dict(env_cfg_override)
        )
        play_profile = getattr(self.cfg, "play_profile", None)
        if (
            play_profile is None
            or not getattr(play_profile, "enabled", False)
            or not self.cfg.training.play_only
        ):
            return env_cfg_override

        env_profile = getattr(play_profile, "env", None)
        if env_profile is not None:
            self._apply_env_profile(env_cfg_override, env_profile)

        scene_override = getattr(play_profile, "scene", None)
        if scene_override is None or not getattr(scene_override, "enabled", False):
            return env_cfg_override

        source_model_file = getattr(scene_override, "source_model_file", None)
        if not source_model_file:
            raise ValueError("play_profile.scene.source_model_file must be configured")

        env_cfg_override["scene"] = SceneCfg(
            model_file=self.scene_materializer(
                self._resolve_root_relative_path(str(source_model_file)),
                ground_texture_file=(
                    self._resolve_root_relative_path(str(scene_override.ground_texture_file))
                    if getattr(scene_override, "ground_texture_file", None)
                    else None
                ),
                ground_texrepeat=getattr(scene_override, "ground_texrepeat", None),
                skybox_rgb1=getattr(scene_override, "skybox_rgb1", None),
                skybox_rgb2=getattr(scene_override, "skybox_rgb2", None),
            )
        )
        return env_cfg_override

    def _apply_env_profile(self, env_cfg_override: dict[str, Any], env_profile: Any) -> None:
        def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
            result = deepcopy(base)
            for key, value in patch.items():
                result[key] = (
                    merge(result[key], value)
                    if isinstance(value, dict) and isinstance(result.get(key), dict)
                    else deepcopy(value)
                )
            return result

        merged = merge(env_cfg_override, self._to_plain_dict(env_profile))
        env_cfg_override.update(merged)

    def report_effective_settings(self, env: dict[str, Any], *, mode: str) -> None:
        """Print the resolved experiment inputs, including explicit playback overrides."""
        if str(getattr(self.cfg.training, "task_name", "")) != "G1WalkFlat":
            return
        if not bool(getattr(self.cfg.training, "report_effective_settings", False)):
            return
        from unilab.envs.locomotion.g1.walk_config import normalize_g1_gait_reward

        reward = normalize_g1_gait_reward(env.get("reward_config", {}))
        curriculum = env.get("curriculum", {})
        scale = float(curriculum.get("initial_scale", 0.5)) if curriculum.get("enabled", False) else 1.0
        penalties = set(reward.get("penalty_curriculum_terms", ()))
        print(f"\n[G1 配置表 · {mode}] 命令行覆盖已合并；范围为配置上限，DR 课程可缩小实际范围。")
        print("Reward 项 | 配置权重 | 初始有效权重")
        for name, weight in reward["scales"].items():
            effective = weight * scale if name in penalties else weight
            print(f"  {name}: {weight:g} | {effective:g}" + ("（关闭）" if weight == 0 else ""))
        print(f"相位输入={env.get('gait_phase_enabled', True)}；模式={reward['feet_phase_mode']}；"
              f"固定步频={reward.get('gait_frequency')}；支撑脚姿态约束={reward.get('feet_orientation_stance_only', False)}")
        dr = env.get("domain_rand", {})
        flags = {key: value for key, value in dr.items() if isinstance(value, bool)}
        print("DR 开关：" + "；".join(f"{key}={'开' if value else '关'}" for key, value in flags.items()))
        ranges = {key: value for key, value in dr.items() if key not in flags and key != "actuator_strength"}
        for key, value in ranges.items():
            print(f"  DR.{key}={value}")
        strength = dr.get("actuator_strength", {})
        print(f"固定关节故障={strength.get('enabled', False)}；随机单关节削弱已移除")
        print(f"DR 课程={strength.get('group_curriculum_enabled', False)}；"
              f"等级={strength.get('group_curriculum_scales')}；轮次={strength.get('curriculum_iteration_boundaries')}")
        print(f"Reward 课程={curriculum.get('enabled', False)}；初始倍率={scale:g}")
        print(f"命令={env.get('commands', {})}")
        print(f"门控时钟={env.get('command_gated_phase_contact', {})}")
        print(f"观测噪声={env.get('noise_config', {})}\n")

    def _resolve_root_relative_path(self, path_value: str) -> str:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return str(candidate)
        return str((self.root_dir / candidate).resolve())

    def _to_plain_dict(self, value: Any) -> dict[str, Any]:
        if OmegaConf.is_config(value):
            resolved = OmegaConf.to_container(value, resolve=True)
        elif isinstance(value, dict):
            resolved = value
        else:
            return {}
        if not isinstance(resolved, dict):
            return {}
        return {str(key): item for key, item in resolved.items()}
