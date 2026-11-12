from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from unilab.terrains.terrain_generator import TerrainGeneratorCfg


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass
class ScenePoleCfg:
    """A collidable free cylinder; position is its world-space body center."""

    name: str
    position: tuple[float, float, float]
    radius: float = 0.075
    height: float = 0.8
    mass: float = 1.0
    friction: float = 0.6

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ScenePoleCfg.name must be a nonempty string")
        try:
            position = tuple(float(value) for value in self.position)
        except (TypeError, ValueError) as exc:
            raise ValueError("ScenePoleCfg.position must contain three finite coordinates") from exc
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("ScenePoleCfg.position must contain three finite coordinates")
        self.position = position
        for name in ("radius", "height", "mass", "friction"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ScenePoleCfg.{name} must be finite") from exc
            if not math.isfinite(value) or value < 0 or (name != "friction" and value == 0):
                raise ValueError(f"ScenePoleCfg.{name} must be {'nonnegative' if name == 'friction' else 'positive'} and finite")
            setattr(self, name, value)


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    task_poles: list[ScenePoleCfg] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.task_poles, (list, tuple)):
            raise ValueError("SceneCfg.task_poles must be a list of pole configurations")
        poles = []
        for pole in self.task_poles:
            if isinstance(pole, Mapping):
                pole = ScenePoleCfg(**pole)
            elif isinstance(pole, ScenePoleCfg):
                pole.__post_init__()
            else:
                raise ValueError("SceneCfg.task_poles entries must be ScenePoleCfg or dictionaries")
            poles.append(pole)
        if len({pole.name for pole in poles}) != len(poles):
            raise ValueError("SceneCfg.task_poles names must be unique")
        if poles and (self.fragment_files or self.terrain is not None):
            raise ValueError("SceneCfg.task_poles cannot be combined with fragments or terrain")
        self.task_poles = poles
