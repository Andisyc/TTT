"""Oracle mass-DR choices reach environment construction without changing other DR."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training.backend_adapter import BackendAdapter
from unilab.training.oracle.config import validate_oracle_config


@pytest.fixture
def oracle_cfg():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "conf/offpolicy")):
        return compose(
            config_name="config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_motion_quality_oracle",
                "algo.max_iterations=50000",
                "oracle.resume_checkpoint=null",
            ],
        )


@pytest.mark.parametrize("enabled,bounds", [(False, [0.0, 0.0]), (True, [-1.5, 1.5])])
def test_mass_choice_reaches_oracle_environment(oracle_cfg, enabled, bounds):
    oracle_cfg.env.domain_rand.randomize_base_mass = enabled
    oracle_cfg.env.domain_rand.added_mass_range = bounds
    env = BackendAdapter(oracle_cfg, root_dir=Path.cwd()).build_task_env_cfg_override()
    assert env["domain_rand"]["randomize_base_mass"] is enabled
    assert env["domain_rand"]["added_mass_range"] == bounds
    assert env["domain_rand"]["randomize_body_mass"] is False
    assert env["domain_rand"]["random_com"] is False
    assert env["runtime_payload"]["enabled"] is False
    for name in ("randomize_kp", "randomize_kd", "randomize_ground_friction"):
        assert env["domain_rand"][name] is True


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("randomize_base_mass", "false", "must be a boolean"),
        ("randomize_base_mass", 0, "must be a boolean"),
        ("added_mass_range", [-2.0, 2.0], "added_mass_range"),
        ("randomize_body_mass", True, "randomize_body_mass"),
        ("random_com", True, "random_com"),
        ("randomize_kp", False, "randomize_kp"),
    ],
)
def test_mass_choice_preserves_oracle_guardrails(oracle_cfg, path, value, message):
    OmegaConf.update(oracle_cfg, f"env.domain_rand.{path}", value)
    with pytest.raises(ValueError, match=message):
        validate_oracle_config(oracle_cfg)
