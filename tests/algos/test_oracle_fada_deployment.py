from __future__ import annotations

from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from tests.algos.test_oracle_fada_source import recipe as recipe
from tests.algos.test_oracle_fada_source import source as source
from unilab.algos.torch.distill.fada.oracle_deployment import (
    build_oracle_student_play_env_cfg,
    oracle_student_source_profile,
)
from unilab.algos.torch.distill.fada.oracle_source import build_oracle_fada_source


@pytest.fixture
def profile(source):
    path, _, _ = source
    env, _, result = build_oracle_fada_source(path, num_envs=1)
    env.close()
    result["environment_config"]["control_config"] = {"action_scale": 1.0}
    result["environment_config"]["gait_phase_enabled"] = True
    return result


def _cfg():
    return OmegaConf.create({
        "training": {"play_only": False, "fada": {"scene_model_file": None}},
        "env": {"sim_dt": 0.001, "ctrl_dt": 0.01, "control_config": {"action_scale": 0.25},
                "runtime_payload": {"enabled": True, "added_mass_kg": 3.0}},
        "play_profile": {"enabled": True, "env": {
            "noise_config": {"level": 0.0}, "domain_rand": {"randomize_kp": False},
        }},
    })


def test_source_controls_win_over_legacy_task_while_payload_and_nominal_play_are_applied(profile, tmp_path):
    cfg = _cfg()
    before = deepcopy(profile)
    restored = build_oracle_student_play_env_cfg(profile, cfg, root_dir=tmp_path)
    assert restored["sim_dt"] == 1 / 150
    assert restored["ctrl_dt"] == 0.02
    assert restored["control_config"]["action_scale"] == 1.0
    assert restored["gait_phase_enabled"] is True
    assert restored["runtime_payload"] == {"enabled": True, "added_mass_kg": 3.0}
    assert restored["noise_config"]["level"] == 0.0
    assert restored["domain_rand"]["randomize_kp"] is False
    assert profile == before
    assert cfg.training.play_only is False


@pytest.mark.parametrize("field,value", [
    ("ctrl_dt", 0.01), ("control_config", {"action_scale": 0.25}), ("gait_phase_enabled", False),
])
def test_play_profile_cannot_change_student_control_or_phase_contract(profile, tmp_path, field, value):
    cfg = _cfg()
    cfg.play_profile.env[field] = value
    with pytest.raises(ValueError, match=field):
        build_oracle_student_play_env_cfg(profile, cfg, root_dir=tmp_path)


def test_teacher_file_is_provenance_only_and_scene_can_relocate(profile, tmp_path):
    profile["teacher_identity"]["checkpoint_path"] = str(tmp_path / "absent_teacher.pt")
    restored_profile = oracle_student_source_profile(
        {"runtime_config": {"source_route": "unilab_oracle", "source_profile": profile}},
        observation_contract="g1_fada_state_v2",
    )
    assert restored_profile is not None
    cfg = _cfg()
    cfg.training.fada.scene_model_file = "relocated/scene.xml"
    restored = build_oracle_student_play_env_cfg(restored_profile, cfg, root_dir=tmp_path)
    assert restored["scene"]["model_file"] == str(tmp_path / "relocated/scene.xml")
    assert restored_profile["control_contract"]["asset_sha256"] == profile["control_contract"]["asset_sha256"]


def test_new_route_rejects_missing_or_unsupported_profile_but_old_student_keeps_its_route(profile):
    for checkpoint in ({}, {"runtime_config": None}, {"runtime_config": {"source_route": "official"}}):
        assert oracle_student_source_profile(checkpoint, observation_contract="legacy_actor_obs_v1") is None
    with pytest.raises(ValueError, match="embedded"):
        oracle_student_source_profile(
            {"runtime_config": {"source_route": "unilab_oracle"}}, observation_contract="g1_fada_state_v2",
        )
    profile["version"] = 2
    with pytest.raises(ValueError, match="kind/version"):
        oracle_student_source_profile(
            {"runtime_config": {"source_profile": profile}}, observation_contract="g1_fada_state_v2",
        )
