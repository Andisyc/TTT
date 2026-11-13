"""CPU-only deployment reconstruction and runtime-free batch planning regressions."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from unilab.algos.torch.distill.fada import adaptation_checkpoint as adapted_owner
from unilab.algos.torch.distill.fada import checkpoint as source_owner
from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig, fada_adapter_parameters, inject_fada_idm_lora,
)
from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION, validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.official_learning import canonical_official_training_config
from unilab.algos.torch.distill.fada.official_model import build_official_fada_student
from unilab.algos.torch.distill.fada.upstream_checkpoint import load_upstream_fada_checkpoint

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=['source', 'adapted', 'official', 'upstream'])
def checkpoint_case(request, tmp_path):
    kind = request.param
    path = tmp_path / 'policy.pt'
    if kind in ('source', 'adapted'):
        config = FADAArchitectureConfig(
            obs_dim=4, action_dim=2, command_dim=3, history_length=3,
            prediction_horizon=2, hidden_dim=8, num_heads=2, planner_layers=1,
            idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=16,
        )
        policy = FADAPlannerIDMPolicy(config)
        if kind == 'source':
            payload = {'schema_version': 2, 'architecture': asdict(config),
                       'planner_state_dict': policy.planner.state_dict(),
                       'idm_state_dict': policy.idm.state_dict()}
            torch.save(payload, path)
            loader = source_owner.load_fada_policy_checkpoint
        else:
            adapted = inject_fada_idm_lora(policy, FADALoRAConfig(dropout=0.0))
            assert isinstance(adapted.policy, FADAPlannerIDMPolicy)
            optimizer = torch.optim.Adam(fada_adapter_parameters(adapted.policy), lr=1e-4)
            adapted_owner.save_fada_adapted_checkpoint(
                path, adapted.policy, optimizer, lora_config=adapted.lora_config,
                source_checkpoint_sha256='a' * 64, target_artifact_sha256='b' * 64,
                target_artifact_schema_version='fada-target-batch/v3',
                completed_steps=0, samples_seen=0, runtime_config={},
            )
            loader = adapted_owner.load_fada_adapted_checkpoint
    else:
        config = dict(history_len=3, pred_horizon=2, planner_d_model=16, planner_nhead=2,
                      planner_num_layers=1, planner_dim_feedforward=32, planner_dropout=0.0,
                      idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1,
                      idm_decoder_num_layers=1, idm_dim_feedforward=32, idm_dropout=0.0)
        policy = build_official_fada_student(config).eval()
        identity = dict(upstream_revision=OFFICIAL_FADA_SOURCE_REVISION,
                        expert_checkpoint=str(tmp_path / 'expert.pt'),
                        expert_checkpoint_sha256='a' * 64, experiment_config_sha256='b' * 64)
        terms = ['base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity']
        preprocessing = validate_official_fada_preprocessing(dict(
            term_order=terms, term_scale=dict.fromkeys(terms, 1.0),
            term_noise=dict.fromkeys(terms, 0.0), noise_enabled=False,
        ))
        if kind == 'official':
            payload = dict(format='unilab_official_fada_student', schema_version=2,
                           model_config=policy.official_model_config,
                           training_config=canonical_official_training_config(),
                           preprocessing=preprocessing, source_identity=identity,
                           student_state_dict=policy.state_dict(),
                           idm_optimizer_state_dict={}, planner_optimizer_state_dict={},
                           update_count=0, completed_iterations=0, samples_seen=0,
                           rollout_step_count=0, online_buffer_path=None)
            loader = source_owner.load_official_fada_playback_checkpoint
        else:
            control = dict(source_identity=identity, joint_names=[f'joint_{i}' for i in range(29)],
                           default_joint_pos=[0.0] * 29, action_scale=[1.0] * 29,
                           stiffness=[10.0] * 29, damping=[1.0] * 29, clip_actions=None,
                           sim_dt=1 / 150, ctrl_dt=0.02, gait_frequency=1.0,
                           gait_clock_mode='continuous', phase_init=None)
            payload = dict(cfg=policy.official_model_config, compact_obs=preprocessing,
                           model_state_dict=policy.state_dict(), obs_dim=64, act_dim=29, cmd_dim=7,
                           unilab_upstream_source=dict(version=1, source_identity=identity,
                                                      control=control,
                                                      source_profile={'kind': 'unilab_oracle', 'version': 1}))
            loader = load_upstream_fada_checkpoint
        torch.save(payload, path)
    expected = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    return path, loader, expected


@pytest.mark.parametrize('dispatch', [False, True])
def test_load_once_on_cpu_then_transfer_exact_weights(checkpoint_case, dispatch, monkeypatch):
    path, direct_loader, expected = checkpoint_case
    real_load, real_to = torch.load, torch.nn.Module.to
    reads, transfers = [], []

    def read_once(file, *args, **kwargs):
        reads.append((file, kwargs))
        if len(reads) != 1:
            raise AssertionError('Deployment reopened the checkpoint')
        assert kwargs['map_location'] == 'cpu'
        assert kwargs['weights_only'] is True
        return real_load(file, *args, **kwargs)

    def transfer(module, *args, **kwargs):
        if args and str(args[0]) == 'meta':
            state = module.state_dict()
            assert set(state) == set(expected)
            for name, value in state.items():
                assert value.device.type == 'cpu'
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            transfers.append(module)
        return real_to(module, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', read_once)
    monkeypatch.setattr(torch.nn.Module, 'to', transfer)
    loader = adapted_owner.load_fada_deployable_policy_checkpoint if dispatch else direct_loader
    loaded = loader(path, device='meta')
    assert len(reads) == 1
    assert transfers == [loaded.policy]
    assert loaded.policy.training is False
    assert all(p.device.type == 'meta' for p in loaded.policy.parameters())
    # The checkpoint remains CPU-owned; only the reconstructed model is transferred.
    def check_cpu(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == 'cpu'
        elif isinstance(value, dict):
            for item in value.values():
                check_cpu(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_cpu(item)
    check_cpu(loaded.checkpoint)


def test_dispatch_preserves_strict_weight_validation(checkpoint_case):
    path, _, _ = checkpoint_case
    payload = torch.load(path, map_location='cpu', weights_only=True)
    key = next(k for k in ('policy_state_dict', 'student_state_dict', 'model_state_dict', 'planner_state_dict') if k in payload)
    payload[key].pop(next(iter(payload[key])))
    torch.save(payload, path)
    with pytest.raises((RuntimeError, ValueError), match="Missing key|incompatible"):
        adapted_owner.load_fada_deployable_policy_checkpoint(path)


def test_real_batch_dry_run_has_no_runtime_imports(tmp_path):
    output = tmp_path / 'must-not-exist'
    code = r'''
import importlib.abc, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'mediapy', 'matplotlib'}:
            raise AssertionError('Runtime imported by parent: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from unilab.visualization import experiment_ablation, experiment_sensitivity
sys.argv = ['scripts/run_oat_ablation.py', '--output-dir', sys.argv[1], '--dry-run']
runpy.run_path('scripts/run_oat_ablation.py', run_name='__main__')
'''
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    result = subprocess.run([sys.executable, '-c', code, str(output)], cwd=ROOT,
                            env=environment, capture_output=True, text=True, check=True)
    assert 'Planned trials: 160' in result.stdout
    assert not output.exists()
