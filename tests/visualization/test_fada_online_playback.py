"""Behavior checks for the official fixed-label FADA streaming session."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    fada_adapter_parameters,
)
from unilab.algos.torch.distill.fada.checkpoint import LoadedFADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.model import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
)
from unilab.algos.torch.distill.fada.online_data import FADATrajectoryPair
from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.algos.torch.distill.fada.online_updater import build_expert_samples
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import FADAPlaybackSession, RslRlPlaybackConfig


OBS_DIM = 66
ACTION_DIM = 29
COMMAND_DIM = 3
HISTORY = 4
HORIZON = 6


@pytest.fixture
def arch_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        command_dim=COMMAND_DIM,
        observation_contract="g1_fada_state_v2",
        history_length=HISTORY,
        prediction_horizon=HORIZON,
        hidden_dim=32,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=64,
        dropout=0.0,
    )


@pytest.fixture
def policy(arch_config: FADAArchitectureConfig) -> FADAPlannerIDMPolicy:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return FADAPlannerIDMPolicy(arch_config).eval()


def _raw_obs(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((1, 98), dtype=np.float32)


def _command() -> np.ndarray:
    return np.array([[0.4, 0.0, 0.0]], dtype=np.float32)


class _FakeActionSpace:
    shape = (ACTION_DIM,)
    low = -np.ones(ACTION_DIM, dtype=np.float32)
    high = np.ones(ACTION_DIM, dtype=np.float32)


class _FakeEnv:
    """Single-env fake honoring the FADA playback env contract."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.num_envs = 1
        self.action_space = _FakeActionSpace()
        self.cfg = SimpleNamespace(
            ctrl_dt=0.02, control_config=SimpleNamespace(action_scale=0.5)
        )
        self._done = np.zeros(1, dtype=bool)
        self.closed = False
        self.action_offset = 0.0
        self.state = SimpleNamespace(
            info={
                "commands": _command(),
                "current_actions": np.zeros((1, ACTION_DIM), dtype=np.float32),
            }
        )

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset = bool(enabled)

    def set_next_done(self, done: bool) -> None:
        self._done = np.array([done], dtype=bool)

    def close(self):
        self.closed = True


class _FakeWrappedEnv:
    """Raw wrapped-env contract consumed by FADAPlaybackSession."""

    def __init__(self, env: _FakeEnv, rng: np.random.Generator) -> None:
        self.env = env
        self._rng = rng
        self._obs = _raw_obs(rng)
        self.num_obs = 98

    def reset(self):
        self._obs = _raw_obs(self._rng)
        self.env.set_next_done(False)
        self.env.state.info["commands"] = _command()
        self.env.state.info["current_actions"] = np.zeros((1, ACTION_DIM), np.float32)
        return {"actor": torch.from_numpy(self._obs)}, {}

    def step(self, actions):
        np.copyto(
            self.env.state.info["current_actions"],
            actions.detach().cpu().numpy() + self.env.action_offset,
        )
        np.copyto(self._obs, _raw_obs(self._rng))
        reward = torch.zeros(1)
        return {"actor": torch.from_numpy(self._obs)}, reward, torch.from_numpy(self.env._done.copy()), {}

    def get_observations(self):
        return {"actor": torch.from_numpy(self._obs)}

    def close(self):
        self.env.close()


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    path = tmp_path / "fada.pt"
    path.write_bytes(b"checkpoint")
    return path


def _deps(policy: FADAPlannerIDMPolicy, env: _FakeEnv, wrapped: _FakeWrappedEnv, ckpt: Path):
    return {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: ckpt,
        "load_fada_policy": lambda checkpoint, device: LoadedFADAPlannerIDMPolicy(
            policy=policy, checkpoint={"completed_steps": 1}
        ),
        "create_env": lambda *a, **k: env,
        "wrapper_cls": lambda env, device, policy_obs_mode: wrapped,
        "build_env_cfg_override": lambda cfg: {},
    }


def _playback_cfg(ckpt: Path) -> RslRlPlaybackConfig:
    return RslRlPlaybackConfig(
        task="t",
        load_run="-1",
        checkpoint=None,
        checkpoint_path=str(ckpt),
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name="distill",
        log_root=None,
        num_envs=1,
    )


def _cfg(online: dict | None):
    from omegaconf import OmegaConf

    node = {"training": {"task_name": "T"}}
    if online is not None:
        node["online_adaptation"] = online
    return OmegaConf.create(node)


def _make_session(policy, env, wrapped, online, ckpt: Path):
    return create_fada_playback_session(
        playback_cfg=_playback_cfg(ckpt),
        cfg=_cfg(online),
        root_dir="/tmp",
        device="cpu",
        deps=_deps(policy, env, wrapped, ckpt),
        log=lambda msg: None,
    )


_ONLINE = {
    "enabled": True,
    "window_size": 3,
    "warmup_steps": 0,
    "gamma": 0.05,
    "sigma": 1e-4,
    "update_interval": 2,
    "learning_rate": 1e-3,
    "max_grad_norm": 1.0,
    "lora_rank": 2,
    "lora_alpha": 4.0,
    "lora_dropout": 0.0,
}


@pytest.mark.parametrize("mode", [None, "trajectory_base", "action_base", "trajectory_lora"])
def test_disabled_preserves_checkpoint_actions_and_parameters(policy, checkpoint_file, mode):
    env = _FakeEnv(np.random.default_rng(0))
    wrapped = _FakeWrappedEnv(env, env._rng)
    online = None if mode is None else dict(_ONLINE, enabled=False, mode=mode)
    session, _, _ = _make_session(policy, env, wrapped, online, checkpoint_file)
    session.reset()
    projected = session.controller.project_observation(session.obs)
    with torch.no_grad():
        expected = policy(
            projected[:, None].repeat(1, HISTORY, 1),
            torch.zeros(1, HISTORY, ACTION_DIM),
            torch.from_numpy(_command()),
        ).action
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    session.step_once()
    torch.testing.assert_close(session.actions, expected)
    for _ in range(3):
        session.step_once()
        assert session.last_update_stats is None
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert list(fada_adapter_parameters(policy)) == []


def test_frozen_base_survives_real_online_update_and_reset(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(6))
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    controller = session.controller
    base = controller.base_idm
    assert all(not module.training for module in base.modules())
    assert all(not parameter.requires_grad for parameter in base.parameters())
    base_tensors = tuple(base.parameters()) + tuple(base.buffers())
    online_tensors = tuple(policy.idm.parameters()) + tuple(policy.idm.buffers())
    assert {
        tensor.untyped_storage().data_ptr() for tensor in base_tensors if tensor.numel()
    }.isdisjoint({
        tensor.untyped_storage().data_ptr() for tensor in online_tensors if tensor.numel()
    })
    optimizer_ids = {
        id(parameter)
        for group in controller.updater.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimizer_ids for parameter in base.parameters())
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    adapters = tuple(fada_adapter_parameters(policy))
    before = tuple(parameter.detach().clone() for parameter in adapters)
    realized = []
    for _ in range(HORIZON):
        session.step_once()
        realized.append(controller.project_observation(session.obs).clone())
        assert session.last_update_stats is None
    pair = controller.collector.get_window()[0]
    snapshot = pair.snapshot
    inputs = (snapshot.observation_history, snapshot.action_history, snapshot.nominal_future)
    with torch.no_grad():
        expected = base(*inputs).clone()
        torch.testing.assert_close(policy.idm(*inputs), expected)
    torch.testing.assert_close(pair.realized_future, torch.stack(realized, dim=1))
    for _ in range(2):  # K=6, T=3, interval=2: first full-window tick is step 8.
        session.step_once()
    assert session.last_update_stats["num_samples"] == 3
    assert any(not torch.equal(parameter, old) for parameter, old in zip(adapters, before))
    with torch.no_grad():
        torch.testing.assert_close(base(*inputs), expected, atol=0, rtol=0)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, base_state[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in base.parameters())
    optimizer = controller.updater.optimizer
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    learned = tuple(parameter.detach().clone() for parameter in adapters)
    adam_steps = [state["step"].clone() for state in optimizer.state.values()]
    session.reset()
    assert controller.base_idm is base and controller.updater.optimizer is optimizer
    for parameter, old in zip(adapters, learned):
        torch.testing.assert_close(parameter, old, atol=0, rtol=0)
    for state, old in zip(optimizer.state.values(), adam_steps):
        torch.testing.assert_close(state["step"], old, atol=0, rtol=0)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(base(*inputs), expected, atol=0, rtol=0)


def test_online_uses_full_realized_future_and_keeps_expert_labels_static(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(2))
    env.action_offset = 7.0  # Execution-side action echoes are not the regression target.
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    controller = session.controller
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    states, actions = [], []
    for _ in range(HORIZON):
        session.step_once()
        assert session.last_update_stats is None
        states.append(controller.project_observation(session.obs).clone())
        actions.append(session.actions.clone())
    first = controller.collector.get_window()[0]
    torch.testing.assert_close(first.realized_future, torch.stack(states, dim=1))
    torch.testing.assert_close(first.snapshot.action_snapshot, actions[0])
    samples = build_expert_samples(controller.base_idm, (first,), gamma=controller.gamma)
    label = samples[0].target_action.clone()
    matched = FADATrajectoryPair(
        first.snapshot, first.snapshot.nominal_future, first.episode_id, first.timestep
    )
    matched_sample, = build_expert_samples(controller.base_idm, (matched,), gamma=controller.gamma)
    torch.testing.assert_close(matched_sample.target_action, actions[0], atol=0, rtol=0)
    assert not torch.allclose(label, actions[0] + env.action_offset)
    session.step_once()
    assert [pair.timestep for pair in controller.collector.get_window()] == [0, 1]
    assert not controller.collector.ready and session.last_update_stats is None
    session.step_once()
    stats = session.last_update_stats
    assert stats["num_samples"] == 3
    assert stats["loss"] == pytest.approx(stats["regression_loss"] + stats["anchor_loss"])
    assert controller.collector.get_window() == ()
    controller.updater.compute_loss(samples)
    torch.testing.assert_close(samples[0].target_action, label, atol=0, rtol=0)
    rebuilt, = build_expert_samples(controller.base_idm, (first,), gamma=controller.gamma)
    torch.testing.assert_close(rebuilt.target_action, label, atol=0, rtol=0)
    assert not samples[0].target_action.requires_grad
    changed = {name for name, p in policy.named_parameters() if not torch.equal(p, before[name])}
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert changed and changed <= trainable
    assert all(name.startswith("idm.") for name in trainable)
    assert not policy.idm.training


@pytest.mark.parametrize("warmup", [40, 17])
def test_fresh_batches_wait_for_warmup_and_preserve_learning_across_consumption(
    policy, checkpoint_file, warmup
):
    env = _FakeEnv(np.random.default_rng(4))
    wrapped = _FakeWrappedEnv(env, env._rng)
    online = dict(_ONLINE, window_size=10, warmup_steps=warmup, update_interval=10)
    session, _, _ = _make_session(policy, env, wrapped, online, checkpoint_file)
    session.reset()
    controller = session.controller
    optimizer = controller.updater.optimizer
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    first_update = ((warmup + 9) // 10) * 10
    expected_ticks = [first_update + offset for offset in (0, 10, 20)]
    ticks = []
    episode = None
    actions = []
    for step in range(1, expected_ticks[-1] + 1):
        session.step_once()
        actions.append(session.actions.detach().clone())
        samples = controller.collector.get_window()
        if step in expected_ticks:
            ticks.append(step)
            assert session.last_update_stats["num_samples"] == 10
            assert samples == ()
            assert controller.updater.optimizer is optimizer
            assert {int(state["step"].item()) for state in optimizer.state.values()} == {len(ticks)}
        else:
            assert session.last_update_stats is None
            if step == first_update - 1:
                last_origin = step - HORIZON
                assert [sample.timestep for sample in samples] == list(range(last_origin - 9, last_origin + 1))
                episode = samples[-1].episode_id
            if step - 1 in expected_ticks:
                assert len(samples) == 1
                origin = step - HORIZON
                assert (samples[0].episode_id, samples[0].timestep) == (episode, origin)
                torch.testing.assert_close(samples[0].snapshot.action_history[:, -1], actions[origin - 1])
        for name, value in controller.updater.theta_base.items():
            torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    assert ticks == expected_ticks
    learned = {name: p.detach().clone() for name, p in policy.named_parameters()}
    session.reset()
    for step in range(1, first_update):
        session.step_once()
        assert session.last_update_stats is None
        assert {int(state["step"].item()) for state in optimizer.state.values()} == {3}
        if step == HORIZON:
            sample = controller.collector.get_window()[0]
            assert (sample.episode_id, sample.timestep) == (episode + 1, 0)
            torch.testing.assert_close(
                sample.snapshot.action_history, torch.zeros_like(sample.snapshot.action_history)
            )
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, learned[name], atol=0, rtol=0)
    session.step_once()
    assert session.last_update_stats["num_samples"] == 10
    assert controller.collector.get_window() == ()
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {4}


def test_failed_real_update_does_not_consume_batch(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(5))
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    for _ in range(7):
        session.step_once()
    controller = session.controller
    # Corrupt stored evidence; the real expert forward must fail without consuming the batch.
    controller.collector.get_window()[-1].realized_future.fill_(float("nan"))
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    with pytest.raises(ValueError, match="finite"):
        session.step_once()
    assert [sample.timestep for sample in controller.collector.get_window()] == [0, 1, 2]
    assert controller.updater.optimizer.state == {}
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("autoreset", [True, False])
def test_terminal_and_manual_reset_keep_learning_state_but_isolate_samples(
    policy, checkpoint_file, autoreset
):
    import copy

    env = _FakeEnv(np.random.default_rng(3))
    wrapped = _FakeWrappedEnv(env, env._rng)
    online = dict(_ONLINE, window_size=2, update_interval=1)
    session, _, _ = _make_session(policy, env, wrapped, online, checkpoint_file)
    session.reset()
    session.set_autoreset(autoreset)
    for _ in range(HORIZON):
        session.step_once()
    old_episode = session.controller.collector.get_window()[-1].episode_id
    session.step_once()
    controller = session.controller
    params = {name: p.detach().clone() for name, p in policy.named_parameters()}
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    optimizer = controller.updater.optimizer
    adam_state = copy.deepcopy(optimizer.state_dict()["state"])
    assert adam_state
    env.set_next_done(True)
    session.step_once()
    assert session.last_update_stats is None
    assert controller.collector.get_window() == ()
    if autoreset:
        env.set_next_done(False)
    else:
        session.step_once()  # Physics still advances; collection must remain suspended.
        assert controller.collector.get_window() == ()
        assert session.last_update_stats is None
        session.reset()
    for _ in range(HORIZON):
        session.step_once()
    sample = controller.collector.get_window()[0]
    assert (sample.episode_id, sample.timestep) == (old_episode + 1, 0)
    torch.testing.assert_close(
        sample.snapshot.action_history, torch.zeros_like(sample.snapshot.action_history)
    )
    torch.testing.assert_close(
        sample.snapshot.observation_history,
        sample.snapshot.observation_history[:, -1:].repeat(1, HISTORY, 1),
    )
    assert controller.updater.optimizer is optimizer
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, adam_state[key][field], atol=0, rtol=0)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, params[name], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["wrapper", "io", "adapted"])
def test_failed_online_construction_releases_environment(policy, checkpoint_file, failure):
    env = _FakeEnv(np.random.default_rng(4))
    wrapped = _FakeWrappedEnv(env, env._rng)
    deps = _deps(policy, env, wrapped, checkpoint_file)
    if failure == "wrapper":
        def fail_wrapper(*args, **kwargs):
            raise ValueError("wrapper construction failed")
        deps["wrapper_cls"] = fail_wrapper
    elif failure == "io":
        wrapped.num_obs = 97
    else:
        from unilab.algos.torch.distill.fada.adaptation import inject_fada_idm_lora
        inject_fada_idm_lora(policy, FADALoRAConfig(rank=2, alpha=4.0))
    with pytest.raises(ValueError):
        create_fada_playback_session(
            playback_cfg=_playback_cfg(checkpoint_file), cfg=_cfg(dict(_ONLINE)),
            root_dir="/tmp", device="cpu", deps=deps, log=lambda msg: None,
        )
    assert env.closed


@pytest.mark.parametrize("failure", ["multiple_envs", "zero_actions", "old_config", "mode", "fada_anchor"])
def test_invalid_online_request_fails_before_environment_creation(policy, checkpoint_file, failure):
    env = _FakeEnv(np.random.default_rng(5))
    wrapped = _FakeWrappedEnv(env, env._rng)
    playback = _playback_cfg(checkpoint_file)
    online = dict(_ONLINE)
    if failure == "multiple_envs":
        object.__setattr__(playback, "num_envs", 2)
    elif failure == "zero_actions":
        object.__setattr__(playback, "action_mode", "zero")
    elif failure == "old_config":
        online["lambda_forget"] = 0.9
    elif failure == "fada_anchor":
        online["mode"] = "fada_ttt"
    else:
        online["mode"] = "invalid"
    deps = _deps(policy, env, wrapped, checkpoint_file)
    def forbidden_env(*args, **kwargs):
        pytest.fail("invalid online request must not create an environment")
    deps["create_env"] = forbidden_env
    if failure in ("mode", "fada_anchor"):
        deps["load_fada_policy"] = forbidden_env
    with pytest.raises((ValueError, TypeError)):
        create_fada_playback_session(
            playback_cfg=playback, cfg=_cfg(online), root_dir="/tmp",
            device="cpu", deps=deps, log=lambda msg: None,
        )


def _mode_session(policy, mode, checkpoint_file=None):
    env = _FakeEnv(np.random.default_rng(6))
    wrapped = _FakeWrappedEnv(env, env._rng)
    sigma = 0.0 if mode == "fada_ttt" else 1e-4
    if checkpoint_file is not None:
        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).resolve().parents[2] / "conf" / "distill"
        overrides = ["task=g1_walk_flat/mujoco"] + [
            f"online_adaptation.{key}={str(value).lower()}"
            for key, value in dict(_ONLINE, sigma=sigma).items()
        ]
        overrides.append(f"online_adaptation.mode={mode}")
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=overrides)
        session, _, _ = create_fada_playback_session(
            playback_cfg=_playback_cfg(checkpoint_file), cfg=cfg, root_dir=config_dir.parents[1],
            device="cpu", deps=_deps(policy, env, wrapped, checkpoint_file), log=print,
        )
        session.reset()
        return session
    controller = OnlineAdaptationController(
        policy, FADALoRAConfig(rank=2, alpha=4.0, dropout=0.0),
        window_size=3, warmup_steps=0, update_interval=2,
        gamma=0.05, sigma=sigma, lr=1e-3, mode=mode,
    )
    session = FADAPlaybackSession(
        env=env, wrapped_env=wrapped, device="cpu", action_mode="policy",
        controller=controller, num_envs=1,
    )
    session.reset()
    return session


@pytest.mark.parametrize("mode", ["trajectory_base", "action_base", "trajectory_lora", "fada_ttt"])
@pytest.mark.parametrize("via_factory", [False, True])
def test_controller_mode_updates_use_preupdate_targets_and_retain_learning(
    policy, monkeypatch, mode, checkpoint_file, via_factory,
):
    import copy

    session = _mode_session(policy, mode, checkpoint_file if via_factory else None)
    controller = session.controller
    base_before = {} if controller.base_idm is None else {
        name: value.clone() for name, value in controller.base_idm.state_dict().items()
    }
    if mode == "fada_ttt":
        assert controller.base_idm is None
    planner_before = {name: value.clone() for name, value in policy.planner.state_dict().items()}
    # Separate the reference candidates before checking routing; zero LoRA
    # initialization would otherwise hide an incorrect reference selection.
    with torch.no_grad():
        for parameter in fada_adapter_parameters(policy):
            parameter.add_(0.03)
    real_update = controller.updater.update
    completed = []

    def verify_then_update(samples):
        reference = policy.idm if mode == "trajectory_lora" else controller.base_idm
        expected_targets = []
        fixed_targets = tuple(sample.target_action.clone() for sample in samples)
        with torch.no_grad():
            for sample in samples:
                snapshot = sample.trajectory.snapshot
                inputs = (snapshot.observation_history, snapshot.action_history)
                if mode == "fada_ttt":
                    target = snapshot.action_snapshot
                else:
                    realized = reference(*inputs, sample.trajectory.realized_future)[:, 0]
                    nominal = (
                        snapshot.action_snapshot if mode == "action_base"
                        else reference(*inputs, snapshot.nominal_future)[:, 0]
                    )
                    target = snapshot.action_snapshot + 0.05 * (nominal - realized)
                torch.testing.assert_close(sample.target_action, target, rtol=1e-6, atol=1e-7)
                assert not sample.target_action.requires_grad and sample.target_action.grad_fn is None
                expected_targets.append(target.clone())
            snapshots = tuple(sample.trajectory.snapshot for sample in samples)
            predictions = policy.idm(
                torch.cat([snapshot.observation_history for snapshot in snapshots]),
                torch.cat([snapshot.action_history for snapshot in snapshots]),
                torch.cat([
                    sample.trajectory.realized_future if mode == "fada_ttt"
                    else sample.trajectory.snapshot.nominal_future for sample in samples
                ]),
            )[:, 0]
            expected_regression = (
                torch.nn.functional.mse_loss(predictions, torch.cat(expected_targets))
                if mode == "fada_ttt" else
                0.5 * (predictions - torch.cat(expected_targets)).square().sum(dim=-1).mean()
            )
        before = tuple(parameter.detach().clone() for parameter in fada_adapter_parameters(policy))
        stats = real_update(samples)
        assert stats["regression_loss"] == pytest.approx(float(expected_regression), rel=1e-6, abs=1e-8)
        if mode == "fada_ttt":
            assert stats["anchor_loss"] == 0.0
        assert any(not torch.equal(old, new) for old, new in zip(before, fada_adapter_parameters(policy)))
        for sample, target in zip(samples, fixed_targets):
            torch.testing.assert_close(sample.target_action, target, rtol=0, atol=0)
        completed.append(stats)
        return stats

    monkeypatch.setattr(controller.updater, "update", verify_then_update)
    for _ in range(HORIZON + 6):
        session.step_once()
    assert len(completed) == 2
    if controller.base_idm is not None:
        for name, value in controller.base_idm.state_dict().items():
            torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    for name, value in policy.planner.state_dict().items():
        torch.testing.assert_close(value, planner_before[name], rtol=0, atol=0)
    if controller.base_idm is not None:
        assert all(parameter.grad is None for parameter in controller.base_idm.parameters())
    assert all(parameter.grad is None for parameter in policy.planner.parameters())
    learned = {name: value.clone() for name, value in policy.state_dict().items()}
    optimizer = controller.updater.optimizer
    adam_before = copy.deepcopy(optimizer.state_dict()["state"])
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    session.reset()
    assert controller.collector.get_window() == ()
    assert controller.updater.optimizer is optimizer
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, learned[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, adam_before[key][name], rtol=0, atol=0)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], rtol=0, atol=0)
    for _ in range(HORIZON - 1):
        session.step_once()
        assert session.last_update_stats is None


@pytest.mark.parametrize("mode", ["trajectory_base", "action_base", "trajectory_lora", "fada_ttt"])
def test_controller_mode_failed_labels_preserve_window_and_parameters(policy, mode):
    session = _mode_session(policy, mode)
    for _ in range(HORIZON + 1):
        session.step_once()
    controller = session.controller
    controller.collector.get_window()[-1].realized_future.fill_(float("nan"))
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    with pytest.raises(ValueError, match="finite"):
        session.step_once()
    assert [sample.timestep for sample in controller.collector.get_window()] == [0, 1, 2]
    assert controller.updater.optimizer.state == {}
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_controller_mode_invalid_rejects_before_policy_mutation(policy):
    parameters = dict(policy.named_parameters())
    with pytest.raises(ValueError, match="unsupported online target mode"):
        OnlineAdaptationController(policy, FADALoRAConfig(rank=2, alpha=4.0), mode="invalid")
    current = dict(policy.named_parameters())
    assert current.keys() == parameters.keys()
    assert all(current[name] is parameter for name, parameter in parameters.items())


@pytest.fixture
def official_controller_inputs():
    from unilab.algos.torch.distill.fada.official_model import build_official_fada_student
    from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
    from unilab.algos.torch.distill.fada.runtime_policy import OfficialFADAPlaybackProfile

    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = build_official_fada_student({
            "history_len": 3, "pred_horizon": 2,
            "planner_d_model": 8, "planner_nhead": 2, "planner_num_layers": 1,
            "planner_dim_feedforward": 16, "planner_dropout": 0.0,
            "idm_d_model": 8, "idm_nhead": 2, "idm_encoder_num_layers": 1,
            "idm_decoder_num_layers": 1, "idm_dim_feedforward": 16, "idm_dropout": 0.0,
            "idm_use_current_command_for_history": True,
        }).eval()
    scales = {"base_ang_vel": 0.25, "dof_pos": 2.0, "dof_vel": 0.05, "projected_gravity": 1.5}
    policy.official_preprocessing = {
        "term_order": list(scales), "noise_enabled": False, "term_scale": scales,
        "term_noise": {name: 0.0 for name in scales},
    }
    # Explicit synthetic source profile, never a deployment default.
    source_identity = {"test_run": "synthetic-official-controller"}
    names = tuple(f"test_joint_{index}" for index in range(29))
    defaults = torch.linspace(-0.1, 0.1, 29)
    profile = OfficialFADAPlaybackProfile.from_mapping({
        "source_identity": source_identity, "joint_names": names,
        "default_joint_pos": defaults.tolist(), "action_scale": [0.25] * 29,
        "stiffness": [20.0] * 29, "damping": [0.5] * 29,
        "clip_actions": 0.05, "ctrl_dt": 0.02, "gait_frequency": 1.5,
        "gait_clock_mode": "continuous", "phase_init": [0.0, float(np.pi)],
    }, source_identity=source_identity)
    playback = OfficialFADAPlaybackController(
        policy, profile=profile, target_joint_names=names[::-1], device="cpu",
    )

    def physical(step):
        joints = torch.arange(29, dtype=torch.float32).unsqueeze(0)
        return {
            "base_ang_vel": torch.tensor([[0.02 * step, -0.1, 0.3]]),
            "dof_pos": joints * 0.01 + step * 0.03,
            "dof_vel": joints * 0.02 - step * 0.04,
            "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        }

    def expected_compact(step):
        values = physical(step)
        return torch.cat((
            values["base_ang_vel"] * 0.25,
            (values["dof_pos"].flip(1) - defaults) * 2.0,
            values["dof_vel"].flip(1) * 0.05,
            values["projected_gravity"] * 1.5,
        ), dim=1)

    return policy, playback, physical, expected_compact


@pytest.mark.parametrize("mode", ["trajectory_base", "action_base", "trajectory_lora", "fada_ttt"])
def test_official_controller_modes_complete_two_phase_aligned_updates(official_controller_inputs, monkeypatch, mode):
    policy, playback, physical, expected_compact = official_controller_inputs
    controller = OnlineAdaptationController(
        policy, FADALoRAConfig(rank=2, alpha=4.0, dropout=0.0), playback=playback,
        window_size=3, warmup_steps=0, update_interval=2, gamma=0.05, lr=1e-3, mode=mode,
        sigma=0.0 if mode == "fada_ttt" else 1e-4,
    )
    base_before = {} if controller.base_idm is None else {
        name: value.clone() for name, value in controller.base_idm.state_dict().items()
    }
    if mode == "fada_ttt":
        assert controller.base_idm is None
    planner_before = {name: value.clone() for name, value in policy.planner.state_dict().items()}
    with torch.no_grad():
        for parameter in fada_adapter_parameters(policy):
            parameter.add_(0.03)
    commands, actions, updates = [], [], []
    clipped = []
    real_update = controller.updater.update

    def verify_then_update(samples):
        snapshots = tuple(sample.trajectory.snapshot for sample in samples)
        for sample, snapshot in zip(samples, snapshots):
            origin = sample.trajectory.timestep
            indices = list(range(max(0, origin - 2), origin + 1))
            expected_mask = torch.tensor([[False] * (3 - len(indices)) + [True] * len(indices)])
            torch.testing.assert_close(snapshot.history_valid_mask, expected_mask)
            torch.testing.assert_close(snapshot.command, commands[origin], rtol=0, atol=0)
            torch.testing.assert_close(snapshot.action_snapshot, actions[origin], rtol=0, atol=0)
            torch.testing.assert_close(
                snapshot.observation_history[:, -len(indices):],
                torch.stack([expected_compact(index) for index in indices], dim=1),
            )
            expected_actions = [torch.zeros(1, 29) if index == 0 else actions[index - 1] for index in indices]
            torch.testing.assert_close(
                snapshot.action_history[:, -len(indices):], torch.stack(expected_actions, dim=1),
            )
            assert not bool(snapshot.observation_history[~expected_mask].any())
            assert not bool(snapshot.action_history[~expected_mask].any())
            torch.testing.assert_close(sample.trajectory.realized_future, torch.stack((
                expected_compact(origin + 1), expected_compact(origin + 2),
            ), dim=1))
        observations = torch.cat([snapshot.observation_history for snapshot in snapshots])
        history_actions = torch.cat([snapshot.action_history for snapshot in snapshots])
        command = torch.cat([snapshot.command for snapshot in snapshots])
        mask = torch.cat([snapshot.history_valid_mask for snapshot in snapshots])
        nominal = torch.cat([snapshot.nominal_future for snapshot in snapshots])
        realized = torch.cat([sample.trajectory.realized_future for sample in samples])
        saved_action = torch.cat([snapshot.action_snapshot for snapshot in snapshots])
        reference = policy.idm if mode == "trajectory_lora" else controller.base_idm
        with torch.no_grad():
            prediction_future = realized if mode == "fada_ttt" else nominal
            predicted = policy.idm(observations, history_actions, command, prediction_future, history_valid_mask=mask)[:, 0]
            if mode == "fada_ttt":
                expected_target = saved_action
                expected_regression = torch.nn.functional.mse_loss(predicted, expected_target)
            else:
                realized_action = reference(observations, history_actions, command, realized, history_valid_mask=mask)[:, 0]
                nominal_action = saved_action if mode == "action_base" else reference(
                    observations, history_actions, command, nominal, history_valid_mask=mask,
                )[:, 0]
                expected_target = saved_action + 0.05 * (nominal_action - realized_action)
                expected_regression = 0.5 * (predicted - expected_target).square().sum(dim=-1).mean()
        target = torch.cat([sample.target_action for sample in samples])
        torch.testing.assert_close(target, expected_target, rtol=1e-6, atol=1e-7)
        assert all(not sample.target_action.requires_grad and sample.target_action.grad_fn is None for sample in samples)
        fixed = tuple(sample.target_action.clone() for sample in samples)
        before = tuple(parameter.detach().clone() for parameter in fada_adapter_parameters(policy))
        stats = real_update(samples)
        assert stats["regression_loss"] == pytest.approx(float(expected_regression), rel=1e-6, abs=1e-8)
        if mode == "fada_ttt":
            assert stats["anchor_loss"] == 0.0
        assert any(not torch.equal(old, new) for old, new in zip(before, fada_adapter_parameters(policy)))
        for sample, saved in zip(samples, fixed):
            torch.testing.assert_close(sample.target_action, saved, rtol=0, atol=0)
        updates.append(stats)
        return stats

    monkeypatch.setattr(controller.updater, "update", verify_then_update)
    for step in range(8):
        phase = torch.tensor([[0.2 * step, 0.2 * step + np.pi]])
        command = torch.cat((torch.tensor([[0.3, 0.0, 0.2]]), phase.sin(), phase.cos()), dim=1)
        commands.append(command.clone())
        action = controller.act(physical(step), command)
        actions.append(action.clone())
        clipped.append(bool((action.abs() == playback.profile.clip_actions).any()))
        stats = controller.on_env_step(
            executed_action=action + 5.0, next_observation=physical(step + 1),
        )
        assert (stats is not None) == (step in (3, 7))
    assert len(updates) == 2
    assert any(clipped)
    if controller.base_idm is not None:
        for name, value in controller.base_idm.state_dict().items():
            torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    for name, value in policy.planner.state_dict().items():
        torch.testing.assert_close(value, planner_before[name], rtol=0, atol=0)
    if controller.base_idm is not None:
        assert all(parameter.grad is None for parameter in controller.base_idm.parameters())
    assert all(parameter.grad is None for parameter in policy.planner.parameters())


@pytest.mark.parametrize("family", ["legacy", "official"])
def test_fada_ttt_learns_realized_future_with_original_mse(policy, official_controller_inputs, family):
    from dataclasses import replace

    from unilab.algos.torch.distill.fada.online_updater import build_fada_ttt_samples
    from unilab.algos.torch.distill.fada.runtime_policy import invoke_fada_idm

    if family == "official":
        policy, playback, physical, _ = official_controller_inputs
        observation = physical(0)
        command = torch.tensor([[0.3, 0.0, 0.2, 0.0, 0.0, 1.0, -1.0]])
    else:
        playback = None
        observation = {"actor": torch.zeros(1, 98)}
        command = torch.tensor([[0.3, 0.0, 0.2]])
    controller = OnlineAdaptationController(
        policy, FADALoRAConfig(rank=2, alpha=4.0, dropout=0.0), playback=playback,
        sigma=0.0, gamma=123.0, mode="fada_ttt",
    )
    _, snapshot = controller.playback.act_projected_with_snapshot(
        controller.project_observation(observation), command,
    )
    realized = snapshot.nominal_future + torch.linspace(0.1, 0.9, controller.runtime_spec.obs_dim)
    pair = FADATrajectoryPair(snapshot, realized, 0, 0)
    samples = build_fada_ttt_samples((pair,))
    torch.testing.assert_close(samples[0].target_action, snapshot.action_snapshot, rtol=0, atol=0)
    assert not samples[0].target_action.requires_grad
    with torch.no_grad():
        predicted = invoke_fada_idm(
            policy.idm, controller.runtime_spec, snapshot.observation_history,
            snapshot.action_history, snapshot.command, pair.realized_future,
            history_valid_mask=snapshot.history_valid_mask,
        )[:, 0]
        expected = torch.nn.functional.mse_loss(predicted, snapshot.action_snapshot)
    loss, regression, anchor = controller.updater.compute_loss(samples)
    torch.testing.assert_close(regression, expected, rtol=0, atol=0)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert anchor.item() == 0.0
    changed_nominal = replace(snapshot, nominal_future=snapshot.nominal_future + 100.0)
    nominal_samples = build_fada_ttt_samples((replace(pair, snapshot=changed_nominal),))
    nominal_loss, _, _ = controller.updater.compute_loss(nominal_samples)
    torch.testing.assert_close(nominal_loss, loss, rtol=0, atol=0)
    changed_realized = replace(pair, realized_future=-pair.realized_future.flip(-1) * 2.0)
    realized_loss, _, _ = controller.updater.compute_loss(build_fada_ttt_samples((changed_realized,)))
    assert not torch.isclose(realized_loss, loss, rtol=1e-6, atol=1e-8)
    changed_action = replace(snapshot, action_snapshot=snapshot.action_snapshot + 1.0)
    action_loss, _, _ = controller.updater.compute_loss(
        build_fada_ttt_samples((replace(pair, snapshot=changed_action),)),
    )
    torch.testing.assert_close(
        action_loss, torch.nn.functional.mse_loss(predicted, changed_action.action_snapshot),
        rtol=0, atol=0,
    )
    loss.backward()
    adapters = tuple(fada_adapter_parameters(policy))
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in adapters)
    assert all(parameter.grad is None for parameter in policy.parameters() if not parameter.requires_grad)
