"""Recording timing and status contracts without GPU rendering or physics stepping."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import mujoco as _mujoco
import numpy as np
import pytest
from PIL import Image

from unilab.visualization import fada_recording as recording
from unilab.visualization.fada_rendering import FADADemoRenderConfig

mujoco: Any = _mujoco


class _Session:
    def __init__(self, *, fada=False, stop_step=None, reason="terminated", fail_step=None):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="pelvis" pos="0 0 0.8">'
            '<freejoint/><geom type="sphere" size="0.1" mass="1"/>'
            '</body></worldbody></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.fada, self.stop_step, self.reason, self.fail_step = fada, stop_step, reason, fail_step
        self.autoreset_enabled = fada
        self.checkpoint_path = Path("/runs/mirror/model_50000.pt")
        self.checkpoint_identity = {"iteration": 50000}
        self.command_program = SimpleNamespace(
            ctrl_dt=0.02, total_control_steps=4,
            commands=np.asarray([[0.3, 0.0, 0.1]] * 4),
            reference_points=np.asarray([[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]]),
            segment_index_at_control_step=lambda step: step % 4,
        )
        self.footprint_recorder = None
        self.env = SimpleNamespace(
            cfg=SimpleNamespace(ctrl_dt=0.02, max_episode_seconds=1.0),
            state=None,
            get_base_pos=lambda: self.data.xpos[1:2].copy(),
            get_base_quat=lambda: self.data.xquat[1:2].copy(),
            get_foot_pos=lambda: np.asarray([[[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]]),
            close=lambda: pytest.fail("the recorder must not close the caller's env"),
        )
        self.reset_calls = 0
        if fada:
            self.last_update_stats = None

    def reset(self):
        self.reset_calls += 1
        self.step_count = 0
        self.command_control_steps = np.zeros(1, dtype=np.int64)
        self.finished = False
        self.termination_reason = None
        self.info = {"commands": np.asarray([[0.3, 0.0, 0.1]])}
        self.env.state = SimpleNamespace(terminated=np.zeros(1, bool), truncated=np.zeros(1, bool))
        self.data.qpos[0] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step_once(self):
        if self.finished:
            raise AssertionError("the recorder stepped beyond the terminal frame")
        if self.fail_step == self.step_count + 1:
            raise RuntimeError("inference failed")
        self.step_count += 1
        self.command_control_steps[0] = self.step_count
        done = self.step_count == self.stop_step
        self.env.state.terminated[:] = done and self.reason == "terminated"
        self.env.state.truncated[:] = done and self.reason == "truncated"
        if done and not self.autoreset_enabled:
            self.finished = True
            self.termination_reason = self.reason
        if self.fada:
            self.last_update_stats = {"loss": 0.25} if self.step_count == 1 else None
        self.data.qpos[0] = self.step_count * 0.01
        self.data.time = self.step_count * 0.02
        mujoco.mj_forward(self.model, self.data)

    def physics_state(self):
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        values = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, values, spec)
        return values[None, :]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    objects = SimpleNamespace(renderers=[], writers=[], renderings=[], logs=[], render_fail_step=None)
    original_rendering = recording.FADAPlaybackRendering

    class ObservedRendering(original_rendering):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            objects.renderings.append(self)

    class Renderer:
        def __init__(self, model, *, height, width):
            self.height, self.width, self.closed = height, width, False
            self.scene = mujoco.MjvScene(model, maxgeom=128)
            objects.renderers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def update_scene(self, data, *, camera):
            self.scene.ngeom = 0

        def render(self):
            if objects.session.step_count == objects.render_fail_step:
                raise RuntimeError("render failed")
            return np.full((self.height, self.width, 3), objects.session.step_count, np.uint8)

    class Writer:
        def __init__(self, *args, **kwargs):
            self.frames, self.closed = [], False
            objects.writers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def add_image(self, image):
            self.frames.append(image.copy())

    monkeypatch.setattr(recording, "_load_viewer_model", lambda env, **kw: objects.session.model)
    monkeypatch.setattr(recording, "FADAPlaybackRendering", ObservedRendering)
    monkeypatch.setattr(recording.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(recording.media, "VideoWriter", Writer)

    def run(session, *, duration=0.12, fps=10):
        objects.session = session
        args = SimpleNamespace(
            use_env_visual_model=True, camera_focus_body_name="pelvis",
            camera_distance=8.0, camera_elevation=-70.0, camera_azimuth=90.0,
            camera_follow_body=False, camera_height_offset=0.15,
        )
        config = recording.FADAVideoRecordingConfig(
            output_path=str(tmp_path / "recording.mp4"), duration_s=duration,
            fps=fps, width=1024, height=384, progress_interval_s=1000.0,
        )
        kwargs: dict[str, Literal["fada", "oracle"]] = {} if session.fada else {"policy_kind": "oracle"}
        return recording.record_fada_playback(
            session, args, config, render_config=FADADemoRenderConfig(enabled=True),
            log=objects.logs.append, **kwargs,
        )

    objects.run = run
    objects.frame_steps = lambda: [int(frame[-1, -1, 0]) for frame in objects.writers[-1].frames]
    return objects


@pytest.mark.parametrize(
    "stop_step,fps,reason,expected_frames",
    [(2, 10, "terminated", [0, 2]), (5, 10, "truncated", [0, 5]),
     (1, 100, "terminated", [0, 1, 1])],
)
def test_oracle_stops_with_terminal_frame(harness, tmp_path, stop_step, fps, reason, expected_frames):
    session = _Session(stop_step=stop_step, reason=reason)
    harness.run(session, fps=fps)
    assert session.step_count == stop_step
    assert session.reset_calls == 1
    assert harness.frame_steps() == expected_frames
    assert harness.writers[-1].closed and harness.renderers[-1].closed
    rendering = harness.renderings[-1]
    assert rendering.reset_count == rendering.update_count == 0
    status = rendering.status(session.data)
    assert "mirror/model_50000.pt" in " ".join(status.details)
    assert "50000" in " ".join(status.details)
    assert not any("LoRA" in row or "resets=" in row or "yaw error=" in row for row in status.details)
    assert reason.upper() in status.alert
    assert any(f"reason={reason} actual_steps={stop_step}" in line for line in harness.logs)
    assert any(f"actual_sim_time={stop_step * 0.02:.3f}s" in line for line in harness.logs)
    if stop_step == 2:
        image_path = tmp_path / "oracle_terminal_overlay.png"
        Image.fromarray(harness.writers[-1].frames[-1]).save(image_path)
        print(f"ORACLE_OVERLAY_IMAGE={image_path}")


@pytest.mark.parametrize("fada", [False, True])
def test_one_cycle_does_not_start_the_next_cycle(harness, fada):
    session = _Session(fada=fada)
    harness.run(session, duration=None)
    assert session.step_count == session.command_program.total_control_steps
    assert session.reset_calls == 1
    assert harness.frame_steps() == [0, 4]
    assert harness.writers[-1].closed and harness.renderers[-1].closed


def test_oracle_fixed_duration_preserves_off_grid_final_state(harness):
    session = _Session()
    harness.run(session, duration=0.03, fps=30)
    assert session.step_count == 2
    assert harness.frame_steps() == [0, 2]
    assert any("reason=completed actual_steps=2 actual_sim_time=0.040s" in line for line in harness.logs)


def test_fada_default_keeps_autoreset_and_original_frame_schedule(harness):
    session = _Session(fada=True, stop_step=2)
    harness.run(session)
    assert session.step_count == 6
    assert session.reset_calls == 1
    assert harness.frame_steps() == [0, 5]
    assert harness.renderings[-1].reset_count == 1
    assert harness.renderings[-1].update_count == 1
    assert harness.renderings[-1].last_loss == 0.25


@pytest.mark.parametrize("failure", ["inference", "render"])
def test_recording_exception_closes_encoder_and_renderer(harness, failure):
    session = _Session(fail_step=2 if failure == "inference" else None)
    if failure == "render":
        harness.render_fail_step = 2
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        harness.run(session, fps=50)
    assert harness.writers[-1].closed and harness.renderers[-1].closed
