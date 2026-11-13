#!/usr/bin/env bash
set -euo pipefail
usage() {
  printf 'Usage: %s --checkpoint PATH --duration-s SECONDS --output-dir DIR [--method LABEL]\n' "$0"
}

CHECKPOINT= DURATION= OUT= METHOD=figure-eight
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --checkpoint|--duration-s|--output-dir|--method)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --checkpoint) CHECKPOINT=$2 ;;
        --duration-s) DURATION=$2 ;;
        --output-dir) OUT=$2 ;;
        --method) METHOD=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$CHECKPOINT" || -z "$DURATION" || -z "$OUT" ]]; then
  usage >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
TABLE=logs/fada_online/v14_g8_return_trim/right_less05.npz
if [[ ! -f "$CHECKPOINT" || ! -r "$CHECKPOINT" ]]; then
  printf 'Checkpoint not readable: %s\n' "$CHECKPOINT" >&2
  exit 1
fi
test -r "$TABLE"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
echo "Running $METHOD; log: $OUT/run.log"
env LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen --no-sync python - "$METHOD" "$CHECKPOINT" "$TABLE" "$OUT" "$DURATION" > "$OUT/run.log" 2>&1 <<'PY'
import math
import sys
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.envs.locomotion.g1.fada_privileged import build_g1_fada_checkpoint_layout_identity
from unilab.training import ensure_registries
from unilab.training.seed import apply_training_seed
from unilab.visualization.demo_commands import FigureEightCommandProgram
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.interactive_playback import create_fada_playback_session
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config

method = sys.argv[1]
checkpoint, table, out = [Path(x).resolve() for x in sys.argv[2:5]]
duration_s = float(sys.argv[5])
if not math.isfinite(duration_s) or duration_s <= 0:
    raise ValueError('--duration-s must be finite and positive')
root = Path.cwd()
original = FigureEightCommandProgram.from_config.__func__

def replay(cls, *args, **kwargs):
    program = original(cls, *args, **kwargs)
    with np.load(table, allow_pickle=False) as data:
        commands = data['commands'].copy()
        if (commands.shape != (52, 3) or commands.shape != program.commands.shape
                or not np.isfinite(commands).all()
                or program.segment_steps != int(data['segment_steps'])
                or not np.isclose(program.ctrl_dt, float(data['ctrl_dt']))
                or not np.allclose(program.reference_points, data['reference_points'])):
            raise ValueError('Current figure-eight profile does not match the saved v14 table')
    commands.setflags(write=False)
    program.commands = commands
    return program

FigureEightCommandProgram.from_config = classmethod(replay)
ensure_registries()
apply_training_seed(0)
with TemporaryDirectory(prefix='table2_g1_scene_') as tmp:
    scene = copytree(root / 'src/unilab/assets/robots/g1', Path(tmp) / 'g1') / 'scene_flat.xml'
    digest = build_g1_fada_checkpoint_layout_identity(
        body_names=['probe'], actuated_joint_names=['probe'], model_file=scene,
    ).asset_sha256
    if digest != '6339734dada64e025b04d4d264cbca899b312705485178840ff1d9f414c28402':
        raise ValueError(f'G1 asset hash mismatch: {digest}')
    with initialize_config_dir(config_dir=str(root / 'conf/distill'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=[
            'task=g1_walk_flat/mujoco', 'recording_profile=figure_eight', 'algo.seed=0',
            'training.device=cuda', 'online_adaptation.enabled=false',
            'env.runtime_payload.enabled=true', 'env.runtime_payload.body_name=torso_link',
            'env.runtime_payload.added_mass_kg=8.0', 'env.runtime_payload.attach_time_s=0.0',
            'env.runtime_payload.attachment_pos_m=[0.004,0.22,0.15]',
            'env.runtime_payload.half_size_m=[0.08,0.06,0.12]',
            'experiment_recording.enabled=true', f'video_recording.duration_s={duration_s}',
        ])
    cfg.training.play_checkpoint_path = str(checkpoint)
    OmegaConf.update(cfg, 'training.fada.scene_model_file', str(scene), force_add=True)
    cfg.experiment_recording.output_dir = str(out / 'raw')
    cfg.video_recording.output_path = str(out / 'trial.mp4')
    args = _build_play_args(cfg, algo='fada')
    session, _, _ = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1), cfg=cfg, root_dir=root, device='cuda',
    )
    try:
        program = session.command_program
        dt = program.ctrl_dt
        if not np.isclose(dt, 0.02) or program.total_control_steps != 2080:
            raise ValueError('Expected a 41.6 s v14 command cycle')
        steps = max(1, math.ceil(duration_s / dt - 1e-9))
        video = FADAVideoRecordingConfig(**OmegaConf.to_container(cfg.video_recording, resolve=True))
        render = FADADemoRenderConfig(**OmegaConf.to_container(cfg.demo.rendering, resolve=True))
        positions, quats = [], []
        def measure():
            positions.append(session.env.get_base_pos()[0].copy())
            quats.append(session.env.get_base_quat()[0].copy())
        with experiment_recording_run(session):
            session.set_autoreset(False)
            session.reset()
            measure()
            completed = 'command_cycle_completed' if steps % program.total_control_steps == 0 else 'duration_reached'
            reason = completed
            with capture_fada_trial(session, args, video, label=method, render_config=render) as capture:
                capture(0)
                for step in range(1, steps + 1):
                    session.step_once()
                    state = session.env.state
                    if step == 1 and not np.isclose(state.info['runtime_payload_added_mass_kg'][0], 8.0):
                        raise RuntimeError('8 kg did not attach on the first physics step')
                    measure()
                    if np.any(state.terminated):
                        reason = 'terminated'
                    elif np.any(state.truncated):
                        reason = 'truncated'
                    done = reason != completed
                    capture(step, final=done or step == steps)
                    if done:
                        break
        np.savez_compressed(out / 'trajectory.npz', time_s=np.arange(step + 1) * dt,
                            base_positions=np.asarray(positions), base_quats=np.asarray(quats),
                            reference_points=program.reference_points, commands=program.commands,
                            segment_steps=program.segment_steps, stop_reason=reason)
        print(f'[eight] {reason}: {step}/{steps} steps, {step*dt:.2f} s')
    finally:
        session.wrapped_env.close()
PY
tail -n 1 "$OUT/run.log"
