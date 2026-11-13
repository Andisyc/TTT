#!/usr/bin/env bash
set -euo pipefail
usage() {
  printf 'Usage: %s --checkpoint BASE_PT --output-dir DIR [--method full_oat|fada_ttt] [--payload-kg KG] [--approach-s SECONDS] [--keep-updating] [--command-table NPZ] [--official-profile-path PATH]\n' "$0"
}

CHECKPOINT= OUT= METHOD=full_oat
PAYLOAD_KG=8 APPROACH_S=30 KEEP_UPDATING=0 OFFICIAL_PROFILE=
TABLE=logs/fada_online/v14_g8_return_trim/right_less05.npz
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --keep-updating) KEEP_UPDATING=1; shift ;;
    --checkpoint|--output-dir|--method|--payload-kg|--approach-s|--command-table|--official-profile-path)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --checkpoint) CHECKPOINT=$2 ;;
        --output-dir) OUT=$2 ;;
        --method) METHOD=$2 ;;
        --payload-kg) PAYLOAD_KG=$2 ;;
        --approach-s) APPROACH_S=$2 ;;
        --command-table) TABLE=$2 ;;
        --official-profile-path) OFFICIAL_PROFILE=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$CHECKPOINT" || -z "$OUT" ]]; then
  usage >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -f "$CHECKPOINT" || ! -r "$CHECKPOINT" ]]; then
  printf 'Checkpoint not readable: %s\n' "$CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$OFFICIAL_PROFILE" && ( ! -f "$OFFICIAL_PROFILE" || ! -r "$OFFICIAL_PROFILE" ) ]]; then
  printf 'Official control profile not readable: %s\n' "$OFFICIAL_PROFILE" >&2
  exit 1
fi
test -r "$TABLE"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
echo "Running $METHOD; log: $OUT/run.log"
env LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen --no-sync python - "$METHOD" "$CHECKPOINT" "$TABLE" "$OUT" "$APPROACH_S" "$PAYLOAD_KG" "$OFFICIAL_PROFILE" "$KEEP_UPDATING" > "$OUT/run.log" 2>&1 <<'PY'
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
from unilab.visualization.experiment_common import GROUP_MODES
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.figure_eight_trial import run_figure_eight_trial
from unilab.visualization.interactive_playback import create_fada_playback_session
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config

method = sys.argv[1]
checkpoint, table, out = [Path(x).resolve() for x in sys.argv[2:5]]
approach_s = float(sys.argv[5])
try:
    payload_kg = float(sys.argv[6])
except ValueError:
    raise ValueError('--payload-kg must be finite and nonnegative') from None
if not math.isfinite(payload_kg) or payload_kg < 0:
    raise ValueError('--payload-kg must be finite and nonnegative')
payload_enabled = payload_kg > 0
official_profile = Path(sys.argv[7]).resolve() if sys.argv[7] else None
keep_updating = sys.argv[8] == '1'
if method not in ('full_oat', 'fada_ttt'):
    raise ValueError('--method must be full_oat or fada_ttt')
if keep_updating and not payload_enabled:
    raise ValueError('--keep-updating is not supported for the unloaded reference')
if not math.isfinite(approach_s) or approach_s <= 0:
    raise ValueError('--approach-s must be finite and positive')
approach_steps = math.ceil(approach_s / 0.02 - 1e-9)
if approach_steps < 1:
    raise ValueError('--approach-s must cover at least one control step')
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
            'training.device=cuda',
            f'env.runtime_payload.enabled={str(payload_enabled).lower()}',
            'env.runtime_payload.body_name=torso_link',
            f'env.runtime_payload.added_mass_kg={payload_kg}',
            'env.runtime_payload.attach_time_s=0.0',
            'env.runtime_payload.attachment_pos_m=[0.004,0.22,0.15]',
            'env.runtime_payload.half_size_m=[0.08,0.06,0.12]',
            'experiment_recording.enabled=true', f'video_recording.duration_s={(approach_steps + 2080) * 0.02}',
            'video_recording.show_status_overlay=false',
            'demo.rendering.show_reference_route=false',
            # 2080 steps, at most one rising contact edge per foot every two steps.
            'demo.footprints.max_records=2080',
            'demo.rendering.max_footprint_markers=${demo.footprints.max_records}',
        ])
    if payload_enabled:
        # Match the Table 2 straight-approach learning budget and method settings.
        online = OmegaConf.load(root / 'conf/distill/oat_component_ablation.yaml').protocol.online
        cfg.online_adaptation = OmegaConf.merge(cfg.online_adaptation, online)
        cfg.online_adaptation.enabled = True
        cfg.online_adaptation.mode = GROUP_MODES[method]
        if method == 'fada_ttt':
            cfg.online_adaptation.sigma = 0.0
    else:
        method = 'nominal'
    cfg.training.play_checkpoint_path = str(checkpoint)
    if official_profile is not None:
        cfg.official_playback.profile_path = str(official_profile)
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
        if not np.allclose(program.reference_points[0], program.reference_points[len(program.commands) // 2]):
            raise ValueError('Expected v14 segment 0 to start at the figure-eight crossing')
        video = FADAVideoRecordingConfig(**OmegaConf.to_container(cfg.video_recording, resolve=True))
        render = FADADemoRenderConfig(**OmegaConf.to_container(cfg.demo.rendering, resolve=True))
        def capture_trial():
            return capture_fada_trial(session, args, video, label=method, render_config=render)

        with experiment_recording_run(session):
            reason, steps = run_figure_eight_trial(
                session, approach_steps=approach_steps, payload_kg=payload_kg,
                keep_updating=keep_updating, capture_trial=capture_trial,
                trajectory_path=out / 'trajectory.npz',
            )
        print(f'[eight] {reason}: {steps}/{program.total_control_steps} figure-eight steps, {steps*dt:.2f} s')
    finally:
        session.wrapped_env.close()
PY
tail -n 1 "$OUT/run.log"
