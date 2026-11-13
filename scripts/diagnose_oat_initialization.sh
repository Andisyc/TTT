#!/usr/bin/env bash
# Diagnostic only: no policy action, physics control step, update, or formal output mutation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run --frozen --no-sync python - "$ROOT" "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time

WORKER = r'''
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import traceback

root, trial_path, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
output.mkdir(parents=True, exist_ok=True)
# These files belong only to this capture; never open a formal trial for writing.
pid = os.getpid()
stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
(output / 'process.json').write_text(json.dumps({'pid': pid, 'start': stat[19]}))
resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
sys.path.insert(0, str(root / 'src'))
phase = 'entry'
try:
    from unilab.visualization.experiment_execution import build_experiment_config
    trial = json.loads(trial_path.read_text())
    trial['output_dir'] = str(output / 'trial')
    phase = 'config'
    cfg = build_experiment_config(trial, root_dir=root)
    if mode == 'session_reset':
        from unilab.training.seed import apply_training_seed
        from unilab.visualization.playback_distill_sessions import create_fada_playback_session
        from unilab.visualization.playback_sessions import RslRlPlaybackConfig
        from unilab.visualization.experiment_recording import experiment_recording_run
        apply_training_seed(int(cfg.algo.seed))
        playback = RslRlPlaybackConfig(
            task=str(trial['task_owner']), load_run='-1', checkpoint=None,
            checkpoint_path=str(trial['checkpoint_path']), action_mode='policy',
            policy_obs_mode='actor', algo_log_name='distill', log_root=None,
            num_envs=1, keyboard=False,
        )
        phase = 'create_session'
        session, _, _ = create_fada_playback_session(
            playback_cfg=playback, cfg=cfg, root_dir=root, device=cfg.training.device,
        )
        try:
            with experiment_recording_run(session):
                session.set_autoreset(False)
                phase = 'reset'
                session.reset()
                phase = 'recording_import_after_reset'
                importlib.import_module('unilab.visualization.fada_recording')
                phase = 'recording_import_complete'
        finally:
            phase_before_close = phase
            phase = 'close_session'
            session.wrapped_env.close()
            phase = phase_before_close
    else:
        # Same project imports and config composition, without checkpoint/environment creation.
        phase = 'recording_import_without_session'
        importlib.import_module('unilab.visualization.fada_recording')
    result = {'outcome': 'PASS', 'phase': phase, 'mode': mode,
              'path_class': 'ALTERNATE-PATH: production initialization owners only',
              'control_steps': 0, 'optimizer_steps': 0}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print('HPD_PASS', json.dumps(result), flush=True)
except BaseException as exc:
    result = {'outcome': 'FAIL', 'phase': phase, 'mode': mode,
              'exception': type(exc).__name__, 'message': str(exc)}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    traceback.print_exc()
    print('HPD_FAIL', json.dumps(result), flush=True)
    # Retain the Python exception and its traceback in the stopped process.
    # An explicit abort is a capture trigger, NOT evidence of a native SIGABRT.
    sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()
    if os.environ.get('HPD_CAPTURE_ABORT') == '1':
        print('HPD_EXPLICIT_ABORT_FOR_CAPTURE', flush=True)
        os.abort()
    raise
'''


def run_text(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=15)
    return {'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def terminate_owned_tree(parent):
    """Used only at the declared timeout; do not leave a GDB inferior behind."""
    owned = []
    def collect(pid):
        start = process_start(pid)
        if start is None:
            return
        owned.append((pid, start))
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            children = []
        for child in children:
            collect(int(child))
    collect(parent.pid)
    for pid, start in reversed(owned):
        if process_start(pid) == start:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    parent.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description='Bounded initialization-only Heisenbug capture; never run a rollout.')
    parser.add_argument('repo_root', type=Path)
    parser.add_argument('--trial', required=True, type=Path, help='Existing failed request JSON; read only')
    parser.add_argument('--capture-root', required=True, type=Path, help='Absolute server-local diagnostic storage directory')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Capture requires the Linux server and its existing GDB')
    root, trial_path = args.repo_root.resolve(), args.trial.expanduser().resolve()
    if not args.capture_root.is_absolute():
        parser.error('--capture-root must be absolute')
    capture_root = args.capture_root.resolve()
    trial = json.loads(trial_path.read_text())
    checkpoint = Path(trial['checkpoint_path']).resolve()
    original_output = Path(trial['output_dir']).resolve()
    if capture_root == root or capture_root == original_output or original_output in capture_root.parents:
        parser.error('Capture root must not be the repository root or inside the formal trial')
    if not checkpoint.is_file() or not (root / 'src/unilab/visualization/experiment_execution.py').is_file():
        parser.error('Checkpoint or repository execution owner is missing')
    gdb = shutil.which('gdb')
    if gdb is None:
        parser.error('Existing GDB is required; no tools will be installed')
    disk_path = capture_root
    while not disk_path.exists():
        disk_path = disk_path.parent
    if shutil.disk_usage(disk_path).free < 6 * 1024**3:
        parser.error('Require at least 6 GiB free on capture storage')
    capture_root.mkdir(parents=True, exist_ok=True)
    capture = capture_root / (time.strftime('initialization-%Y%m%d-%H%M%S') + f'-{os.getpid()}')
    capture.mkdir()
    worker = capture / 'worker.py'
    worker.write_text(WORKER)
    (capture / 'request.json').write_text(json.dumps(trial, indent=2))
    identity = {
        'repo': str(root), 'python': sys.executable, 'python_version': sys.version,
        'trial_source': str(trial_path),
        'request_sha256': hashlib.sha256(trial_path.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'commit': run_text(['git', 'rev-parse', 'HEAD'], root),
        'dirty': run_text(['git', 'status', '--short'], root),
        'ulimit': run_text(['bash', '-c', 'ulimit -a'], root),
        'gdb': run_text([gdb, '--version'], root),
        'environment_names': sorted(os.environ),
        'environment_allowlist': {k: os.environ[k] for k in (
            'LD_PRELOAD', 'MUJOCO_GL', 'PYTHONMALLOC', 'PYTHONFAULTHANDLER',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
        ) if k in os.environ},
        'packages': {name: importlib.metadata.version(name) for name in
                     ('mujoco-uni', 'torch', 'mediapy', 'matplotlib', 'pyparsing')},
        'root_core_limit': resource.getrlimit(resource.RLIMIT_CORE),
        'observer': 'plain vs GDB + PYTHONMALLOC=debug; observer-sensitive nonfailure is not a repair',
        'budget': 'At most 4 fresh processes, 60 seconds each; no control steps, gradients, installs or core files',
        'hypotheses': ['native initialization damages runtime state',
                       'interpreter/hardware execution instability',
                       'recording dependency import failure independent of simulation'],
    }
    distribution = importlib.metadata.distribution('mujoco-uni')
    identity['native_library_sha256'] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for item in (distribution.files or [])
        if any(name in str(item) for name in ('_batch_env.', '_specs.', '_functions.', 'libmujoco.so'))
        for p in [Path(distribution.locate_file(item))]
        if p.is_file()
    }
    for name in ('core_pattern', 'yama/ptrace_scope'):
        p = Path('/proc/sys/kernel') / name
        identity[name] = p.read_text().strip() if p.is_file() else None
    cpu = Path('/proc/cpuinfo').read_text()
    identity['cpu'] = [line for line in cpu.splitlines() if line.startswith(('model name', 'microcode'))][:2]
    if shutil.which('nvidia-smi'):
        identity['gpu'] = run_text(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader'], root)
    (capture / 'identity.json').write_text(json.dumps(identity, indent=2))
    results = []
    failed_mode = None
    for observer in ('plain', 'gdb_debug_allocator'):
        for mode in ('recording_only', 'session_reset'):
            if observer != 'plain' and failed_mode is not None and mode != failed_mode:
                continue
            destination = capture / f'{observer}-{mode}'
            destination.mkdir()
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONFAULTHANDLER'] = '1'
            target = [sys.executable, '-u', str(worker), str(root), str(trial_path), str(destination), mode]
            command = target
            if observer != 'plain':
                env['PYTHONMALLOC'] = 'debug'
                env['HPD_CAPTURE_ABORT'] = '1'
                capture_code = (
                    "import gdb\n"
                    "if gdb.selected_inferior().pid:\n"
                    " print('HPD_GDB_STOPPED_INFERIOR')\n"
                    " gdb.execute('thread apply all bt 30')\n"
                    " try: gdb.execute('py-bt')\n"
                    " except gdb.error as exc: print('PY_BT_UNAVAILABLE', exc)\n"
                    " gdb.execute('info sharedlibrary')\n"
                    " gdb.execute('kill')\n"
                    "else: print('HPD_GDB_INFERIOR_EXITED')\n"
                )
                command = [gdb, '--nx', '--quiet', '--batch', '--return-child-result',
                           '-ex', 'set pagination off', '-ex', 'set confirm off',
                           '-ex', 'set startup-with-shell off',
                           '-ex', 'set disable-randomization off',
                           '-ex', 'handle SIGABRT stop print nopass',
                           '-ex', 'handle SIGSEGV stop print nopass',
                           '-ex', 'run', '-ex', 'python exec(' + repr(capture_code) + ')',
                           '--args', *target]
            else:
                env.pop('HPD_CAPTURE_ABORT', None)
            print(f'HPD_START {observer} {mode}; maximum 60s; no control steps', flush=True)
            start = time.monotonic()
            with (destination / 'capture.log').open('w') as stream:
                proc = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_owned_tree(proc)
            result_file = destination / 'result.json'
            result = json.loads(result_file.read_text()) if result_file.is_file() else {'outcome': 'UNRESOLVED'}
            process_file = destination / 'process.json'
            inferior_exited = None
            if process_file.is_file():
                process = json.loads(process_file.read_text())
                inferior_exited = process_start(process['pid']) != process['start']
            if timed_out or inferior_exited is False:
                result['outcome'] = 'UNRESOLVED'
            if proc.returncode != 0 and result['outcome'] == 'PASS':
                result.update(outcome='UNRESOLVED', reason='Failure after worker completion or debugger setup failure')
            if not result_file.is_file() and proc.returncode in (-signal.SIGABRT, -signal.SIGSEGV):
                result.update(outcome='FAIL', reason='Native fatal signal before Python result')
            if not result_file.is_file() and 'HPD_GDB_STOPPED_INFERIOR' in (destination / 'capture.log').read_text(errors='replace'):
                result.update(outcome='FAIL', reason='Debugger stopped inferior before Python result; inspect capture.log for signal')
            result.update(observer=observer, mode=mode, exit_code=proc.returncode,
                          timed_out=timed_out, wall_s=time.monotonic()-start,
                          inferior_exited=inferior_exited)
            results.append(result)
            (capture / 'results.json').write_text(json.dumps(results, indent=2))
            print('HPD_RESULT', json.dumps(result), flush=True)
            if result['outcome'] != 'PASS':
                failed_mode = mode
                break
        if observer != 'plain' and failed_mode is not None:
            break
    archive = capture / 'RETURN_ME.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for p in sorted(capture.rglob('*')):
            if p == archive or not p.is_file():
                continue
            bundle.add(p, arcname=str(p.relative_to(capture)))
    print(f'HPD_ARTIFACT {archive}', flush=True)
    print('No production files, checkpoint, existing logs, or experiment data were modified.', flush=True)
    print('All passes mean no reproduction within this bounded capture, not Heisenbug elimination.', flush=True)
    return 0 if all(r['outcome'] == 'PASS' for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
PY
