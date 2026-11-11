"""Run official Oracle training with fatal-signal capture.

Arm both training and collector processes without Python exception breakpoints.
Run --self-test to verify collector termination and main-process SIGSEGV capture
without loading models or simulation. Save registers and core, then kill the
failed process; never retry or change training semantics. Private captures
contain process memory.
"""
from __future__ import annotations

import argparse
import dis
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import time


def process_start(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def build_layout(root: Path) -> Path:
    """Compile only sizeof/offsetof constants; never link or call the simulator."""
    fields = {
        "_PyRuntimeState": ["gilstate.tstate_current"],
        "PyThreadState": ["cframe"],
        "_PyCFrame": ["current_frame"],
        "_PyInterpreterFrame": ["f_code", "previous", "prev_instr"],
        "PyCodeObject": ["co_name", "co_filename", "co_firstlineno", "co_code_adaptive"],
        "PyVarObject": ["ob_size"],
        "PyASCIIObject": ["length", "state"],
        "PyUnicodeObject": ["data"],
    }
    lines = ["#define Py_BUILD_CORE", "#include <Python.h>",
             '#include "internal/pycore_runtime.h"', '#include "internal/pycore_frame.h"',
             "#include <stddef.h>", "#include <stdio.h>", "int main(void) {"]
    for kind, names in fields.items():
        for name in names:
            lines.append(f'printf("{kind}.{name}=%zu\\n", offsetof({kind},{name}));')
    for kind in ("PyASCIIObject", "PyCompactUnicodeObject"):
        lines.append(f'printf("sizeof.{kind}=%zu\\n", sizeof({kind}));')
    lines.append("return 0; }")
    source, binary = root / "gdb-layout.c", root / "gdb-layout"
    source.write_text("\n".join(lines))
    subprocess.run(["cc", "-I" + sysconfig.get_paths()["include"], str(source), "-o", str(binary)],
                   check=True, timeout=15)
    values = subprocess.check_output([str(binary)], text=True, timeout=2)
    layout = {
        "python": sys.version,
        "offsets": {key: int(value) for key, value in
                    (line.split("=", 1) for line in values.splitlines())},
        "deopt": [vars(dis)["_deoptop"](op) for op in range(256)],
        "caches": vars(dis)["_inline_cache_entries"],
        "opcode_names": dis.opname,
    }
    path = root / "gdb-layout.json"
    path.write_text(json.dumps(layout))
    return path


def start_debugger(ready: dict, gdb: str, script: Path, layout: Path) -> dict:
    pid = int(ready["pid"])
    if process_start(pid) != ready["start_ticks"]:
        raise RuntimeError("capture PID identity changed")
    directory = Path(ready["directory"])
    role = ready["role"]
    core = directory / ("training.core" if role == "training_main" else "collector.core")
    output = (directory / "gdb.log").open("w")
    try:
        debugger = subprocess.Popen(
            [gdb, "-q", "-nx", "-batch", "-p", str(pid), "-x", str(script)],
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "LC_ALL": "C", "PYTHONPATH": "", "DEBUGINFOD_URLS": "",
                 "UNILAB_NATIVE_CAPTURE_DIRECTORY": str(directory),
                 "UNILAB_NATIVE_CAPTURE_LAYOUT": str(layout),
                 "UNILAB_NATIVE_CAPTURE_ROLE": role,
                 "UNILAB_NATIVE_CAPTURE_CORE": str(core)},
        )
    finally:
        output.close()
    return {"pid": pid, "role": role, "core": core, "directory": directory, "debugger": debugger,
            "start_ticks": ready["start_ticks"], "started": time.monotonic(), "event_started": None}


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            pass


def summarize_capture(session: dict) -> dict:
    directory = session["directory"]
    core = session["core"]
    complete_path = directory / "complete.json"
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
    result = {
        "pid": session["pid"], "role": session["role"], "phase": complete.get("phase"),
        "core": str(core), "gdb_log": str(directory / "gdb.log"),
        "native_report": str(directory / "native_report.json"),
        "gdb_exitcode": session["debugger"].poll(),
        "core_complete": bool(complete.get("core_complete")) and not session.get("limit_reached"),
        "core_bytes": core.stat().st_size if core.exists() else 0,
    }
    error = directory / "debugger_error.json"
    if error.is_file():
        result["debugger_error"] = json.loads(error.read_text())["error"]
    if session.get("limit_reached"):
        result["capture_limit_reached"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--full-training", action="store_true")
    parser.add_argument("--allocator", choices=("default", "debug"), default="default")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--core-gib", type=int, default=20)
    args = parser.parse_args()
    root, repo = args.capture_root, args.repo.resolve()
    if not root.is_absolute() or root == repo or root.exists():
        parser.error("--capture-root must be a new absolute directory, not the repo root")
    if args.seconds < 0 or args.core_gib <= 0 or (args.seconds == 0 and not args.full_training):
        parser.error("seconds must be positive, or zero with --full-training; core limit must be positive")
    gdb = shutil.which("gdb")
    if (not sys.platform.startswith("linux") or platform.machine() != "x86_64"
            or sys.version_info[:2] != (3, 11) or not gdb or not shutil.which("cc")):
        raise RuntimeError("CPython 3.11/Linux x86-64, existing GDB and cc required; no run started")
    if not root.parent.is_dir():
        raise RuntimeError("capture parent must already exist")
    if shutil.disk_usage(root.parent).free < (args.core_gib + 4) * 1024**3:
        raise RuntimeError("insufficient capture space; no run started")
    os.umask(0o077)
    root.mkdir(mode=0o700)
    layout = build_layout(root)
    env = os.environ.copy()
    env.update(UNILAB_FADA_CAPTURE_ROOT=str(root),
               UNILAB_FADA_CAPTURE_SUPERVISOR_PID=str(os.getpid()), PYTHONFAULTHANDLER="1")
    if args.full_training:
        env.pop("UNILAB_FADA_CAPTURE_ONLY", None)
    else:
        env["UNILAB_FADA_CAPTURE_ONLY"] = "1"
    if args.allocator == "debug":
        env["PYTHONMALLOC"] = "debug"
    else:
        env.pop("PYTHONMALLOC", None)
    overlay = Path("/ssd1/cyx/.deps/fada-native-repair/overlay").resolve()
    if not args.self_test and not (overlay / "mujoco").is_dir():
        raise RuntimeError("verified MuJoCo overlay missing; no run started")
    env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(repo / "src")))
    if args.self_test:
        helper = repo / "src/unilab/algos/torch/distill/fada/failure_capture.py"
        bootstrap = (
            "import ctypes, importlib.util, os, signal, subprocess, sys\n"
            f"assert os.environ.get('UNILAB_FADA_CAPTURE_ONLY') == {None if args.full_training else '1'!r}\n"
            f"spec=importlib.util.spec_from_file_location('capture_fixture', {str(helper)!r})\n"
            "module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
            "module.arm_collection_capture()\n"
        )
        collector = bootstrap + "os.kill(os.getpid(), signal.SIGTERM)\n"
        fixture = (
            bootstrap
            + f"child=subprocess.run([sys.executable, '-u', '-c', {collector!r}])\n"
            + "assert child.returncode == -signal.SIGTERM\n"
            + "print('COLLECTOR_TERMINATED', flush=True)\n"
            + "ctypes.string_at(0)\n"
        )
        command = [sys.executable, "-u", "-c", fixture]
    else:
        command = [sys.executable, "-u", "scripts/train_distill.py", "--config-name=oracle_fada",
                   "teacher.checkpoint_path=/ssd1/cyx/TTT/logs/oracle/mirror_yaw025_50k_20261108-054124/model_50000.pt",
                   "training.device=cuda", "training.fada.num_envs=22",
                   "training.fada.batch_size=4096", "training.fada.replay_capacity=524288",
                   "training.fada.windows_per_iteration=131072", "training.fada.iterations=31",
                   "training.fada.idm_updates=2000", "training.fada.planner_updates=2000",
                   f"training.fada.checkpoint_path={root}/student.pt",
                   f"training.fada.async_artifact_dir={root}/source_batches"]
    identity = {
        "command": command,
        "observer": f"native-fatal-signals-only; allocator={args.allocator}",
        "python": sys.version, "executable": sys.executable,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "limits": {"seconds": args.seconds, "arm_seconds": 30, "capture_seconds": 90,
                   "core_gib": args.core_gib, "memory_read_bytes": 8192},
        "environment": {key: env.get(key) for key in
                        ("PYTHONPATH", "PYTHONMALLOC", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "versions": {},
    }
    for name in ("torch", "numpy", "mujoco-uni"):
        try:
            identity["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity["versions"][name] = "metadata unavailable"
    (root / "identity.json").write_text(json.dumps(identity, indent=2))
    started = time.monotonic()
    sessions: list[dict] = []
    seen: set[Path] = set()
    process = None
    timed_out = False
    supervisor_error = None
    script = Path(__file__).with_name("fada_native_capture_gdb.py")
    budget = args.core_gib * 1024**3
    try:
        with (root / "train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                for path in root.glob("native-*/ready.json"):
                    if path not in seen:
                        seen.add(path)
                        ready = json.loads(path.read_text())
                        main_process = int(ready["pid"]) == process.pid
                        ready["role"] = "training_main" if main_process else "collector"
                        sessions.append(start_debugger(ready, gdb, script, layout))
                now = time.monotonic()
                for session in sessions:
                    directory, debugger = session["directory"], session["debugger"]
                    if not (directory / "armed.json").exists():
                        if debugger.poll() is not None or now - session["started"] > 30:
                            raise RuntimeError("native debugger failed to arm; training stopped")
                    if (directory / "event.json").exists():
                        if session["event_started"] is None:
                            session["event_started"] = now
                        core = session["core"]
                        if (now - session["event_started"] > 90
                                or (core.exists() and core.stat().st_size > budget)):
                            session["limit_reached"] = True
                            raise RuntimeError("native capture reached time/core limit; training stopped")
                    if debugger.poll() is not None and not (directory / "debugger_done.json").exists():
                        raise RuntimeError("native debugger exited unexpectedly; training stopped")
                if args.seconds and now - started >= args.seconds:
                    timed_out = True
                    break
                time.sleep(0.1)
    except Exception as error:
        supervisor_error = str(error)
    finally:
        # Kill still-stopped targets before debugger detach can resume them.
        if supervisor_error or timed_out:
            for session in sessions:
                try:
                    if process_start(session["pid"]) == session["start_ticks"]:
                        os.kill(session["pid"], signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        if process is not None:
            stop_process_group(process)
        for session in sessions:
            debugger = session["debugger"]
            if debugger.poll() is None:
                try:
                    debugger.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    debugger.kill()
                    debugger.wait(timeout=5)
    if process is None:
        raise RuntimeError(supervisor_error or "training process was not started")
    captures = [summarize_capture(session) for session in sessions]
    complete = bool(captures) and any(item["core_complete"] for item in captures)
    fixture_ok = False
    if args.self_test and complete:
        for capture in captures:
            path = Path(capture["native_report"])
            if not path.is_file():
                continue
            report = json.loads(path.read_text())
            common_ok = (capture["core_complete"] and capture["gdb_exitcode"] == 0
                         and process.returncode == -signal.SIGKILL and not supervisor_error
                         and not timed_out)
            collectors = [item for item in captures if item["role"] == "collector"]
            fixture_ok = (
                common_ok and capture["role"] == "training_main"
                and report.get("phase") == "native_signal" and report.get("signal") == "SIGSEGV"
                and bool(report.get("registers")) and bool(report.get("native_backtrace"))
                and len(collectors) == 1 and collectors[0]["gdb_exitcode"] == 0
                and not collectors[0]["core_complete"] and "debugger_error" not in collectors[0]
                and "COLLECTOR_TERMINATED" in (root / "train.log").read_text(errors="replace")
            )
            if fixture_ok:
                break
    result = {
        "outcome": ("PASS" if fixture_ok else "UNRESOLVED") if args.self_test
                   else ("FAIL" if complete else "UNRESOLVED"),
        "self_test": args.self_test, "exitcode": process.returncode,
        "timed_out": timed_out, "supervisor_error": supervisor_error,
        "elapsed_seconds": time.monotonic() - started, "captures": captures,
        "training_completed": args.full_training and process.returncode == 0
                              and not timed_out and not supervisor_error,
        "root": str(root),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.self_test:
        return 0 if fixture_ok else 2
    if timed_out:
        return 124
    if supervisor_error:
        return 2
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
