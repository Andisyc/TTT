"""Opt-in fatal-signal capture arming and a post-unwind diagnostic fallback."""
from __future__ import annotations

import ctypes
import dis
import json
import os
from pathlib import Path
import signal
import sys
import time
import types
from typing import Any


_armed = False


def arm_collection_capture() -> None:
    """Pause once on the cold path until the supervisor arms fatal-signal capture."""
    global _armed
    root_value = os.environ.get("UNILAB_FADA_CAPTURE_ROOT")
    if not root_value or _armed:
        return
    directory = Path(root_value) / f"native-{os.getpid()}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    supervisor = int(os.environ["UNILAB_FADA_CAPTURE_SUPERVISOR_PID"])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, supervisor, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PTRACER failed")
    ready = {
        "pid": os.getpid(), "directory": str(directory),
        "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
    }
    pending = directory / "ready.tmp"
    pending.write_text(json.dumps(ready), encoding="utf-8")
    pending.replace(directory / "ready.json")
    os.kill(os.getpid(), signal.SIGSTOP)
    acknowledgement = directory / "armed.json"
    if not acknowledgement.is_file() or json.loads(acknowledgement.read_text())["pid"] != os.getpid():
        raise RuntimeError("native fatal-signal capture was not armed; execution refused")
    _armed = True


def _identity(value: object) -> dict[str, object]:
    kind = type(value)
    result: dict[str, object] = {
        "id": id(value), "type": f"{kind.__module__}.{kind.__qualname__}",
    }
    if kind in (str, int, float, bool, type(None)):
        result["value"] = value
    elif kind in (types.BuiltinFunctionType, types.FunctionType):
        result["name"] = value.__name__
        result["module"] = value.__module__
    return result


def capture_collection_failure(error: BaseException, transaction: Any | None = None) -> None:
    """Keep Python error details when an exception was not caught at its native entry."""
    root_value = os.environ.get("UNILAB_FADA_CAPTURE_ROOT")
    if not root_value:
        return
    directory = Path(root_value) / f"failure-{os.getpid()}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    report: dict[str, object] = {
        "pid": os.getpid(), "ppid": os.getppid(), "time": time.time(),
        "exception": _identity(error), "message": str(error),
        "python": sys.version, "executable": sys.executable,
        "allocator": os.environ.get("PYTHONMALLOC", "default"),
        "capture_phase": "post-unwind-fallback",
        "frames": [],
    }
    frames: list[dict[str, object]] = []
    current = error.__traceback__
    while current is not None:
        frame = current.tb_frame
        code = frame.f_code
        item: dict[str, object] = {
            "file": code.co_filename, "function": code.co_name,
            "line": current.tb_lineno, "lasti": current.tb_lasti,
            "code_hex": code.co_code.hex(),
            "builtins": {name: _identity(frame.f_builtins.get(name))
                         for name in ("getattr", "enumerate", "isinstance", "callable")},
            "locals": {name: _identity(value) for name, value in frame.f_locals.items()
                       if name in {"record", "records", "name", "value", "offset", "index"}},
        }
        module = frame.f_globals.get("np")
        if type(module) is types.ModuleType:
            item["numpy"] = {name: _identity(module.__dict__.get(name))
                             for name in ("asarray", "isfinite", "all")}
        try:
            item["instructions"] = [
                {"offset": instruction.offset, "opname": instruction.opname,
                 "argrepr": instruction.argrepr}
                for instruction in dis.get_instructions(code, adaptive=True, show_caches=True)
                if abs(instruction.offset - current.tb_lasti) <= 32
            ]
        except Exception as diagnostic_error:
            item["disassembly_error"] = str(diagnostic_error)
        frames.append(item)
        current = current.tb_next
    report["frames"] = frames
    if transaction is not None:
        report["transaction"] = {
            "env_steps": transaction.env_steps,
            "windows": transaction.windows.window_count,
            "target_windows": transaction.num_windows,
        }
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
