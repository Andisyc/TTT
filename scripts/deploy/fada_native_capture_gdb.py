"""GDB-only CPython 3.11/x86-64 watcher; never calls code in the inferior.

Both training and collector processes stop only on fatal native signals.
No Python exception or opcode breakpoints are installed. Read at most 8 KiB per
memory request, save VM/C state and core, then kill the failed process; never retry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import traceback

import gdb


DIRECTORY = Path(os.environ["UNILAB_NATIVE_CAPTURE_DIRECTORY"])
ROLE = os.environ["UNILAB_NATIVE_CAPTURE_ROLE"]
CORE = Path(os.environ["UNILAB_NATIVE_CAPTURE_CORE"])
LAYOUT = json.loads(Path(os.environ["UNILAB_NATIVE_CAPTURE_LAYOUT"]).read_text())
OFFSETS = LAYOUT["offsets"]
HIT: dict = {}
LAST_SIGNAL: str | None = None


def publish(name: str, value: dict) -> None:
    pending = DIRECTORY / (name + ".tmp")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(DIRECTORY / name)


def memory(address: int, size: int) -> bytes:
    if address <= 0 or size < 0 or size > 8192:
        raise ValueError(f"bounded memory read rejected: {address:#x}, {size}")
    return bytes(gdb.selected_inferior().read_memory(address, size))


def integer(address: int, size: int = 8, signed: bool = False) -> int:
    return int.from_bytes(memory(address, size), "little", signed=signed)


def member(address: int, key: str, size: int = 8, signed: bool = False) -> int:
    return integer(address + OFFSETS[key], size, signed)




def unicode_string(address: int) -> str:
    length = member(address, "PyASCIIObject.length", signed=True)
    state = member(address, "PyASCIIObject.state", 4)
    kind, compact, ascii_only = (state >> 2) & 7, bool(state & 32), bool(state & 64)
    if length < 0 or length > 2048 or kind not in (1, 2, 4):
        raise ValueError("invalid/beyond-limit Python Unicode object")
    if ascii_only:
        data = address + OFFSETS["sizeof.PyASCIIObject"]
    elif compact:
        data = address + OFFSETS["sizeof.PyCompactUnicodeObject"]
    else:
        data = member(address, "PyUnicodeObject.data")
    if not length:
        return ""
    return memory(data, length * kind).decode({1: "latin-1", 2: "utf-16-le", 4: "utf-32-le"}[kind])




def python_frames() -> tuple[int, list[dict]]:
    runtime = int(gdb.parse_and_eval("(void*)&_PyRuntime"))
    tstate = integer(runtime + OFFSETS["_PyRuntimeState.gilstate.tstate_current"])
    cframe = member(tstate, "PyThreadState.cframe")
    frame = member(cframe, "_PyCFrame.current_frame")
    frames, seen = [], set()
    while frame and len(frames) < 64:
        if frame in seen:
            raise ValueError("cycle in interpreter frame chain")
        seen.add(frame)
        code = member(frame, "_PyInterpreterFrame.f_code")
        start = code + OFFSETS["PyCodeObject.co_code_adaptive"]
        frames.append({
            "frame": frame, "code": code,
            "function": unicode_string(member(code, "PyCodeObject.co_name")),
            "file": unicode_string(member(code, "PyCodeObject.co_filename")),
            "first_line": member(code, "PyCodeObject.co_firstlineno", 4, True),
            "lasti": member(frame, "_PyInterpreterFrame.prev_instr") - start,
        })
        frame = member(frame, "_PyInterpreterFrame.previous")
    return tstate, frames


def code_snapshot(item: dict) -> dict:
    code = item["code"]
    length = member(code, "PyVarObject.ob_size", signed=True) * 2
    if length <= 0 or length > 65536:
        raise ValueError("invalid/beyond-limit code size")
    raw = memory(code + OFFSETS["PyCodeObject.co_code_adaptive"], min(length, 8192))
    offset, cursor, kind, opcode = item["lasti"], 0, "outside_snapshot", None
    while cursor < len(raw):
        instruction = LAYOUT["deopt"][raw[cursor]]
        cache_end = cursor + 2 * (1 + LAYOUT["caches"][instruction])
        if offset == cursor:
            kind, opcode = "instruction", instruction
            break
        if cursor < offset < cache_end:
            kind, opcode = "inline_cache", 0
            break
        cursor = cache_end
    return {
        "byte_length": length, "adaptive_hex": raw.hex(),
        "truncated": length > len(raw), "failed_offset_kind": kind,
        "canonical_opcode": opcode,
        "canonical_opname": LAYOUT["opcode_names"][opcode] if opcode is not None else None,
        "word_at_lasti": raw[offset:offset + 2].hex() if 0 <= offset < len(raw) else None,
    }



def on_stop(event) -> None:
    global LAST_SIGNAL
    if isinstance(event, gdb.SignalEvent):
        LAST_SIGNAL = event.stop_signal


def main() -> None:
    os.umask(0o077)
    gdb.execute("set pagination off")
    gdb.execute("set confirm off")
    gdb.execute("set print elements 32")
    gdb.execute("set debuginfod enabled off")
    gdb.execute("handle SIGSTOP nostop noprint nopass")
    gdb.execute("handle SIGCONT nostop noprint pass")
    gdb.execute("handle SIGTERM SIGINT SIGHUP nostop noprint pass")
    gdb.execute("handle SIGSEGV SIGBUS SIGILL SIGABRT SIGFPE stop print nopass")
    gdb.events.stop.connect(on_stop)
    inferior = gdb.selected_inferior()
    try:
        publish("armed.json", {"pid": inferior.pid, "role": ROLE,
                               "entries": [], "phase": "native_signals"})
        os.kill(inferior.pid, signal.SIGCONT)
        while inferior.pid and not HIT:
            gdb.execute("continue")
            if LAST_SIGNAL in {"SIGSEGV", "SIGBUS", "SIGILL", "SIGABRT", "SIGFPE"}:
                HIT.update(phase="native_signal", signal=LAST_SIGNAL)
            elif LAST_SIGNAL is not None:
                raise RuntimeError(f"unexpected debugger stop: {LAST_SIGNAL}")
        if HIT:
            publish("event.json", {"pid": inferior.pid, "role": ROLE, "phase": HIT["phase"]})
            report = {**HIT, "pid": inferior.pid, "role": ROLE}
            for name, command in (("registers", "info registers"), ("native_backtrace", "bt 40")):
                try:
                    report[name] = gdb.execute(command, to_string=True)
                except gdb.error as error:
                    report[name + "_error"] = str(error)
            try:
                report["native_stack_hex"] = memory(int(gdb.parse_and_eval("$rsp")), 512).hex()
            except Exception as error:
                report["native_stack_error"] = str(error)
            try:
                report["tstate"], report["python_frames"] = python_frames()
            except Exception as error:
                report["python_frames_error"] = str(error)
            if report.get("python_frames"):
                try:
                    report["code_snapshot"] = code_snapshot(report["python_frames"][0])
                except Exception as error:
                    report["code_snapshot_error"] = str(error)
            publish("native_report.json", report)
            try:
                gdb.execute("thread apply all bt 20")
            except gdb.error as error:
                gdb.write(f"Thread backtrace unavailable: {error}\n")
            gdb.execute("generate-core-file " + str(CORE))
            publish("complete.json", {"pid": inferior.pid, "role": ROLE, "phase": HIT["phase"],
                                      "core_complete": CORE.is_file() and CORE.stat().st_size > 0,
                                      "core_bytes": CORE.stat().st_size if CORE.exists() else 0})
            # Never continue a failed formal run or unwind the damaged Python frame.
            gdb.execute("kill")
        publish("debugger_done.json", {"captured": bool(HIT)})
    except BaseException:
        publish("debugger_error.json", {"error": traceback.format_exc()})
        if inferior.pid:
            try:
                gdb.execute("kill")
            except gdb.error:
                os.kill(inferior.pid, signal.SIGKILL)
        raise


main()
