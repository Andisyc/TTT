"""GDB-only CPython 3.11/x86-64 watcher; never calls code in the inferior.

Collectors stop at the native error formatter before the target exception unwinds.
The training main process watches fatal signals only. Read at most 8 KiB per
memory request, save VM/C state and core, then kill the failed process; never retry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
STACK_PROFILE: dict = {}


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


def c_string(address: int, limit: int = 512) -> str:
    result = bytearray()
    while len(result) < limit:
        count = min(limit - len(result), 4096 - address % 4096)
        chunk = memory(address, count)
        end = chunk.find(b"\0")
        if end >= 0:
            result.extend(chunk[:end])
            break
        result.extend(chunk)
        address += count
    return result.decode("utf-8", errors="replace")


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


def object_identity(address: int) -> dict:
    result: dict[str, object] = {"address": address}
    if address:
        try:
            kind = member(address, "PyObject.ob_type")
            result["type"] = c_string(member(kind, "PyTypeObject.tp_name"), 200)
        except Exception as error:
            result["decode_error"] = str(error)
    return result


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

def discover_stack_profile() -> dict:
    """Locate the actual stack-pointer spill before BEFORE_WITH reuses its register."""
    table = int(gdb.parse_and_eval("(void*)&'_PyEval_EvalFrameDefault.opcode_targets'"))
    entry = integer(table + 53 * 8)
    instructions = gdb.selected_frame().architecture().disassemble(entry, count=16)
    for index, instruction in enumerate(instructions):
        top = re.search(r"(?:-0x8|-8)\(%([a-z0-9]+)\)", instruction["asm"])
        if not top:
            continue
        register = top.group(1)
        for saved in instructions[:index]:
            spill = re.search(r"mov\s+%" + register + r",(-0x[0-9a-f]+)\(%rbp\)", saved["asm"])
            if spill:
                return {"register": register, "home_offset": int(spill.group(1), 16),
                        "evidence": [item["asm"] for item in instructions[:index + 1]]}
    raise RuntimeError("unsupported interpreter stack-pointer profile; collection refused")



def live_stack(item: dict) -> dict:
    frame, code = item["frame"], item["code"]
    nlocals = member(code, "PyCodeObject.co_nlocalsplus", 4, True)
    capacity = member(code, "PyCodeObject.co_stacksize", 4, True)
    if not 0 <= nlocals <= 256 or not 0 <= capacity <= 512:
        raise ValueError("invalid/beyond-limit VM stack allocation")
    base = frame + OFFSETS["_PyInterpreterFrame.localsplus"]
    stack_base, stack_end = base + nlocals * 8, base + (nlocals + capacity) * 8
    native = gdb.newest_frame()
    while native and "_PyEval_EvalFrameDefault" not in (native.name() or ""):
        native = native.older()
    registers = {}
    if native:
        for name in ("rbp", "rbx", "r12", "r13", "r14", "r15"):
            try:
                registers[name] = int(native.read_register(name))
            except gdb.error:
                pass
    # The formatter's caller has already reused the physical register. Read its
    # saved home, derived from this executable's actual BEFORE_WITH instructions.
    home = registers.get("rbp", 0) + STACK_PROFILE["home_offset"]
    pointer = integer(home) if native else 0
    resolved = stack_base <= pointer <= stack_end and (pointer - stack_base) % 8 == 0
    result = {
        "nlocals": nlocals, "capacity": capacity, "base": stack_base, "end": stack_end,
        "saved_frame_stacktop": member(frame, "_PyInterpreterFrame.stacktop", 4, True),
        "eval_frame_registers": registers, "pointer_source": "eval_frame.stack_home",
        "pointer_home": home, "stack_profile": STACK_PROFILE,
        "pointer_resolved": resolved, "pointer": pointer,
        "allocated_slots_hex": memory(base, (nlocals + capacity) * 8).hex(),
    }
    if resolved:
        count = (pointer - stack_base) // 8
        result["live_count"] = count
        result["top_objects"] = [object_identity(integer(pointer - 8 * i))
                                 for i in range(1, min(count, 8) + 1)]
    return result


class TargetError(gdb.Breakpoint):
    def __init__(self, symbol: str):
        address = int(gdb.parse_and_eval(f"(void*)&{symbol}"))
        super().__init__(f"*{address:#x}", internal=True)
        self.symbol = symbol

    def stop(self) -> bool:
        try:
            register = "rdx" if self.symbol == "_PyErr_Format" else "rsi"
            format_pointer = int(gdb.parse_and_eval("$" + register))
            text = c_string(format_pointer)
            if "object is not callable" not in text and "context manager protocol" not in text:
                return False
            tstate, frames = python_frames()
            target = next((frame for frame in frames if frame["function"] == "_validate_records"
                           and frame["file"].endswith("/fada/windows.py")), None)
            if target is None:
                return False
            HIT.update(phase="pre_exception_format", entry=self.symbol, format=text,
                       tstate=tstate, python_frames=frames, target_frame=target)
            return True
        except Exception as error:
            # A matching error with unreadable interpreter state is itself decisive
            # evidence. Fail closed instead of losing it by continuing the collector.
            HIT.update(phase="native_inspection_error", entry=self.symbol,
                       inspection_error=str(error))
            return True


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
    gdb.execute("handle SIGSEGV SIGBUS SIGILL SIGABRT SIGFPE stop print nopass")
    gdb.events.stop.connect(on_stop)
    inferior = gdb.selected_inferior()
    try:
        breakpoints = []
        if ROLE == "collector":
            STACK_PROFILE.update(discover_stack_profile())
            breakpoints = [TargetError(name) for name in ("_PyErr_Format", "PyErr_Format")]
        publish("armed.json", {"pid": inferior.pid, "role": ROLE,
                               "entries": [b.symbol for b in breakpoints],
                               "phase": "native_error_entry" if breakpoints else "native_signals"})
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
            if report["phase"] == "native_signal":
                try:
                    report["tstate"], report["python_frames"] = python_frames()
                except Exception as error:
                    report["python_frames_error"] = str(error)
            if report.get("python_frames"):
                current = report["python_frames"][0]
                actions = [("code_snapshot", code_snapshot)]
                # The BEFORE_WITH stack spill is valid only at the formatter entry.
                if report["phase"] == "pre_exception_format":
                    actions.append(("value_stack", live_stack))
                for name, action in actions:
                    try:
                        report[name] = action(current)
                    except Exception as error:
                        report[name + "_error"] = str(error)
                target = report.get("target_frame")
                if target and current["frame"] != target["frame"]:
                    try:
                        report["target_code_snapshot"] = code_snapshot(target)
                    except Exception as error:
                        report["target_code_snapshot_error"] = str(error)
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
