"""Emit JSON training overrides and resource recommendations for oracle_fada.

Only read current capacity; do not enforce limits or launch verification/training.
No Oracle checkpoint or launch command is needed or generated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

GIB = 1024**3

CGROUP_ROOT = Path("/sys/fs/cgroup")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _cgroup_directory() -> Path | None:
    membership = Path("/proc/self/cgroup")
    if membership.is_file():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                path = (CGROUP_ROOT / line[3:].lstrip("/")).resolve()
                if path.is_relative_to(CGROUP_ROOT):
                    return path
    return None


def _cpu_envelope() -> tuple[list[int], int, int]:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        raise SystemExit("Linux CPU affinity support is required")
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(affinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(int((topology / field).read_text()) for field in ("physical_package_id", "core_id"))
        cores.setdefault(key, []).append(cpu)
    if len(cores) < 3:
        raise SystemExit("At least three allowed physical cores are required to reserve two for SSH")
    training_cores = list(cores.values())[:-2]
    selected = [cpu for siblings in training_cores for cpu in siblings]
    return sorted(selected), len(cores), len(training_cores)


def _memory_envelope() -> tuple[int, int, int, int]:
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    group = _cgroup_directory()
    while group is not None and group != CGROUP_ROOT:
        limit_file = group / "memory.max"
        if limit_file.is_file():
            limit = limit_file.read_text().strip()
            if limit != "max":
                used = int((group / "memory.current").read_text())
                available = min(available, max(0, int(limit) - used))
        group = group.parent
    reserve = 8 * GIB
    maximum = (available - reserve) // 4096 * 4096
    if maximum < 8 * GIB:
        raise SystemExit("Insufficient RAM after reserving system headroom; learning budget unchanged")
    high = (maximum * 4 // 5) // 4096 * 4096
    return available, reserve, high, maximum


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("Read-only GPU capacity query requires nvidia-smi")
    train_cpus, physical_cores, train_cores = _cpu_envelope()
    available_ram, reserve, memory_high, memory_max = _memory_envelope()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    gpu_free_mib = int(subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).strip())
    reserve_vram_mib = 2048
    training_vram_mib = gpu_free_mib - reserve_vram_mib
    if training_vram_mib < 4096:
        raise SystemExit("Insufficient free VRAM after reserving headroom; learning budget unchanged")
    cpu_set = ",".join(map(str, train_cpus))
    plan = {
        "config_name": "oracle_fada",
        "training": {
            "fada": {
                "num_envs": train_cores,
                "batch_size": 4096,
                "replay_capacity": 524288,
                "windows_per_iteration": 131072,
                "iterations": 31,
                "idm_updates": 2000,
                "planner_updates": 2000,
            },
        },
        "resources": {
            "allowed_physical_cores": physical_cores,
            "training_physical_cores": train_cores,
            "reserved_physical_cores": physical_cores - train_cores,
            "available_ram_gib": round(available_ram / GIB, 1),
            "reserved_ram_gib": round(reserve / GIB, 1),
            "memory_high_bytes": memory_high,
            "memory_max_bytes": memory_max,
            "memory_swap_max_bytes": 0,
            "gpu": gpu,
            "free_vram_mib": gpu_free_mib,
            "reserved_vram_mib": reserve_vram_mib,
            "training_vram_budget_mib": training_vram_mib,
            "cpu_set": cpu_set,
            "nice": 5,
            "ionice_class": 2,
            "ionice_priority": 5,
            "thread_env": {key: "1" for key in THREAD_ENV},
        },
    }
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
