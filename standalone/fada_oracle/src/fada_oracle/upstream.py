"""Provision and lazily source-load the independently pinned official FADA tree."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

UPSTREAM_REVISION = "290b3a9c5c0ebe2e9851b61350617544be05e9bc"
WARP_REVISION = "09ec1da65ac44bd68703f9507c9c9e8f22ae7c71"
_UPSTREAM_URL = "https://github.com/LeCAR-Lab/FADA.git"
_PATCH_NAME = "0001-runtime-factories.patch"
# Only these exact original/patched file contents are authorized at the pinned commit.
_FILE_HASHES = {
    "src/holosoma/holosoma/fada/common/compact_obs.py": (
        "e77ac7ca3a952e2c5c5a70a2b974dffc113feb2c193cd8d5c142df479a48dfb3",
        "5d5a04c97d0c239465b80f8bf28d30841a6dcd186fd6d71ee9e91c2892f8afdc",
    ),
    "src/holosoma/holosoma/fada/planner_idm/trainer.py": (
        "2aa97ce3df8cf1c9fb33fdfe0f81c8fda7df74f9c7a5877fc677a6bdd951c05a",
        "a7ab812f8cfd2edb596738955c2184f8bd45c9696ac0a111aa2a5174e399c465",
    ),
    "src/holosoma/holosoma/fada/planner_idm/trainer_rollout.py": (
        "33c6360709aa88c79ad26d02121163cf1467f7f9136002b4b9407284471b0f2f",
        "706d509544a58a2c6a7a1a4e12a940727713bf7a39ee160f986dd27de2a9525b",
    ),
    "src/holosoma/holosoma/simulator/mujoco/mujoco.py": (
        "be88ae2aae08cade9b7d9e55c5116bcca968e798c030a1cadf9d33d630ff5933",
        "9c8ac23950e6edb5d92f3d18793000cb526fe3767a4c183502b54115c539bc67",
    ),
}


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "http.proxy=", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Official FADA git {' '.join(args)} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def _patch_path() -> Path:
    module_dir = Path(__file__).resolve().parent
    # Wheels carry the patch as package data; editable installs keep it in the project.
    for candidate in (
        module_dir / "upstream_patches" / _PATCH_NAME,
        module_dir.parent.parent / "upstream_patches" / _PATCH_NAME,
    ):
        if candidate.is_file():
            return candidate
    raise RuntimeError("The checked-in official FADA runtime-factory patch is missing")


def _checkout_state(checkout: Path) -> str:
    if checkout.is_symlink() or not (checkout / ".git").is_dir():
        raise RuntimeError(f"Official FADA must be a private git checkout: {checkout}")
    alternates = checkout / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text().strip():
        raise RuntimeError("Official FADA checkout must not borrow another repository's objects")
    if _git(checkout, "rev-parse", "HEAD").strip() != UPSTREAM_REVISION:
        raise RuntimeError(f"Official FADA revision drift; expected {UPSTREAM_REVISION}")
    if _git(checkout, "remote", "get-url", "origin").strip() != _UPSTREAM_URL:
        raise RuntimeError("Official FADA origin drift; refusing to rewrite an existing checkout")
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    for entry in status.split("\0"):
        if entry and (entry[:3] != " M " or entry[3:] not in _FILE_HASHES):
            raise RuntimeError(f"Unknown official FADA checkout drift: {entry!r}")
    if _git(checkout, "diff", "--no-ext-diff", "--summary", "HEAD", "--").strip():
        raise RuntimeError("Unknown official FADA file-mode or path drift")
    states = set()
    for relative, (original, patched) in _FILE_HASHES.items():
        path = checkout / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unknown official FADA source path drift: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == original:
            states.add("original")
        elif actual == patched:
            states.add("patched")
        else:
            raise RuntimeError(f"Unknown official FADA source content drift: {relative}")
    if len(states) != 1:
        raise RuntimeError("Official FADA runtime-factory patch is only partially applied")
    return states.pop()


def prepare_upstream(root: Path | str) -> Path:
    """Return root/.deps/FADA at the exact revision with the checked-in patch applied.

    The checkout has its own object database. Existing revision, index, source,
    or untracked-file drift is rejected rather than reset. No environments or
    dependencies are created here.
    """
    project_root = Path(root).expanduser().resolve()
    deps = project_root / ".deps"
    if deps.is_symlink():
        raise RuntimeError("The standalone dependency directory must not be a symlink")
    deps.mkdir(parents=True, exist_ok=True)
    checkout = deps / "FADA"
    if not checkout.exists():
        _git(deps, "clone", "--filter=blob:none", "--no-checkout", _UPSTREAM_URL, str(checkout))
        _git(checkout, "fetch", "--depth=1", "origin", UPSTREAM_REVISION)
        _git(checkout, "checkout", "--detach", UPSTREAM_REVISION)
    state = _checkout_state(checkout)
    patch = _patch_path()
    if state == "original":
        _git(checkout, "apply", "--check", str(patch))
        _git(checkout, "apply", str(patch))
        if _checkout_state(checkout) != "patched":
            raise RuntimeError("Official FADA runtime-factory patch did not produce its exact source")
    else:
        _git(checkout, "apply", "--reverse", "--check", str(patch))
    return checkout


def _reject_foreign_imports(package_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "holosoma" or name.startswith("holosoma."):
            origin = getattr(module, "__file__", None)
            locations = (origin,) if origin is not None else tuple(getattr(module, "__path__", ()))
            if not locations or any(not Path(path).resolve().is_relative_to(package_root) for path in locations):
                raise RuntimeError(
                    f"{name} is already imported outside the private official FADA checkout; "
                    "start a fresh standalone process"
                )


def import_upstream(root: Path | str) -> SimpleNamespace:
    """Lazily expose official FADA types/functions without constructing an environment.

    Returns FADAConfig, trainer (the module), build_student_policy, ReplayBuffer,
    and config_module. The patched worker accepts an importable runtime_factory
    (or picklable partial of one), called with the original build_env_and_expert
    arguments and returning (env, simulation_app, expert_policy, device, config).
    """
    checkout = prepare_upstream(root)
    source_root = checkout / "src" / "holosoma"
    package_root = source_root / "holosoma"
    _reject_foreign_imports(package_root)
    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    importlib.invalidate_caches()
    config_module = importlib.import_module("holosoma.fada.planner_idm.config")
    trainer = importlib.import_module("holosoma.fada.planner_idm.trainer")
    model = importlib.import_module("holosoma.fada.planner_idm.model")
    dataset = importlib.import_module("holosoma.fada.common.dataset")
    _reject_foreign_imports(package_root)
    return SimpleNamespace(
        FADAConfig=config_module.FADAConfig,
        trainer=trainer,
        build_student_policy=model.build_student_policy,
        ReplayBuffer=dataset.ReplayBuffer,
        config_module=config_module,
    )
