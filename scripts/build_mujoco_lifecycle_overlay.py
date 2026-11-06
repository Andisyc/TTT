"""Build and exercise the pinned MuJoCo lifecycle patch without changing the venv.

Run with ``uv run --frozen --no-sync python scripts/build_mujoco_lifecycle_overlay.py
--work-dir /absolute/scratch/path``. The resulting PYTHONPATH overlay is opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://files.pythonhosted.org/packages/d7/af/"
    "c042c865b78400d1309736903168a07dea67d84872a9b999bc19328de412/"
    "mujoco_uni-3.8.0.tar.gz"
)
SOURCE_SHA256 = "8b1e32dce4f6b7f87b4cb7bfa55ab25a9a9e3191b6bde865196eaa04bf1baac5"
ABSEIL_COMMIT = "255c84dadd029fd8ad25c5efb5933e47beaa00c7"
EIGEN_COMMIT = "75bcd155c40cb48e647c87c3f29052360255bc9e"
PYBIND11_COMMIT = "c7fb32eea8c92bebeea9f0735041a72aa20c75f5"


def run(command: list[str], *, timeout: int = 300, **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, timeout=timeout, **kwargs)


def extract(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    children = list(destination.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise RuntimeError(f"expected one source directory in {archive_path}")
    return children[0]


def fetch(url: str, destination: Path) -> Path:
    print(f"Fetching {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("xb") as output:
        shutil.copyfileobj(response, output)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--abseil-source", type=Path)
    parser.add_argument("--eigen-source", type=Path)
    parser.add_argument("--pybind11-source", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not args.work_dir.is_absolute() or args.jobs < 1:
        parser.error("--work-dir must be absolute and --jobs must be positive")
    tools = {name: shutil.which(name) for name in ("cmake", "patch", "c++")}
    if not all(tools.values()):
        raise RuntimeError(f"missing build prerequisite: {tools}")
    import mujoco

    if mujoco.__version__ != "3.8.0":
        raise RuntimeError("this native patch requires the installed MuJoCo 3.8.0 ABI")
    package = Path(mujoco.__file__).resolve().parent
    original_spec = importlib.util.find_spec("mujoco._batch_env")
    if original_spec is None or original_spec.origin is None:
        raise RuntimeError("the original mujoco._batch_env extension is unavailable")
    original_extension = Path(original_spec.origin)
    original_hash = hashlib.sha256(original_extension.read_bytes()).hexdigest()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mujoco-lifecycle-", dir=args.work_dir))
    source_archive = args.source_archive or fetch(SOURCE_URL, work / "mujoco-source.tar.gz")
    if hashlib.sha256(source_archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("MuJoCo source archive differs from the locked production artifact")
    source = extract(source_archive, work / "source")
    run([str(tools["patch"]), "--batch", "--fuzz=0", "-p1", "-i",
         str(ROOT / "patches/mujoco-uni-3.8.0-lifecycle.patch")], cwd=source)

    dependencies = {}
    for name, supplied, url in (
        ("ABSEIL-CPP", args.abseil_source,
         f"https://codeload.github.com/abseil/abseil-cpp/tar.gz/{ABSEIL_COMMIT}"),
        ("EIGEN", args.eigen_source,
         f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_COMMIT}/eigen-{EIGEN_COMMIT}.tar.gz"),
        ("PYBIND11", args.pybind11_source,
         f"https://codeload.github.com/pybind/pybind11/tar.gz/{PYBIND11_COMMIT}"),
    ):
        dependencies[name] = supplied.resolve() if supplied else extract(
            fetch(url, work / f"{name}.tar.gz"), work / name
        )

    build = work / "build"
    configure = [
        str(tools["cmake"]), "-S", str(source / "mujoco"), "-B", str(build),
        f"-DCMAKE_MODULE_PATH={source / 'mujoco/cmake'}",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
        "-DMUJOCO_PYTHON_BUILD_SIMULATE=OFF", "-DBUILD_TESTING=OFF",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPython3_INCLUDE_DIR={sysconfig.get_paths()['include']}",
        f"-DMUJOCO_LIBRARY_DIR={package}", f"-DMUJOCO_INCLUDE_DIR={package / 'include'}",
        f"-DFETCHCONTENT_BASE_DIR={work / 'dependencies-build'}",
    ]
    configure.extend(f"-DFETCHCONTENT_SOURCE_DIR_{name}={path}" for name, path in dependencies.items())
    run(configure)
    # All imported bindings must use the same C++ ABI and exception unwinder.
    targets = ["_callbacks", "_constants", "_enums", "_errors", "_functions",
               "_render", "_rollout", "_specs", "_structs", "_batch_env"]
    run([str(tools["cmake"]), "--build", str(build), "--target", *targets,
         "--parallel", str(args.jobs)], timeout=600)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extensions = []
    for target in targets:
        candidates = list(build.glob(f"{target}*{suffix}"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one built {target} extension, got {candidates}")
        extensions.append(candidates[0])
    rebuilt_names = {extension.name for extension in extensions}
    overlay = work / "overlay"
    overlay_package = overlay / "mujoco"
    overlay_package.mkdir(parents=True)
    for item in package.iterdir():
        if item.name in rebuilt_names or item.name == "__pycache__":
            continue
        (overlay_package / item.name).symlink_to(item, target_is_directory=item.is_dir())
    for extension in extensions:
        shutil.copy2(extension, overlay_package / extension.name)

    executable = work / "threadpool-lifecycle"
    run([str(tools["c++"]), "-std=c++17", "-pthread",
         f"-I{dependencies['ABSEIL-CPP']}", f"-I{source / 'mujoco'}",
         str(ROOT / "tests/native/mujoco_threadpool_lifecycle.cc"),
         str(source / "mujoco/threadpool.cc"), "-o", str(executable)])
    run([str(executable)], timeout=30)
    fault_library = work / "worker-fault.so"
    library = sorted(package.glob("libmujoco.*"))[0]
    run([str(tools["c++"]), "-std=c++17",
         "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
         f"-I{package / 'include'}", str(ROOT / "tests/native/mujoco_worker_fault.cc"),
         str(library), f"-Wl,-rpath,{package}", "-o", str(fault_library)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(overlay) + os.pathsep + environment.get("PYTHONPATH", "")
    run([sys.executable, str(ROOT / "tests/native/mujoco_batch_lifecycle_smoke.py"),
         str(fault_library)], timeout=60, env=environment)
    if hashlib.sha256(original_extension.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("the original native extension changed during the isolated build")
    print(f"VERIFIED_OVERLAY={overlay}", flush=True)
    print(f"ORIGINAL_EXTENSION_UNCHANGED={original_extension}", flush=True)


if __name__ == "__main__":
    main()
