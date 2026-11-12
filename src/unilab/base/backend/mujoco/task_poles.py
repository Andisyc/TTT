"""Private, cold-path materialization of free physics task poles."""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from unilab.base.scene import ScenePoleCfg


def materialize_task_poles(
    model_file: str, poles: list[ScenePoleCfg], *, robot_root: str | None
) -> tuple[str, tempfile.TemporaryDirectory]:
    """Append props without touching canonical XML or renumbering robot objects.

    Compiling the source first resolves includes and implicit keyframe defaults.
    Every referenced file asset is copied into the private directory, so both
    physics and visual playback remain usable until the backend closes.
    """
    source = Path(model_file).resolve()
    spec = mujoco.MjSpec.from_file(str(source))
    original = spec.compile()
    if robot_root is None:
        robot_root = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_BODY, 1)
    if robot_root is None or mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, robot_root) <= 0:
        raise ValueError("task poles require a named robot root body")
    root = ET.fromstring(spec.to_xml())
    occupied = {element.get("name") for element in root.iter() if element.get("name")}
    sensor_names = {f"task_pole_{kind}_{index}" for index in range(len(poles))
                    for kind in ("pos", "quat", "contact")}
    if occupied & sensor_names:
        raise ValueError("task pole sensor namespace is already occupied")
    occupied.update(sensor_names)
    for index, pole in enumerate(poles):
        names = {pole.name, f"{pole.name}_geom", f"{pole.name}_free"}
        if names & occupied:
            raise ValueError(f"Task pole '{pole.name}' collides with an existing model name")
        occupied.update(names)

    output = tempfile.TemporaryDirectory(prefix="unilab-task-poles-")
    try:
        directory = Path(output.name)
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        assetdir = compiler.get("assetdir", "")
        meshdir = compiler.get("meshdir", assetdir)
        texturedir = compiler.get("texturedir", assetdir)
        files: dict[Path, str] = {}
        for asset in root.iter():
            filename = asset.get("file")
            if filename is None:
                continue
            subdir = meshdir if asset.tag == "mesh" else texturedir if asset.tag == "texture" else assetdir
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / subdir / path
            path = path.resolve()
            if path not in files:
                relative = f"assets/{len(files)}_{path.name}"
                destination = directory / relative
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(path, destination)
                files[path] = relative
            asset.set("file", files[path])
        for attribute in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(attribute, None)
        compiler.set("strippath", "false")

        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        sensor = root.find("sensor")
        if sensor is None:
            sensor = ET.SubElement(root, "sensor")
        for index, pole in enumerate(poles):
            body = ET.SubElement(worldbody, "body", name=pole.name,
                                 pos=" ".join(map(str, pole.position)))
            ET.SubElement(body, "freejoint", name=f"{pole.name}_free")
            ET.SubElement(
                body, "geom", name=f"{pole.name}_geom", type="cylinder",
                size=f"{pole.radius} {pole.height / 2}", mass=str(pole.mass),
                friction=f"{pole.friction} 0.005 0.0001", contype="1", conaffinity="1",
                group="0", rgba="0.95 0.45 0.08 1", margin="0", gap="0",
            )
        # Group sensor kinds contiguously for zero-copy batched pose views.
        for kind, tag in (("pos", "framepos"), ("quat", "framequat")):
            for index, pole in enumerate(poles):
                ET.SubElement(sensor, tag, name=f"task_pole_{kind}_{index}",
                              objtype="xbody", objname=pole.name)
        for index, pole in enumerate(poles):
            ET.SubElement(sensor, "contact", name=f"task_pole_contact_{index}",
                          geom1=f"{pole.name}_geom", subtree2=robot_root,
                          data="found", num="1", reduce="mindist")

        # Full keyframes, including omitted qvel, must cover every free prop.
        prop_qpos = np.asarray([(*pole.position, 1.0, 0.0, 0.0, 0.0) for pole in poles]).ravel()
        for key_id, key in enumerate(root.findall("keyframe/key")):
            qpos = np.concatenate((original.key_qpos[key_id], prop_qpos))
            qvel = np.concatenate((original.key_qvel[key_id], np.zeros(6 * len(poles))))
            key.set("qpos", " ".join(map(str, qpos)))
            key.set("qvel", " ".join(map(str, qvel)))
        path = directory / "scene.xml"
        ET.indent(root, space="  ")
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
        return str(path), output
    except BaseException:
        output.cleanup()
        raise
