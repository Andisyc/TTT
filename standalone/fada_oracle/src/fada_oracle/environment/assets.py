"""Materialize an independently owned torque-controlled copy of the source scene."""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedScene:
    scene_file: Path
    source_sha256: str
    transformed_sha256: str
    provenance_file: Path


def _xml_sha256(scene: Path) -> str:
    paths = sorted(path for path in scene.parent.rglob("*.xml") if path.is_file())
    if scene not in paths:
        paths.append(scene)
        paths.sort()
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(scene.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reachable_xml(scene: Path) -> list[tuple[Path, ET.ElementTree]]:
    """Parse only the scene's include closure, leaving historical XML untouched."""
    root = scene.parent
    pending = [scene]
    parsed = {}
    while pending:
        path = pending.pop()
        if path in parsed:
            continue
        tree = ET.parse(path)
        parsed[path] = tree
        for include in tree.getroot().iter("include"):
            filename = include.get("file")
            if not filename:
                raise ValueError(f"Scene include has no file attribute: {path}")
            included = (path.parent / filename).resolve(strict=True)
            if not included.is_relative_to(root):
                raise ValueError(f"Scene include escapes the owned asset tree: {included}")
            pending.append(included)
    return sorted(parsed.items())


def materialize_scene(source_scene: Path, target_dir: Path, metadata: dict) -> PreparedScene:
    """Copy every source asset, replacing only position actuator KP/KD shortcuts.

    ``source_sha256`` is the checkpoint's original XML-tree identity; the separate
    transformed hash is never presented as that identity. No source file is edited.
    """
    source_scene = Path(source_scene).resolve(strict=True)
    source_root = source_scene.parent
    target_dir = Path(target_dir).resolve()
    if not source_scene.is_file():
        raise ValueError(f"Source scene is not a file: {source_scene}")
    if target_dir.is_relative_to(source_root) or source_root.is_relative_to(target_dir):
        raise ValueError("Source and target asset trees must be disjoint (no aliases or recursion)")
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Source asset tree must not contain symlinks: {path}")
    source_hash = _xml_sha256(source_scene)
    if metadata.get("asset_sha256") != source_hash:
        raise ValueError("Source XML asset_sha256 does not match checkpoint metadata")
    scene_file = target_dir / source_scene.name
    provenance_file = target_dir / "scene_provenance.json"
    if target_dir.exists() and any(target_dir.iterdir()):
        if not provenance_file.is_file():
            raise ValueError(f"Refusing to overwrite a nonempty asset directory: {target_dir}")
        provenance = json.loads(provenance_file.read_text())
        if (
            provenance.get("source_sha256") != source_hash
            or provenance.get("scene_file") != source_scene.name
            or provenance.get("transformation") != "reachable_position_to_motor_remove_kp_kv"
            or not scene_file.is_file()
            or provenance.get("transformed_sha256") != _xml_sha256(scene_file)
        ):
            raise ValueError("Existing prepared scene provenance does not match source/transformed assets")
        return PreparedScene(scene_file, source_hash, provenance["transformed_sha256"], provenance_file)
    reachable_xml = _reachable_xml(source_scene)
    shutil.copytree(source_root, target_dir, dirs_exist_ok=True)
    transformed_files = []
    for source_path, tree in reachable_xml:
        path = target_dir / source_path.relative_to(source_root)
        changed = False
        for actuator in tree.getroot().findall(".//actuator/position"):
            actuator.tag = "motor"
            actuator.attrib.pop("kp", None)
            actuator.attrib.pop("kv", None)
            changed = True
        if changed:
            tree.write(path, encoding="utf-8", xml_declaration=True)
            transformed_files.append(path.relative_to(target_dir).as_posix())
    if not transformed_files:
        raise ValueError("Source scene tree has no position actuators to convert")
    transformed_hash = _xml_sha256(scene_file)
    provenance_file.write_text(json.dumps({
        "source_scene": str(source_scene),
        "scene_file": source_scene.name,
        "source_sha256": source_hash,
        "transformed_sha256": transformed_hash,
        "transformation": "reachable_position_to_motor_remove_kp_kv",
        "transformed_files": transformed_files,
    }, indent=2) + "\n")
    return PreparedScene(scene_file, source_hash, transformed_hash, provenance_file)
