"""Opt-in reward telemetry, materialized once before the native batch pool exists."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from unilab.base.backend.mujoco.xml import _mujoco_module, _write_temp_xml


def inject_reward_telemetry(model_file: str) -> str:
    mujoco = _mujoco_module()
    model = mujoco.MjModel.from_xml_path(model_file)
    tree = ET.parse(model_file)
    sensor = tree.getroot().find("sensor")
    if sensor is None:
        sensor = ET.SubElement(tree.getroot(), "sensor")
    if any(child.get("name", "").startswith("mq_") for child in sensor):
        raise ValueError("reward telemetry sensor namespace is already occupied")
    joint_ids = model.actuator_trnid[:, 0]
    root_width = 6 if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
    if (
        np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
        or not np.array_equal(model.jnt_dofadr[joint_ids], root_width + np.arange(model.nu))
        or not np.all(model.actuator_gear[:, 0] == 1)
        or np.any(model.actuator_gear[:, 1:] != 0)
    ):
        raise ValueError("reward telemetry requires unit-gear actuators in joint velocity order")
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            raise ValueError("reward contact telemetry requires named bodies")
        ET.SubElement(sensor, "contact", name=f"mq_contact_{body_id}",
                      body1=name, data="force", num="1", reduce="netforce")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None:
            raise ValueError("reward torque telemetry requires named actuators")
        ET.SubElement(sensor, "actuatorfrc", name=f"mq_torque_{actuator_id}", actuator=name)
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is None:
            raise ValueError("reward site telemetry requires named sites")
        ET.SubElement(sensor, "framelinvel", name=f"mq_site_vel_{site_id}", objtype="site", objname=name)
    return _write_temp_xml(tree, model_file)
