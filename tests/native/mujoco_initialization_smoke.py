"""Check XML sensor serialization and one official Oracle env; never step/train."""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from hydra import compose, initialize_config_dir
import mujoco
import numpy as np

from unilab.base.registry import ensure_registries
from unilab.training import create_env
from unilab.training.offpolicy.factory import build_offpolicy_env_cfg_override


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    spec = mujoco.MjSpec.from_string(
        '<mujoco model="xml_probe"><worldbody><body name="body">'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    spec.add_sensor(
        name="probe_position", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_XBODY, objname="body",
    )
    document = ET.fromstring(spec.to_xml())
    sensor = document.find("sensor/framepos")
    assert sensor is not None and sensor.get("name") == "probe_position"
    assert sensor.get("objname") == "body"
    print("PASS: MjSpec sensor XML export and reparse", flush=True)

    ensure_registries()
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=[
            "task=sac/g1_walk_flat/mujoco_motion_quality_oracle",
            "training.device=cpu",
        ])
    env = create_env(
        cfg, num_envs=1,
        env_cfg_override=build_offpolicy_env_cfg_override("sac", cfg),
    )
    if env is None:
        raise RuntimeError("official Oracle environment construction returned None")
    try:
        state = env.reset_all()
        obs, info = state.obs, state.info
        assert isinstance(obs, dict) and isinstance(info, dict)
        assert obs["obs"].shape == (1, 98)
        assert np.isfinite(obs["obs"]).all()
    finally:
        env.close()
    print("PASS: official Oracle single-env initialization/reset/close; no step or training", flush=True)


if __name__ == "__main__":
    main()
