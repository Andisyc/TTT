"""Long-route videos must not put the scene beyond the camera's far plane."""
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.fada_rendering import FADADemoRenderConfig, FADAPlaybackRendering

mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("distance", [None, 40.0])
def test_long_route_fits_within_camera_far_plane(distance):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><statistic extent="0.35"/><worldbody>'
        '<geom type="plane" size="0 0 0.05"/></worldbody></mujoco>'
    )
    rendering = object.__new__(FADAPlaybackRendering)
    rendering.model = model
    rendering.session = SimpleNamespace(env=None)
    rendering.config = FADADemoRenderConfig(enabled=True)
    rendering.args = SimpleNamespace(
        camera_distance=distance, camera_follow_body=False,
        camera_elevation=-70.0, camera_azimuth=90.0,
    )
    rendering.reference_route = np.array([[0.0, 0.0, 0.015], [13.0, 0.0, 0.015]])
    old_far_m = float(model.vis.map.zfar * model.stat.extent)
    camera = mujoco.MjvCamera()
    rendering.initialize_camera(camera)
    # The farthest route endpoint must remain visible, even before adding the robot's height.
    endpoint_radius = np.linalg.norm(rendering.reference_route[-1] - camera.lookat)
    assert old_far_m < camera.distance
    assert model.vis.map.zfar * model.stat.extent > camera.distance + endpoint_radius
    if distance is not None:
        assert camera.distance == distance
