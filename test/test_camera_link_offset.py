"""front -> camera_link per cube (ARGUS Plan075).

ID 0-39 uses the Plan074 measurement, ID 40-79 the design value.
"""
import numpy as np
import pytest

from charuco_ros2 import multi_cube_charuco_detector as m


def test_default_is_the_design_value_and_rotation_is_unchanged():
    T = m.MultiCharucoDetectorNode._compute_T_camera_link_in_front()
    assert np.allclose(T[:3, 3], [0.040, 0.070, 0.020])
    node = m.MultiCharucoDetectorNode
    T2 = node._compute_T_camera_link_in_front(m.LEFT_CUBE_CAMERA_LINK_IN_FRONT_M)
    assert np.allclose(T2[:3, :3], T[:3, :3])
    assert np.allclose(T2[:3, 3], [0.03879, 0.06852, 0.02256])


def test_values_are_metres():
    assert m.DESIGN_CAMERA_LINK_IN_FRONT_M == (0.040, 0.070, 0.020)
    with pytest.raises(ValueError):
        # millimetres by mistake
        m.MultiCharucoDetectorNode._compute_T_camera_link_in_front([38.79, 68.52, 22.56])
