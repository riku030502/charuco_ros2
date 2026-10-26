"""Pose model selection of multi_cube_charuco_detector: auto/registry, fail closed (no ROS)."""

import numpy as np
import pytest

from charuco_ros2.calibration import registry
from charuco_ros2.multi_cube_charuco_detector import MultiCharucoDetectorNode

K_INFO_NEW = np.array([[1380.0, 0.0, 960.0], [0.0, 1380.0, 540.0], [0.0, 0.0, 1.0]])
K_INFO_OLD = np.array([[1360.6, 0.0, 972.0], [0.0, 1359.4, 550.5], [0.0, 0.0, 1.0]])


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(("info", text))

    def warn(self, text):
        self.lines.append(("warn", text))

    def error(self, text):
        self.lines.append(("error", text))


def doc(serial, k_info, name="M3", width=1920, height=1080):
    return {
        "name": name,
        "status": "VALIDATED",
        "serial_no": serial,
        "width": width,
        "height": height,
        "stream": "color",
        "image_processing": "raw",
        "distortion_model": "plumb_bob",
        "K": (k_info * [[0.99, 1, 1.001], [1, 0.99, 1.001], [1, 1, 1]]).tolist(),
        "D": [0.1, -0.2, 0.0, 0.0, 0.05],
        "camera_info_k": k_info.tolist(),
    }


def node(
    tmp_path,
    serial,
    k_info,
    mismatch="fail",
    width=1920,
    height=1080,
    topic="/camera/hand_camera/color/image_raw",
):
    item = MultiCharucoDetectorNode.__new__(MultiCharucoDetectorNode)
    logger = Logger()
    item.get_logger = lambda: logger
    item.pose_camera_model_auto = True
    item.pose_camera_model_registry = [tmp_path / "user", tmp_path / "shipped"]
    item.pose_camera_model_file = "auto"
    item.pose_camera_model_mismatch = mismatch
    item.pose_camera_model = None
    item.pose_pipeline = "legacy"
    item.serial_mismatch = False
    item.realsense_serial_no = serial
    item.realsense_node_name = "/camera/hand_camera"
    item.image_topic = topic
    item.camera_info_width, item.camera_info_height = width, height
    item.camera_matrix, item.dist_coeffs = k_info, np.zeros(5)
    return item, logger


@pytest.fixture
def registries(tmp_path):
    registry.write_model_yaml(tmp_path / "user/new.yaml", doc("819312073416", K_INFO_NEW))
    registry.write_model_yaml(tmp_path / "shipped/old.yaml", doc("337122071608", K_INFO_OLD))
    return tmp_path


def test_auto_selects_the_connected_serial(registries):
    for serial, k_info, file in (
        ("819312073416", K_INFO_NEW, "new.yaml"),
        ("337122071608", K_INFO_OLD, "old.yaml"),
    ):
        item, logger = node(registries, serial, k_info)
        item.update_pose_model()
        assert item.active_pose_model["path"].endswith(file)
        assert item.active_pose_model["serial_no"] == serial
        assert any("sha256" in text for _, text in logger.lines)


def test_auto_fails_closed(registries):
    cases = [
        (
            "819312073416",
            K_INFO_NEW,
            1280,
            720,
            "/camera/hand_camera/color/image_raw",
        ),  # unregistered profile
        (
            "999999999999",
            K_INFO_NEW,
            1920,
            1080,
            "/camera/hand_camera/color/image_raw",
        ),  # unregistered serial
        (
            "819312073416",
            K_INFO_OLD,
            1920,
            1080,
            "/camera/hand_camera/color/image_raw",
        ),  # CameraInfo K differs
        (
            "819312073416",
            K_INFO_NEW,
            1920,
            1080,
            "/camera/hand_camera/color/image_rect",
        ),  # rectified input
        (None, K_INFO_NEW, 1920, 1080, "/camera/hand_camera/color/image_raw"),
    ]  # serial not read
    for serial, k_info, width, height, topic in cases:
        item, _ = node(registries, serial, k_info, width=width, height=height, topic=topic)
        item.update_pose_model()
        assert item.active_pose_model is None, (serial, width, topic)
        assert item.pose_model_error


def test_auto_ambiguous_entry_fails(registries):
    registry.write_model_yaml(
        registries / "user/dup.yaml", doc("819312073416", K_INFO_NEW, name="M2")
    )
    item, _ = node(registries, "819312073416", K_INFO_NEW)
    item.update_pose_model()
    assert item.active_pose_model is None and "AMBIGUOUS" in item.pose_model_error


def test_camera_info_fallback_only_when_opted_in(registries):
    item, logger = node(registries, "999999999999", K_INFO_NEW, mismatch="camera_info")
    item.update_pose_model()
    assert item.active_pose_model["name"] == "camera_info"
    assert any(level == "warn" and "CameraInfo" in text for level, text in logger.lines)
