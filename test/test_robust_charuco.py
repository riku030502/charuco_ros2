"""Unit tests for charuco_ros2.robust_charuco and the pose camera model guard (no ROS graph)."""

import os
import tempfile

import cv2
import numpy as np

from charuco_ros2.multi_cube_charuco_detector import camera_model_matches, load_pose_camera_model
from charuco_ros2.robust_charuco import (
    DECODE_P1,
    RobustCharucoDetector,
    ippe_lm,
    scaled_camera_matrix,
)

K = np.array([[1360.6, 0.0, 972.0], [0.0, 1359.4, 550.5], [0.0, 0.0, 1.0]])
D = np.zeros(5)
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)


def board(first_id):
    ids = np.arange(first_id, first_id + 8, dtype=np.int32)
    return cv2.aruco.CharucoBoard((4, 4), 0.010, 0.007, DICTIONARY, ids)


def render(face_board, rvec, tvec, size=(1920, 1080)):
    """Render a board (40 mm) seen by K at pose (rvec, tvec) onto a grey 1920x1080 image."""
    texture = face_board.generateImage((800, 800), marginSize=0, borderBits=1)
    pixels_per_m = 800 / 0.040
    corners_board = np.array([[0, 0, 0], [0.04, 0, 0], [0.04, 0.04, 0], [0, 0.04, 0]], float)
    corners_image = cv2.projectPoints(corners_board, rvec, tvec, K, D)[0].reshape(-1, 2)
    source = (corners_board[:, :2] * pixels_per_m).astype(np.float32)
    homography = cv2.getPerspectiveTransform(source, corners_image.astype(np.float32))
    canvas = cv2.warpPerspective(texture, homography, size, flags=cv2.INTER_AREA, borderValue=128)
    mask = cv2.warpPerspective(np.full_like(texture, 255), homography, size, borderValue=0)
    image = np.where(mask > 0, canvas, 128).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_scaled_camera_matrix_half_pixel_rule():
    scaled = scaled_camera_matrix(K, 2.0)
    assert scaled[0, 0] == 2 * K[0, 0]
    assert scaled[0, 2] == (K[0, 2] + 0.5) * 2 - 0.5


def test_p1_decode_is_held_by_every_branch_detector():
    detector = RobustCharucoDetector({"front": board(0)}, DICTIONARY, K, D, 1920, 1080)
    values = detector.effective_decode()
    assert len(values) == 3 * 2
    for _, margin, pixels in values:
        assert abs(margin - DECODE_P1["perspectiveRemoveIgnoredMarginPerCell"]) < 1e-6
        assert pixels == DECODE_P1["perspectiveRemovePixelPerCell"]


def test_synthetic_face_pose_is_recovered_and_quality_passes():
    rvec = np.array([[0.35], [-0.25], [0.05]])
    tvec = np.array([[-0.02], [0.01], [0.22]])
    image = render(board(8), rvec, tvec)
    detector = RobustCharucoDetector(
        {"left": board(8), "front": board(0)}, DICTIONARY, K, D, 1920, 1080
    )
    results = detector.detect_faces(image, ["left", "front"])
    left = results["left"]
    assert left.quality_pass, left.reason
    assert len(left.charuco_ids) == 9
    assert np.linalg.norm(left.tvec.reshape(3) - tvec.reshape(3)) < 0.0005
    delta = cv2.Rodrigues(left.rvec)[0].T @ cv2.Rodrigues(rvec)[0]
    # 9 corners on a 40 mm board: rendering/sub-pixel noise gives a few tenths of a degree
    assert np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))) < 1.0
    assert not results["front"].marker_ids
    assert not results["front"].quality_pass


def test_marker_only_view_never_passes_quality():
    detector = RobustCharucoDetector({"left": board(8)}, DICTIONARY, K, D, 1920, 1080)
    # far away: markers may still decode but too few ChArUco corners -> no accepted pose
    image = render(board(8), np.array([[0.2], [0.1], [0.0]]), np.array([[0.0], [0.0], [1.6]]))
    left = detector.detect_faces(image, ["left"])["left"]
    assert not left.quality_pass


def test_image_size_mismatch_is_rejected():
    detector = RobustCharucoDetector({"left": board(8)}, DICTIONARY, K, D, 1920, 1080)
    try:
        detector.detect_faces(np.zeros((720, 1280, 3), np.uint8), ["left"])
    except ValueError:
        return
    raise AssertionError("size mismatch accepted")


def test_ippe_lm_rejects_too_few_points():
    assert ippe_lm(np.zeros((3, 3)), np.zeros((3, 2)), K, D) is None


def test_pose_camera_model_guard():
    text = (
        "name: M3\nserial_no: \"337122071608\"\nwidth: 1920\nheight: 1080\n"
        "K: [[1351.0, 0, 975.8], [0, 1349.5, 547.8], [0, 0, 1]]\nD: [0.16, -0.52, 0, 0, 0.52]\n"
        "camera_info_k: [[1360.6, 0, 972.0], [0, 1359.4, 550.5], [0, 0, 1]]\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.yaml")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        model = load_pose_camera_model(path)
    assert model["serial_no"] == "337122071608"
    assert camera_model_matches(model, 1920, 1080, K)
    assert not camera_model_matches(model, 1280, 720, K)
    other = K.copy()
    other[0, 0] = 905.0
    assert not camera_model_matches(model, 1920, 1080, other)
