"""Unit tests for charuco_ros2.calibration (no ROS): known K/D, gate, selection, registry."""

import json
import time

import cv2
import numpy as np
import pytest
import yaml

from charuco_ros2.calibration import pipeline, quality, registry, solver
from charuco_ros2.calibration.board import BoardDetector, CalibrationBoard
from charuco_ros2.calibration.capture import Collector
from charuco_ros2.multi_cube_charuco_detector import load_pose_camera_model

cv2.setNumThreads(1)  # one worker, as in the real analysis

SIZE = (640, 480)
K_TRUE = np.array([[610.0, 0.0, 322.0], [0.0, 608.0, 236.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([0.12, -0.25, 0.0004, -0.0003, 0.08])
K0 = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])  # "CameraInfo" guess
MARGIN, PPM = 100, 10000.0  # texture margin (px) and pixels per metre

BOARD = CalibrationBoard.load()
DETECTOR = BoardDetector(BOARD)
TEXTURE = DETECTOR.cv_board.generateImage(
    (int(0.225 * PPM) + 2 * MARGIN, int(0.150 * PPM) + 2 * MARGIN), marginSize=MARGIN
)


def render(rvec, tvec, k=K_TRUE, d=D_TRUE, size=SIZE, supersample=3):
    """Render the A4 board seen through (k, d).

    Sub-pixel samples are undistorted, traced to the board plane and averaged per pixel
    (anti-aliasing; point sampling would bias corner refinement by up to 0.5 px).
    """
    s = supersample
    w, h = size[0] * s, size[1] * s
    sub = (
        lambda n: (np.arange(n) + 0.5) / s - 0.5
    )  # noqa: E731 - sub-sample centres in pixel units
    grid = np.stack(np.meshgrid(sub(w), sub(h)), -1).reshape(-1, 1, 2).astype(np.float64)
    rays = cv2.undistortPointsIter(
        grid, k, d, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    ).reshape(-1, 2)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, float))
    homography = np.column_stack(
        [rotation[:, 0], rotation[:, 1], np.asarray(tvec, float).reshape(3)]
    )
    plane = np.linalg.solve(homography, np.column_stack([rays, np.ones(len(rays))]).T)
    plane = (plane[:2] / plane[2]).T
    map_xy = (plane * PPM + MARGIN - 0.5).astype(np.float32).reshape(h, w, 2)
    image = cv2.remap(TEXTURE, map_xy[..., 0], map_xy[..., 1], cv2.INTER_LINEAR, borderValue=128)
    image = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(np.clip(image + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def poses(n=70, seed=3):
    """Board poses spread over the image: tilts 0-45 deg, rolls, depths 0.30-0.50 m."""
    rng = np.random.default_rng(seed)
    out = []
    targets = [(x, y) for y in np.linspace(0.12, 0.88, 4) for x in np.linspace(0.1, 0.9, 5)]
    for i in range(n):
        u, v = targets[i % len(targets)]
        z = rng.uniform(0.30, 0.50)
        tilt = 0.0 if i % 7 == 0 else rng.uniform(5, 45)
        axis = rng.normal(size=2)
        axis = np.append(axis / np.linalg.norm(axis), 0.0)
        roll = rng.uniform(-30, 30)
        rotation = (
            cv2.Rodrigues(axis * np.radians(tilt))[0]
            @ cv2.Rodrigues(np.array([0.0, 0.0, np.radians(roll)]))[0]
        )
        centre = (
            np.array(
                [
                    (u * SIZE[0] - K_TRUE[0, 2]) / K_TRUE[0, 0],
                    (v * SIZE[1] - K_TRUE[1, 2]) / K_TRUE[1, 1],
                    1.0,
                ]
            )
            * z
        )
        tvec = centre - rotation @ BOARD.centre_m
        out.append((cv2.Rodrigues(rotation)[0].reshape(3), tvec))
    return out


def write_session(path, supersample=2):
    """Write a synthetic capture session exactly as the live collector lays it out."""
    (path / "poses").mkdir()
    lines = []
    for index, (rvec, tvec) in enumerate(poses()):
        image = render(rvec, tvec, supersample=supersample)
        cv2.imwrite(str(path / "poses" / f"pose_{index:03d}.png"), image)
        lines.append(json.dumps({"pose_index": index, "file": f"pose_{index:03d}.png"}))
    (path / "poses.jsonl").write_text("\n".join(lines) + "\n")
    camera = {
        "width": SIZE[0],
        "height": SIZE[1],
        "k": K0.reshape(-1).tolist(),
        "d": [0.0] * 5,
        "distortion_model": "plumb_bob",
        "frame_id": "synthetic_optical",
    }
    (path / "camera_info.json").write_text(json.dumps(camera))
    meta = {
        "serial_no": "000000000001",
        "camera_name": "synthetic",
        "board": {"path": BOARD.path, **BOARD.summary()},
        "image_topic": "/synthetic/color/image_raw",
        "camera_info_topic": "/synthetic/color/camera_info",
        "color_profile": "640x480x30",
    }
    (path / "session.json").write_text(json.dumps(meta))
    return path


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    return write_session(tmp_path_factory.mktemp("session"))


def test_board_matches_plan012_definition():
    assert (BOARD.squares_x, BOARD.squares_y) == (9, 6)
    assert BOARD.marker_ids == tuple(range(32, 59))
    assert DETECTOR.corners_3d.shape == (40, 3)
    assert np.allclose(DETECTOR.corners_3d[39], [0.200, 0.125, 0.0])


def test_detects_all_corners_on_a_rendered_frontal_board():
    image = render(np.zeros(3), np.array([-0.1125, -0.075, 0.40]))
    ids, points = DETECTOR.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert len(ids) == 40
    projected = cv2.projectPoints(
        DETECTOR.objects(ids), np.zeros(3), np.array([-0.1125, -0.075, 0.40]), K_TRUE, D_TRUE
    )[0].reshape(-1, 2)
    shift = np.median(points - projected, axis=0)
    # OpenCV CharucoDetector adds an extra (+0.5, +0.5) px after cornerSubPix
    # (opencv/opencv#25539, fixed on 5.x / milestone 4.14). Models are calibrated with the
    # same detector as production PnP, so M3's cx/cy absorb it. If this changes after an
    # OpenCV upgrade, registered models must be re-validated.
    assert np.allclose(shift, 0.5, atol=0.1) or np.allclose(shift, 0.0, atol=0.1), shift
    assert np.median(np.linalg.norm(points - projected - shift, axis=1)) < 0.15


def test_edge_width_of_a_blurred_step():
    image = np.zeros((200, 200), np.uint8)
    image[:, 100:] = 220
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5).astype(np.float32)
    ids = [0, 1]  # one horizontal pair across the vertical edge
    points = np.array([[100.0, 60.0], [100.0, 140.0]])
    result = quality.sharpness(blurred, ids, points, columns=1, rows=2)
    # 10-90 width of a Gaussian edge is 2.563 sigma
    assert result["valid_fraction"] == 1.0
    assert abs(result["width_median_px"] - 2.563 * 1.5) < 0.4


def test_split_and_novelty():
    assert [quality.split_of(i) for i in range(6)] == ["validation"] + ["calibration"] * 4 + [
        "validation"
    ]
    accepted = [(np.zeros(3), np.zeros(3))]
    assert not quality.novel(np.zeros(3), np.array([0.01, 0, 0]), accepted)
    assert quality.novel(np.zeros(3), np.array([0.031, 0, 0]), accepted)
    assert quality.novel(np.array([np.radians(9), 0, 0]), np.zeros(3), accepted)


def test_gate_refuses_too_few_poses():
    rows = [
        {
            "pose_index": i,
            "charuco_n": 40,
            "charuco_points": [[320, 240]] * 40,
            "width_median_px": 1.0,
            "valid_fraction": 1.0,
            "tilt_deg": 0.0,
        }
        for i in range(10)
    ]
    gate = quality.data_gate(rows, 640, 480, (320, 240))
    assert not gate["pass"]
    assert not gate["checks"]["admitted_poses"]["pass"]
    assert not gate["checks"]["outer_corners_calibration"]["pass"]


@pytest.fixture(scope="module")
def report(session):
    return pipeline.analyse(session, bootstrap_n=20, log=lambda *_: None)


def test_end_to_end_recovers_known_intrinsics(session, report):
    assert report["gate"]["pass"], json.dumps(
        json.loads((session / "analysis/gate.json").read_text())["checks"]
    )
    m3 = report["parameters"]["M3"]
    assert abs(m3["fx"] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.005
    assert abs(m3["cx"] - K_TRUE[0, 2]) < 2.0
    assert abs(m3["k1"] - D_TRUE[0]) < 0.03
    assert (
        report["heldout"]["M3"] < 0.1 and report["heldout"]["M3"] < 0.5 * report["heldout"]["M0"]
    )
    assert report["status"] == "PASS" and report["selected"] == "M3"
    model = registry.parse_model(session / "candidate_model.yaml")
    assert model["status"] == "VALIDATED" and model["serial_no"] == "000000000001"
    assert np.allclose(model["camera_info_k"], K0)
    legacy = load_pose_camera_model(str(session / "candidate_model.yaml"))
    assert legacy["name"] == "M3" and legacy["width"] == SIZE[0]
    for name in ("corner_coverage.png", "radial_distortion.png", "heldout_residuals.png"):
        assert (session / "analysis/figures" / name).stat().st_size > 1000


def test_rerun_is_reproducible(session, report):
    again = pipeline.analyse(session, bootstrap_n=20, log=lambda *_: None)
    assert again["parameters"] == report["parameters"]
    assert again["heldout"] == report["heldout"]


def test_insufficient_data_blocks_calibration(tmp_path, session):
    for name in ("session.json", "camera_info.json", "poses.jsonl"):
        (tmp_path / name).write_text((session / name).read_text())
    (tmp_path / "poses").mkdir()
    for path in sorted((session / "poses").glob("*.png"))[:15]:
        (tmp_path / "poses" / path.name).write_bytes(path.read_bytes())
    result = pipeline.analyse(tmp_path, bootstrap_n=5, log=lambda *_: None)
    assert result["status"] == "INSUFFICIENT_DATA" and result["selected"] == "M0"
    assert "admitted_poses" in result["failed_gate_rules"]
    assert not (tmp_path / "candidate_model.yaml").exists()


def evaluation(median, p95, views=16):
    return {"views": views, "rmse_median_px": median, "rmse_p95_px": p95}


STABLE = {
    "relative_std_fx": 0.001,
    "relative_std_fy": 0.001,
    "bootstrap_width_rel_fx": 0.01,
    "bootstrap_width_rel_fy": 0.01,
    "bootstrap_width_k1": 0.03,
}


def test_selection_rule():
    finite = {"M2": True, "M3": True}
    both = solver.select(
        {
            "M0": evaluation(0.40, 0.77),
            "M2": evaluation(0.2375, 0.578),
            "M3": evaluation(0.2325, 0.582),
        },
        {"M2": STABLE, "M3": STABLE},
        finite,
    )
    assert (
        both["selected"] == "M3" and both["status"] == "PASS"
    )  # Plan020 numbers: M3 better by 2.1 %
    tie = solver.select(
        {"M0": evaluation(0.40, 0.77), "M2": evaluation(0.240, 0.5), "M3": evaluation(0.238, 0.5)},
        {"M2": STABLE, "M3": STABLE},
        finite,
    )
    assert tie["selected"] == "M2"
    none = solver.select(
        {"M0": evaluation(0.40, 0.77), "M2": evaluation(0.36, 0.7), "M3": evaluation(0.35, 0.7)},
        {"M2": STABLE, "M3": STABLE},
        finite,
    )
    assert none["selected"] == "M0" and none["status"] == "NO_IMPROVEMENT"
    unstable = solver.select(
        {"M0": evaluation(0.40, 0.77), "M2": evaluation(0.36, 0.7), "M3": evaluation(0.2, 0.5)},
        {"M2": STABLE, "M3": dict(STABLE, bootstrap_width_rel_fx=0.05)},
        finite,
    )
    assert unstable["selected"] == "M0"
    few = solver.select(
        {
            "M0": evaluation(0.4, 0.7, views=5),
            "M2": evaluation(0.2, 0.5, 5),
            "M3": evaluation(0.2, 0.5, 5),
        },
        {"M2": STABLE, "M3": STABLE},
        finite,
    )
    assert few["status"] == "INSUFFICIENT_DATA"


def model_doc(serial="819312073416", width=1920, height=1080, name="M3", status="VALIDATED"):
    return {
        "name": name,
        "status": status,
        "serial_no": serial,
        "width": width,
        "height": height,
        "stream": "color",
        "image_processing": "raw",
        "distortion_model": "plumb_bob",
        "K": np.eye(3).tolist(),
        "D": [0.1, 0, 0, 0, 0],
        "camera_info_k": np.eye(3).tolist(),
    }


OLD_M3 = """name: M3
serial_no: "337122071608"
width: 1920
height: 1080
distortion_model: plumb_bob
K: [[1351.0, 0.0, 975.8], [0.0, 1349.5, 547.8], [0.0, 0.0, 1.0]]
D: [0.16, -0.52, 0.0, 0.0002, 0.52]
camera_info_k: [[1360.6, 0.0, 972.0], [0.0, 1359.4, 550.5], [0.0, 0.0, 1.0]]
source: "plan/history/020/tables/phase2_models.json, model M3"
"""


def test_registry_selects_by_serial_and_profile(tmp_path):
    user, shipped = tmp_path / "user", tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "old.yaml").write_text(OLD_M3)
    registry.write_model_yaml(user / "new.yaml", model_doc())
    dirs = [user, shipped]
    assert registry.find_model("_819312073416", 1920, 1080, dirs=dirs)["path"].endswith("new.yaml")
    assert registry.find_model("337122071608", 1920, 1080, dirs=dirs)["name"] == "M3"
    for serial, size in (("819312073416", (1280, 720)), ("999", (1920, 1080)), ("", (1920, 1080))):
        with pytest.raises(registry.RegistryError) as error:
            registry.find_model(serial, *size, dirs=dirs)
        assert error.value.code == "NOT_REGISTERED"
    with pytest.raises(registry.RegistryError) as error:
        registry.find_model("819312073416", 1920, 1080, processing="rectified", dirs=dirs)
    assert error.value.code == "NOT_REGISTERED"


def test_registry_fails_closed(tmp_path):
    registry.write_model_yaml(tmp_path / "a.yaml", model_doc())
    registry.write_model_yaml(tmp_path / "b.yaml", model_doc(name="M2"))
    with pytest.raises(registry.RegistryError) as error:
        registry.find_model("819312073416", 1920, 1080, dirs=[tmp_path])
    assert error.value.code == "AMBIGUOUS"
    (tmp_path / "b.yaml").unlink()
    registry.write_model_yaml(
        tmp_path / "c.yaml", model_doc(serial="1", status="INSUFFICIENT_DATA")
    )
    with pytest.raises(registry.RegistryError) as error:
        registry.find_model("1", 1920, 1080, dirs=[tmp_path])
    assert error.value.code == "NOT_VALIDATED"
    (tmp_path / "d.yaml").write_text("name: [broken")
    with pytest.raises(registry.RegistryError) as error:
        registry.find_model("819312073416", 1920, 1080, dirs=[tmp_path])
    assert error.value.code == "CORRUPT"


def test_register_never_overwrites_silently(tmp_path):
    reg = tmp_path / "registry"
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    registry.write_model_yaml(first, model_doc(name="M2"))
    registry.write_model_yaml(second, model_doc(name="M3"))
    result = registry.register(first, [reg])
    assert registry.find_model("819312073416", 1920, 1080, dirs=[reg])["name"] == "M2"
    with pytest.raises(registry.RegistryError) as error:
        registry.register(second, [reg])
    assert error.value.code == "EXISTS"
    replaced = registry.register(second, [reg], replace=True)
    assert registry.find_model("819312073416", 1920, 1080, dirs=[reg])["name"] == "M3"
    assert len(replaced["replaced_backups"]) == 1
    assert yaml.safe_load(open(replaced["replaced_backups"][0]))["name"] == "M2"
    assert result["path"] == replaced["path"]
    bad = tmp_path / "bad.yaml"
    registry.write_model_yaml(bad, model_doc(serial="2", status="INSUFFICIENT_DATA"))
    with pytest.raises(registry.RegistryError):
        registry.register(bad, [reg])


def test_registry_dirs_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv(registry.ENV_VAR, raising=False)
    assert registry.registry_dirs()[0] == registry.DEFAULT_DIR.expanduser()
    monkeypatch.setenv(registry.ENV_VAR, f"{tmp_path}/a:{tmp_path}/b")
    assert registry.registry_dirs() == [tmp_path / "a", tmp_path / "b"]
    assert registry.registry_dirs(f"{tmp_path}/x,{tmp_path}/y") == [tmp_path / "x", tmp_path / "y"]


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_collector_saves_one_still_pose_and_stops_on_camera_change(tmp_path):
    camera = {
        "width": SIZE[0],
        "height": SIZE[1],
        "k": K0.reshape(-1).tolist(),
        "d": [0.0] * 5,
        "distortion_model": "plumb_bob",
        "frame_id": "f",
    }
    collector = Collector(tmp_path, DETECTOR, target=80, maximum=100)
    image = render(np.zeros(3), np.array([-0.1125, -0.075, 0.40]))
    try:
        for i in range(8):
            collector.push(image, camera, float(i))
            assert wait_for(lambda: collector.latest is None or collector.view is not None)
            time.sleep(0.15)
        assert wait_for(lambda: len(collector.records) == 1)
        time.sleep(0.3)
        assert len(collector.records) == 1  # the same still pose is not counted again
        assert (tmp_path / "poses/pose_000.png").exists()
        collector.push(image, dict(camera, k=(K0 * 1.01).reshape(-1).tolist()), 9.0)
        assert wait_for(lambda: collector.error is not None)
        assert "CameraInfo" in collector.error
    finally:
        collector.stop()
