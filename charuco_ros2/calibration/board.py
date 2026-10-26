"""ChArUco calibration board definition and detection (no ROS)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from charuco_ros2.charuco_board_utils import get_aruco_dictionary

BOARD_DIR = Path(__file__).resolve().parent / "boards"
DEFAULT_BOARD_FILE = BOARD_DIR / "a4_charuco_9x6_id32-58.yaml"


@dataclass
class CalibrationBoard:
    """Board definition loaded from YAML. Units: metres."""

    name: str
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    marker_ids: tuple
    sha256: str
    path: str

    @classmethod
    def load(cls, path=DEFAULT_BOARD_FILE) -> "CalibrationBoard":
        path = Path(path)
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
        squares_x, squares_y = int(data["squares_x"]), int(data["squares_y"])
        square, marker = float(data["square_length_m"]), float(data["marker_length_m"])
        ids = tuple(int(v) for v in data["marker_ids"])
        if squares_x < 3 or squares_y < 3 or not 0.0 < marker < square:
            raise ValueError(f"invalid board geometry in {path}")
        if len(ids) != (squares_x * squares_y) // 2 or len(set(ids)) != len(ids):
            raise ValueError(
                f"{path}: need {(squares_x * squares_y) // 2} unique marker IDs, got {len(ids)}"
            )
        return cls(
            str(data["name"]),
            str(data["dictionary"]),
            squares_x,
            squares_y,
            square,
            marker,
            ids,
            hashlib.sha256(raw).hexdigest(),
            str(path.resolve()),
        )

    @property
    def inner_columns(self) -> int:
        return self.squares_x - 1

    @property
    def inner_rows(self) -> int:
        return self.squares_y - 1

    @property
    def centre_m(self) -> np.ndarray:
        return np.array(
            [
                self.squares_x * self.square_length_m / 2,
                self.squares_y * self.square_length_m / 2,
                0.0,
            ]
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "dictionary": self.dictionary_name,
            "squares": [self.squares_x, self.squares_y],
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "marker_ids": [self.marker_ids[0], self.marker_ids[-1]],
            "sha256": self.sha256,
        }


class BoardDetector:
    """OpenCV default ArUco/ChArUco detection restricted to the board IDs (ARGUS Plan012/020)."""

    def __init__(self, board: CalibrationBoard):
        self.board = board
        self.dictionary = get_aruco_dictionary(board.dictionary_name)
        self.cv_board = cv2.aruco.CharucoBoard(
            (board.squares_x, board.squares_y),
            board.square_length_m,
            board.marker_length_m,
            self.dictionary,
            np.asarray(board.marker_ids, np.int32),
        )
        self.corners_3d = np.asarray(self.cv_board.getChessboardCorners(), np.float64)
        self.marker_detector = cv2.aruco.ArucoDetector(
            self.dictionary, cv2.aruco.DetectorParameters()
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(self.cv_board)
        self.allowed = set(board.marker_ids)

    def objects(self, ids) -> np.ndarray:
        return self.corners_3d[np.asarray(list(ids), int)]

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (charuco_ids[N], charuco_points[N,2]) in pixels; empty arrays if none."""
        empty = (np.zeros((0,), int), np.zeros((0, 2), float))
        corners, ids, _ = self.marker_detector.detectMarkers(gray)
        if ids is None:
            return empty
        keep = [i for i, v in enumerate(ids.reshape(-1)) if int(v) in self.allowed]
        if not keep:
            return empty
        try:
            points, corner_ids, _, _ = self.charuco_detector.detectBoard(
                gray,
                markerCorners=tuple(corners[i] for i in keep),
                markerIds=ids[keep].reshape(-1, 1),
            )
        except cv2.error:
            return empty
        if corner_ids is None or points is None:
            return empty
        return np.asarray(corner_ids, int).reshape(-1), np.asarray(points, np.float64).reshape(
            -1, 2
        )
