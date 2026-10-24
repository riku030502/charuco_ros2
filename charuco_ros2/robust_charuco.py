#!/usr/bin/env python3
"""小型キューブ面ChArUcoの高精度検出・姿勢推定（ARGUS Plan014/015/021/023の凍結手順）。

ROSに依存しない。multi_cube_charuco_detector の ``pose_pipeline=b6_p1_ippe_lm``
から使い、ARGUS側の診断コードとの回帰一致試験にも使う。

検出 (B6 + P1):
  1.5倍cubic → N2(2倍cubic・適応窓7/47/20) → 2倍cubic の順に検出し、
  CameraInfoモデルでIPPEを解いて品質ゲートを最初に通った分岐の角点を採用する。
  どれも通らなければ最初にPnPが解けた分岐、それも無ければ1.5倍。
  全分岐のmarker detector / board detector にP1の復号設定を入れる。
姿勢 (IPPE→LM):
  正深度のIPPE解から再投影RMSE最小を選び solvePnPRefineLM。
  姿勢用のK/Dは検出用(CameraInfo)と別に指定できる（同じ2D点を使う）。
品質ゲート:
  ChArUco内角5点以上、凸包面積比0.001以上、正深度、再投影RMSE 0.8 px以下。
  マーカー角だけの姿勢は使わない（合格に混ぜない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Plan015 P1
DECODE_P1 = {
    "perspectiveRemoveIgnoredMarginPerCell": 0.30,
    "perspectiveRemovePixelPerCell": 4,
}
QUALITY = {"min_points": 5, "min_hull_fraction": 0.001, "max_rmse_px": 0.8}
# name, scale, resize the BGR image before grayscale, adaptive window (min, max, step)
BRANCHES = (
    ("resize_1_5_cubic", 1.5, False, None),
    ("N2", 2.0, True, (7, 47, 20)),
    ("resize_2_cubic", 2.0, False, None),
)
POSE_MIN_POINTS = 4


def production_detector_parameters():
    """multi_cube_charuco_detector の既定DetectorParametersと同一。"""
    params = cv2.aruco.DetectorParameters()
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = False
    params.minSideLengthCanonicalImg = 16
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def scaled_camera_matrix(camera_matrix, scale: float) -> np.ndarray:
    """画素中心の半画素規約で拡大画像用のKを作る。"""
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    if scale == 1.0:
        return matrix
    matrix[0, 0] *= scale
    matrix[1, 1] *= scale
    matrix[0, 2] = (matrix[0, 2] + 0.5) * scale - 0.5
    matrix[1, 2] = (matrix[1, 2] + 0.5) * scale - 0.5
    return matrix


def points_to_original(points, scale: float) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    if scale != 1.0:
        result = (result + 0.5) / scale - 0.5
    return result


def hull_fraction(points, width: int, height: int) -> float:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(array)) / (width * height))


def _positive(object_points, rvec, tvec) -> bool:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    points = np.asarray(object_points, np.float64).reshape(-1, 3).T
    depth = rotation @ points + np.asarray(tvec, np.float64).reshape(3, 1)
    return bool(np.all(depth[2] > 0.0))


def reprojection_rmse(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected = cv2.projectPoints(
        np.asarray(object_points, np.float64),
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )[0].reshape(-1, 2)
    residual = projected - np.asarray(image_points, np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def ippe_best(object_points, image_points, camera_matrix, dist_coeffs):
    """正深度IPPE解のうち再投影RMSE最小。(rvec, tvec, rmse) か None。"""
    obj = np.asarray(object_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    if len(obj) < POSE_MIN_POINTS:
        return None
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return None
    candidates = [
        (reprojection_rmse(obj, img, r, t, camera_matrix, dist_coeffs), r, t)
        for r, t in zip(rvecs, tvecs)
        if _positive(obj, r, t)
    ]
    if not candidates:
        return None
    rmse, rvec, tvec = min(candidates, key=lambda item: item[0])
    return (
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1),
        rmse,
    )


def ippe_lm(object_points, image_points, camera_matrix, dist_coeffs):
    """IPPE(正深度・RMSE最小) → solvePnPRefineLM。(rvec, tvec, rmse) か None。"""
    seed = ippe_best(object_points, image_points, camera_matrix, dist_coeffs)
    if seed is None:
        return None
    obj = np.asarray(object_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj.copy(), img.copy(), camera_matrix, dist_coeffs,
            seed[0].copy(), seed[1].copy(),
        )
    except cv2.error:
        return None
    if not _positive(obj, rvec, tvec):
        return None
    rmse = reprojection_rmse(obj, img, rvec, tvec, camera_matrix, dist_coeffs)
    return rvec.reshape(3, 1), tvec.reshape(3, 1), rmse


@dataclass
class FaceResult:
    """1面の結果。quality_pass のときだけ姿勢を合格として使う。"""

    marker_ids: list = field(default_factory=list)
    marker_corners: tuple = ()
    charuco_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    charuco_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float64))
    selected_branch: str = ""
    branch_summary: dict = field(default_factory=dict)
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    rmse_px: float | None = None
    hull_fraction: float = 0.0
    quality_pass: bool = False
    reason: str = "no_face_marker"


class RobustCharucoDetector:
    """B6 + P1 の分岐検出器。boards は {名前: CharucoBoard}。"""

    def __init__(self, boards: dict, dictionary, camera_matrix, dist_coeffs,
                 width: int, height: int, decode=DECODE_P1):
        self.boards = dict(boards)
        self.dictionary = dictionary
        self.decode = dict(decode)
        self.width, self.height = int(width), int(height)
        self.camera_matrix = np.asarray(camera_matrix, np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, np.float64).reshape(-1)
        self.branches = []
        for name, scale, resize_colour, window in BRANCHES:
            params = production_detector_parameters()
            if window is not None:
                (params.adaptiveThreshWinSizeMin,
                 params.adaptiveThreshWinSizeMax,
                 params.adaptiveThreshWinSizeStep) = window
            params.perspectiveRemoveIgnoredMarginPerCell = float(
                self.decode["perspectiveRemoveIgnoredMarginPerCell"]
            )
            params.perspectiveRemovePixelPerCell = int(
                self.decode["perspectiveRemovePixelPerCell"]
            )
            detectors = {}
            for board_name, board in self.boards.items():
                detector = cv2.aruco.CharucoDetector(board)
                detector.setDetectorParameters(params)
                charuco = detector.getCharucoParameters()
                charuco.cameraMatrix = scaled_camera_matrix(self.camera_matrix, scale)
                charuco.distCoeffs = self.dist_coeffs
                detector.setCharucoParameters(charuco)
                detectors[board_name] = detector
            self.branches.append({
                "name": name,
                "scale": scale,
                "resize_colour": resize_colour,
                "marker_detector": cv2.aruco.ArucoDetector(dictionary, params),
                "board_detectors": detectors,
            })

    def effective_decode(self) -> list:
        """全分岐のmarker/board detectorが保持する復号設定を読み戻す。"""
        values = []
        for branch in self.branches:
            held = [branch["marker_detector"].getDetectorParameters()]
            held += [d.getDetectorParameters() for d in branch["board_detectors"].values()]
            for item in held:
                values.append((
                    branch["name"],
                    float(item.perspectiveRemoveIgnoredMarginPerCell),
                    int(item.perspectiveRemovePixelPerCell),
                ))
        return values

    @staticmethod
    def _branch_image(image, branch):
        scale = branch["scale"]
        if image.ndim == 2:
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if branch["resize_colour"]:
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _detect_branch(gray, branch, board_name, board_ids, markers):
        if markers is None:
            markers = branch["marker_detector"].detectMarkers(gray)
        corners, ids, _ = markers
        flat = [] if ids is None else [int(v) for v in np.asarray(ids).reshape(-1)]
        allowed = {int(v) for v in np.asarray(board_ids).reshape(-1)}
        selected = [i for i, v in enumerate(flat) if v in allowed]
        output = {
            "marker_ids": [flat[i] for i in selected],
            "marker_corners": (),
            "ids": np.zeros((0,), np.int32),
            "points": np.zeros((0, 2), np.float64),
            "reason": "no_face_marker",
        }
        if not selected:
            return output, markers
        scale = branch["scale"]
        face_corners = tuple(corners[i] for i in selected)
        face_ids = np.asarray([flat[i] for i in selected], np.int32).reshape(-1, 1)
        output["marker_corners"] = tuple(
            points_to_original(c, scale).reshape(1, 4, 2).astype(np.float32)
            for c in face_corners
        )
        try:
            charuco_corners, charuco_ids, _, _ = branch["board_detectors"][board_name].detectBoard(
                gray, markerCorners=face_corners, markerIds=face_ids
            )
        except cv2.error as error:
            output["reason"] = f"detect_board_error:{error.code}"
            return output, markers
        if charuco_ids is None or charuco_corners is None:
            output["reason"] = "no_charuco_corners"
            return output, markers
        output["ids"] = np.asarray(charuco_ids, np.int32).reshape(-1)
        output["points"] = points_to_original(np.asarray(charuco_corners).reshape(-1, 2), scale)
        enough = len(output["ids"]) >= POSE_MIN_POINTS
        output["reason"] = "ok" if enough else "partial_under_pose_gate"
        return output, markers

    def _preselect(self, detection, board):
        """分岐選択用: CameraInfoモデルでIPPE（LMなし）と品質ゲート。"""
        ids, points = detection["ids"], detection["points"]
        if len(ids) < POSE_MIN_POINTS:
            return {"ok": False, "quality_pass": False, "charuco_n": int(len(ids))}
        objects = np.asarray(board.getChessboardCorners(), np.float64)[ids]
        solved = ippe_best(objects, points, self.camera_matrix, self.dist_coeffs)
        if solved is None:
            return {"ok": False, "quality_pass": False, "charuco_n": int(len(ids))}
        hull = hull_fraction(points, self.width, self.height)
        quality = bool(
            len(ids) >= QUALITY["min_points"]
            and hull >= QUALITY["min_hull_fraction"]
            and solved[2] <= QUALITY["max_rmse_px"]
        )
        return {"ok": True, "quality_pass": quality, "charuco_n": int(len(ids)),
                "rmse_px": solved[2]}

    def detect_faces(self, image, board_names, pose_camera_matrix=None, pose_dist_coeffs=None,
                     visible_ids=None):
        """指定boardをB6で検出し、IPPE→LMと品質ゲートを適用する。

        マーカー検出は分岐ごとに1回だけ行い、面ごとに使い回す。
        visible_ids を渡すと、そのIDにも1.5倍分岐のマーカーにも現れない
        boardは処理しない（写っていない面のために2倍検出を走らせない）。
        """
        if image.shape[1] != self.width or image.shape[0] != self.height:
            raise ValueError(
                f"image {image.shape[1]}x{image.shape[0]} does not match "
                f"camera {self.width}x{self.height}"
            )
        if pose_camera_matrix is None:
            pose_k, pose_d = self.camera_matrix, self.dist_coeffs
        else:
            pose_k = np.asarray(pose_camera_matrix, np.float64).reshape(3, 3)
            pose_d = np.asarray(pose_dist_coeffs, np.float64).reshape(-1)
        cache = {}
        results = {}
        if visible_ids is not None:
            first = self.branches[0]
            gray = self._branch_image(image, first)
            markers = first["marker_detector"].detectMarkers(gray)
            cache[first["name"]] = {"gray": gray, "markers": markers}
            seen = {int(v) for v in np.asarray(visible_ids).reshape(-1)}
            if markers[1] is not None:
                seen |= {int(v) for v in np.asarray(markers[1]).reshape(-1)}
            board_names = [
                name for name in board_names
                if seen & {int(v) for v in np.asarray(self.boards[name].getIds()).reshape(-1)}
            ]
        for board_name in board_names:
            board = self.boards[board_name]
            detections, summary, picked = {}, {}, None
            for branch in self.branches:
                key = branch["name"]
                if key not in cache:
                    cache[key] = {"gray": self._branch_image(image, branch), "markers": None}
                detection, markers = self._detect_branch(
                    cache[key]["gray"], branch, board_name, board.getIds(), cache[key]["markers"]
                )
                cache[key]["markers"] = markers
                detections[key] = detection
                summary[key] = self._preselect(detection, board)
                if summary[key]["quality_pass"]:
                    picked = key
                    break
            if picked is None:
                picked = next(
                    (name for name, _, _, _ in BRANCHES if summary[name]["ok"]),
                    BRANCHES[0][0],
                )
            chosen = detections[picked]
            result = FaceResult(
                marker_ids=chosen["marker_ids"],
                marker_corners=chosen["marker_corners"],
                charuco_ids=chosen["ids"],
                charuco_points=chosen["points"],
                selected_branch=picked,
                branch_summary=summary,
                reason=chosen["reason"],
            )
            result.hull_fraction = hull_fraction(result.charuco_points, self.width, self.height)
            if len(result.charuco_ids) >= POSE_MIN_POINTS:
                objects = np.asarray(board.getChessboardCorners(), np.float64)[result.charuco_ids]
                pose = ippe_lm(objects, result.charuco_points, pose_k, pose_d)
                if pose is None:
                    result.reason = "pnp_failed"
                else:
                    result.rvec, result.tvec, result.rmse_px = pose
                    result.quality_pass = bool(
                        len(result.charuco_ids) >= QUALITY["min_points"]
                        and result.hull_fraction >= QUALITY["min_hull_fraction"]
                        and result.rmse_px <= QUALITY["max_rmse_px"]
                    )
                    result.reason = "ok" if result.quality_pass else "quality_rejected"
            results[board_name] = result
        return results
