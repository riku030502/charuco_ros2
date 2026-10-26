"""Live pose collection for intrinsic calibration (ROS-free; the CLI node pushes frames).

One PNG per accepted pose: >= 12 corners, still for 3 analysed frames
(median corner motion <= 0.5 px), sharp (median edge width <= 4.5 px, valid
edges >= 80 %) and novel (>= 30 mm or >= 8 deg from every accepted pose).
Only the newest frame is kept in memory. A CameraInfo change during capture
stops the collection. The live gate is guidance only; the gate is re-run
offline on the saved PNGs (:mod:`pipeline`).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from charuco_ros2.robust_charuco import ippe_lm

from .board import BoardDetector
from .quality import (
    GATE,
    board_tilt_deg,
    cells,
    data_gate,
    displacement,
    normalised_radius,
    novel,
    sharpness,
    split_of,
)

STILL_PX = 0.5
STILL_FRAMES = 3


def camera_signature(mapping: dict) -> str:
    keys = ("width", "height", "k", "d", "distortion_model", "frame_id")
    return hashlib.sha256(
        json.dumps({k: mapping.get(k) for k in keys}, sort_keys=True).encode()
    ).hexdigest()


class Collector:
    def __init__(self, session: Path, detector: BoardDetector, target: int, maximum: int):
        self.session, self.detector, self.target, self.maximum = (
            Path(session),
            detector,
            target,
            maximum,
        )
        self.poses_dir = self.session / "poses"
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.view = None
        self.records: list = []
        self.accepted: list = []
        self.still_run = 0
        self.previous = None
        self.message = "RGBとCameraInfoを待っています"
        self.error = None
        self.paused = False
        self.signature = None
        self.camera = None
        manifest = self.session / "poses.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line:
                    record = json.loads(line)
                    self.records.append(record)
                    self.accepted.append((record["rvec"], record["tvec_m"]))
        if (self.session / "camera_info.json").exists():
            self.camera = json.loads(
                (self.session / "camera_info.json").read_text(encoding="utf-8")
            )
            self.signature = self.camera["signature"]
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop = True

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.message = message

    def push(self, image: np.ndarray, mapping: dict, stamp: float) -> None:
        with self.lock:
            self.sequence += 1
            self.latest = (image, mapping, stamp, self.sequence)

    def _loop(self) -> None:
        seen = 0
        while not self._stop:
            with self.lock:
                item = self.latest
            if item is None or item[3] == seen:
                time.sleep(0.03)
                continue
            image, mapping, stamp, seen = item
            try:
                self._analyse(image, mapping, stamp)
            except Exception as error:  # noqa: BLE001 - keep the UI alive and show the failure
                with self.lock:
                    self.message = f"解析エラー: {type(error).__name__}: {error}"

    def _check_camera(self, mapping: dict) -> bool:
        signature = camera_signature(mapping)
        if self.signature is None:
            self.signature = signature
            self.camera = dict(mapping, signature=signature)
            (self.session / "camera_info.json").write_text(
                json.dumps(self.camera, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif signature != self.signature:
            self.fail(
                "CameraInfo（解像度・K・D・frame_id）が撮影中に変化しました。撮影を停止します。"
            )
            return False
        return True

    def _analyse(self, image, mapping, stamp) -> None:
        if self.error or not self._check_camera(mapping):
            with self.lock:
                self.view = (image, {}, self.error)
            return
        k0 = np.asarray(self.camera["k"], float).reshape(3, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ids, points = self.detector.detect(gray)
        analysis = {"charuco_n": int(len(ids)), "ids": ids.tolist(), "points": points.tolist()}
        pose = (
            ippe_lm(self.detector.objects(ids), points, k0, np.zeros(5)) if len(ids) >= 4 else None
        )
        board = self.detector.board
        if len(ids) < GATE["corners_min"] or pose is None:
            reason = f"コーナー{GATE['corners_min']}点未満"
            self.still_run = 0
        else:
            moved = displacement(self.previous, ids, points)
            self.still_run = self.still_run + 1 if moved <= STILL_PX else 0
            quality = sharpness(gray, ids, points, board.inner_columns, board.inner_rows)
            rotation, _ = cv2.Rodrigues(pose[0])
            analysis.update(
                quality,
                motion_px=moved,
                tilt_deg=board_tilt_deg(pose[0]),
                rmse_m0_px=pose[2],
                distance_mm=float(
                    np.linalg.norm(rotation @ board.centre_m + pose[1].reshape(3)) * 1000
                ),
            )
            if self.paused:
                reason = "一時停止中"
            elif self.still_run < STILL_FRAMES:
                reason = "静止待ち"
            elif not (
                quality["width_median_px"] <= GATE["edge_width_max_px"]
                and quality["valid_fraction"] >= GATE["edge_valid_fraction_min"]
            ):
                reason = f"ボケ過大（幅 {quality['width_median_px']:.2f} px）"
            elif not novel(pose[0], pose[1], self.accepted):
                reason = "保存済みの姿勢に近い（30 mm または 8° 以上動かしてください）"
            elif len(self.records) >= self.maximum:
                reason = "上限枚数に達しました"
            else:
                self._save(image, stamp, analysis, pose)
                reason = "saved"
        self.previous = analysis if len(ids) >= GATE["corners_min"] else None
        with self.lock:
            self.view = (image, analysis, reason)
            if reason != "saved":
                self.message = reason

    def _save(self, image, stamp, analysis, pose) -> None:
        index = len(self.records)
        name = f"pose_{index:03d}.png"
        path = self.poses_dir / name
        ok, encoded = cv2.imencode(".png", image)
        if path.exists() or not ok:
            raise RuntimeError(f"could not write new file {path}")
        data = encoded.tobytes()
        path.write_bytes(data)
        k0 = np.asarray(self.camera["k"], float).reshape(3, 3)
        record = {
            "pose_index": index,
            "file": name,
            "split": split_of(index),
            "stamp_sec": stamp,
            "saved_unix": time.time(),
            "png_sha256": hashlib.sha256(data).hexdigest(),
            "camera_info_signature": self.signature,
            "charuco_n": analysis["charuco_n"],
            "rvec": pose[0].reshape(3).tolist(),
            "tvec_m": pose[1].reshape(3).tolist(),
            "rmse_m0_px": analysis["rmse_m0_px"],
            "tilt_deg": analysis["tilt_deg"],
            "distance_mm": analysis["distance_mm"],
            "width_median_px": analysis["width_median_px"],
            "valid_fraction": analysis["valid_fraction"],
            "bright_ge_250_fraction": analysis["bright_ge_250_fraction"],
            "motion_px": analysis["motion_px"],
            "radius_max": float(
                normalised_radius(np.asarray(analysis["points"]), k0[:2, 2]).max()
            ),
            "charuco_points": analysis["points"],
            "note": "live estimate; the gate is re-evaluated offline on the PNG",
        }
        with (self.session / "poses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        self.accepted.append((record["rvec"], record["tvec_m"]))
        self.still_run = 0
        with self.lock:
            group = '検証用' if split_of(index) == 'validation' else '校正用'
            self.message = f"保存しました：姿勢 {index + 1}（{group}）"

    def live_gate(self) -> dict:
        if not self.records or self.camera is None:
            return {"checks": {}, "pass": False}
        k0 = np.asarray(self.camera["k"], float).reshape(3, 3)
        return data_gate(
            self.records, int(self.camera["width"]), int(self.camera["height"]), k0[:2, 2]
        )

    def hints(self, gate: dict) -> list:
        """Plain-language guidance for the rules that are still failing."""
        checks, out = gate.get("checks", {}), []
        names = {
            "0,0": "左上",
            "1,0": "上中央",
            "2,0": "右上",
            "0,1": "左中央",
            "1,1": "中央",
            "2,1": "右中央",
            "0,2": "左下",
            "1,2": "下中央",
            "2,2": "右下",
        }
        regions = checks.get("regions_3x3_ge2_poses", {}).get("value", {})
        missing = [names[key] for key, n in regions.items() if n < GATE["region_poses_min"]]
        if missing:
            out.append("画面の " + "・".join(missing) + " にボードの角点が入る姿勢を追加")
        if checks and not checks["frontal_poses_lt15deg"]["pass"]:
            out.append("カメラに正対（傾き15°未満）した姿勢を追加")
        if checks and not checks["tilted_poses_ge30deg"]["pass"]:
            out.append("ボードを30°以上傾けた姿勢を追加")
        if checks and not (
            checks["outer_corners_calibration"]["pass"] and checks["outer_quadrants_ge5"]["pass"]
        ):
            q = checks["outer_quadrants_ge5"]["value"]
            corner = ["左上", "右上", "左下", "右下"]
            weak = [corner[i] for i in range(4) if q[i] < GATE["outer_quadrant_corners_min"]]
            out.append(
                "橙の円の外側（画像の隅・端）に角点を入れる"
                + (f"：特に {'・'.join(weak)}" if weak else "")
            )
        return out

    def status(self) -> dict:
        with self.lock:
            view, message, error = self.view, self.message, self.error
        analysis = dict(view[1]) if view else {}
        analysis.pop("points", None)
        analysis.pop("ids", None)
        gate = self.live_gate()
        return {
            "connected": view is not None,
            "message": message,
            "saved": len(self.records),
            "target": self.target,
            "maximum": self.maximum,
            "analysis": analysis,
            "gate": gate,
            "hints": self.hints(gate),
            "session": str(self.session),
            "error": error,
            "paused": self.paused,
        }

    def jpeg(self, max_width: int = 960):
        with self.lock:
            view = self.view
        if view is None or self.camera is None:
            return None
        image, analysis, reason = view
        image = image.copy()
        height, width = image.shape[:2]
        gx, gy = GATE["cell_grid"]
        k0 = np.asarray(self.camera["k"], float).reshape(3, 3)
        occupied, outer = set(), []
        for record in self.records:
            pts = np.asarray(record["charuco_points"], float).reshape(-1, 2)
            occupied |= cells(pts, width, height)
            outer.extend(pts[normalised_radius(pts, k0[:2, 2]) >= GATE["outer_radius"]])
        shade = image.copy()
        cw, ch = width / gx, height / gy
        for x, y in occupied:
            cv2.rectangle(
                shade,
                (int(x * cw), int(y * ch)),
                (int((x + 1) * cw), int((y + 1) * ch)),
                (60, 160, 60),
                -1,
            )
        image = cv2.addWeighted(shade, 0.18, image, 0.82, 0)
        for x in range(1, gx):
            cv2.line(image, (int(x * cw), 0), (int(x * cw), height), (200, 200, 200), 1)
        for y in range(1, gy):
            cv2.line(image, (0, int(y * ch)), (width, int(y * ch)), (200, 200, 200), 1)
        centre = (int(round(k0[0, 2])), int(round(k0[1, 2])))
        scale = width / 1920.0
        cv2.circle(
            image,
            centre,
            int(GATE["outer_radius"] * np.linalg.norm(k0[:2, 2])),
            (0, 200, 255),
            max(1, int(2 * scale)),
        )
        for x, y in outer:
            cv2.circle(image, (int(x), int(y)), max(2, int(4 * scale)), (0, 200, 255), -1)
        colour = (40, 200, 40) if reason == "saved" else (20, 120, 235)
        for x, y in analysis.get("points", []):
            cv2.circle(image, (int(round(x)), int(round(y))), max(3, int(6 * scale)), colour, -1)
        if width > max_width:
            image = cv2.resize(
                image, (max_width, int(height * max_width / width)), interpolation=cv2.INTER_AREA
            )
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(buffer) if ok else None
