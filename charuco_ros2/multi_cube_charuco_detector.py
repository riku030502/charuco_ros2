#!/usr/bin/env python3
import cv2
import json
import numpy as np
import os
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformListener,
)

from scipy.spatial.transform import Rotation as R

from charuco_ros2.charuco_board_utils import (
    CUBE_CHARUCO_DEFAULTS,
    get_aruco_dictionary,
)


class MultiCharucoDetectorNode(Node):
    def __init__(self):
        super().__init__("multi_cube_charuco_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/hand_camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/hand_camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/charuco/debug_image")
        self.declare_parameter(
            "detect_service_name",
            "/multi_cube_charuco_detector/detect_once",
        )
        self.declare_parameter("detection_window_sec", 2.0)
        self.declare_parameter("detect_continuously", False)

        self.declare_parameter("parent_frame", "hand_camera_color_optical_frame")

        # =========================
        # ChArUco board parameters
        # =========================
        # 生成コードと必ず合わせる
        self.declare_parameter("squares_x", CUBE_CHARUCO_DEFAULTS["squares_x"])
        self.declare_parameter("squares_y", CUBE_CHARUCO_DEFAULTS["squares_y"])
        self.declare_parameter(
            "square_length",
            CUBE_CHARUCO_DEFAULTS["square_length_m"],
        )
        self.declare_parameter(
            "marker_length",
            CUBE_CHARUCO_DEFAULTS["marker_length_m"],
        )
        self.declare_parameter("cube_size", 0.050)  # [m] キューブ一辺の長さ

        # =========================
        # Pose estimation
        # =========================
        # ChArUcoの内部チェス盤コーナーは4x4ボードで9点しかなく、1点補間する
        # だけでも周囲のマーカーが解像されている必要がある。小さいマーカーを
        # 遠く/斜めから見るとマーカーは読めてもコーナーが揃わず姿勢を捨てる
        # ことになるため、その場合はマーカーの角そのものでPnPを解く。
        # マーカー1個につき4点得られる。
        self.declare_parameter("allow_marker_only_pose", True)
        self.declare_parameter("min_markers_for_pose", 2)

        # =========================
        # World camera link TF
        # =========================
        # detect_charuco が検出・保存した world 基準のカメラリンクTFを、
        # 起動時に読んで静的TFとして配信する。この辺が無いと world から
        # left/right_camera_color_optical_frame へ辿れず、左右カメラの
        # 点群を world 座標に変換できない。
        #
        #   world ──> link_base ──> ...          （robot_state_publisher）
        #   world ──> left_camera_link ──> ...   （ここで配信）
        self.declare_parameter("publish_world_camera_tf", True)
        self.declare_parameter(
            "world_tf_cache_file",
            "charuco_ros2/config/world_camera_tfs.json",
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("world_lookup_timeout", 0.05)
        self.declare_parameter("publish_camera_link_from_cube_tf", True)
        self.declare_parameter("save_camera_link_from_cube_tf", True)
        self.declare_parameter("left_cube_camera_link_frame", "left_camera_link")
        self.declare_parameter("right_cube_camera_link_frame", "right_camera_link")

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.detect_service_name = self.get_parameter("detect_service_name").value
        self.detection_window_sec = float(
            self.get_parameter("detection_window_sec").value
        )
        self.detect_continuously = bool(
            self.get_parameter("detect_continuously").value
        )

        self.parent_frame = self.get_parameter("parent_frame").value

        self.squares_x = self.get_parameter("squares_x").value
        self.squares_y = self.get_parameter("squares_y").value
        self.square_length = self.get_parameter("square_length").value
        self.marker_length = self.get_parameter("marker_length").value
        self.cube_size = self.get_parameter("cube_size").value
        self.allow_marker_only_pose = bool(
            self.get_parameter("allow_marker_only_pose").value
        )
        self.min_markers_for_pose = int(
            self.get_parameter("min_markers_for_pose").value
        )

        self.publish_world_camera_tf = bool(
            self.get_parameter("publish_world_camera_tf").value
        )
        self.world_tf_cache_file = self.resolve_cache_file(
            self.get_parameter("world_tf_cache_file").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.world_lookup_timeout = float(
            self.get_parameter("world_lookup_timeout").value
        )
        self.publish_camera_link_from_cube_tf = bool(
            self.get_parameter("publish_camera_link_from_cube_tf").value
        )
        self.save_camera_link_from_cube_tf = bool(
            self.get_parameter("save_camera_link_from_cube_tf").value
        )
        self.cube_camera_link_frames = {
            "left_cube": self.get_parameter("left_cube_camera_link_frame").value,
            "right_cube": self.get_parameter("right_cube_camera_link_frame").value,
        }

        # =========================
        # Board ID configs
        # =========================
        # 生成コード側と対応
        #
        # left_cube
        #   front : 0-7
        #   left  : 8-15
        #   right : 16-23
        #   back  : 24-31
        #   top   : 32-39
        #
        # right_cube
        #   front : 40-47
        #   left  : 48-55
        #   right : 56-63
        #   back  : 64-71
        #   top   : 72-79
        self.board_id_configs = [
            {
                "name": "left_cube_front_charuco",
                "cube_name": "left_cube",
                "face_name": "front",
                "min_id": 0,
                "max_id": 7,
                "child_frame": "left_cube_front_charuco",
                "cube_frame": "left_cube",
            },
            {
                "name": "left_cube_left_charuco",
                "cube_name": "left_cube",
                "face_name": "left",
                "min_id": 8,
                "max_id": 15,
                "child_frame": "left_cube_left_charuco",
                "cube_frame": "left_cube",
            },
            {
                "name": "left_cube_right_charuco",
                "cube_name": "left_cube",
                "face_name": "right",
                "min_id": 16,
                "max_id": 23,
                "child_frame": "left_cube_right_charuco",
                "cube_frame": "left_cube",
            },
            {
                "name": "left_cube_back_charuco",
                "cube_name": "left_cube",
                "face_name": "back",
                "min_id": 24,
                "max_id": 31,
                "child_frame": "left_cube_back_charuco",
                "cube_frame": "left_cube",
            },
            {
                "name": "left_cube_top_charuco",
                "cube_name": "left_cube",
                "face_name": "top",
                "min_id": 32,
                "max_id": 39,
                "child_frame": "left_cube_top_charuco",
                "cube_frame": "left_cube",
            },
            {
                "name": "right_cube_front_charuco",
                "cube_name": "right_cube",
                "face_name": "front",
                "min_id": 40,
                "max_id": 47,
                "child_frame": "right_cube_front_charuco",
                "cube_frame": "right_cube",
            },
            {
                "name": "right_cube_left_charuco",
                "cube_name": "right_cube",
                "face_name": "left",
                "min_id": 48,
                "max_id": 55,
                "child_frame": "right_cube_left_charuco",
                "cube_frame": "right_cube",
            },
            {
                "name": "right_cube_right_charuco",
                "cube_name": "right_cube",
                "face_name": "right",
                "min_id": 56,
                "max_id": 63,
                "child_frame": "right_cube_right_charuco",
                "cube_frame": "right_cube",
            },
            {
                "name": "right_cube_back_charuco",
                "cube_name": "right_cube",
                "face_name": "back",
                "min_id": 64,
                "max_id": 71,
                "child_frame": "right_cube_back_charuco",
                "cube_frame": "right_cube",
            },
            {
                "name": "right_cube_top_charuco",
                "cube_name": "right_cube",
                "face_name": "top",
                "min_id": 72,
                "max_id": 79,
                "child_frame": "right_cube_top_charuco",
                "cube_frame": "right_cube",
            },
        ]

        T_camera_link_in_front = self._compute_T_camera_link_in_front()

        # 各面ボード座標系からfront/camera_link相当への変換行列を計算して追加
        for config in self.board_id_configs:
            config["T_front_in_face"] = self._compute_T_front_in_face(
                config["face_name"], self.cube_size
            )
            config["T_camera_link_in_face"] = (
                config["T_front_in_face"] @ T_camera_link_in_front
            )

        # =========================
        # ArUco dictionary
        # =========================
        # 生成コード側が ID 0-79 を使うため DICT_4X4_100 にする
        self.aruco_dict = get_aruco_dictionary(
            CUBE_CHARUCO_DEFAULTS["dictionary"]
        )

        # =========================
        # Detector parameters
        # =========================
        self.detector_params = cv2.aruco.DetectorParameters()

        # Aruco3は縮小画像で検出するため、marker 1辺が
        # minSideLengthCanonicalImg (default 32px) 未満だと足切りされる。
        # 10mmマスのキューブ面は近接時でも1辺10px程度しかないので無効化する。
        if hasattr(self.detector_params, "useAruco3Detection"):
            self.detector_params.useAruco3Detection = False
            self.get_logger().info("Aruco3 detection: disabled (small markers)")

        self.detector_params.minSideLengthCanonicalImg = 16
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 4
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # debug用に全marker数を見るためのArucoDetector
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params,
        )

        # =========================
        # ChArUco boards / detectors
        # =========================
        self.board_entries = []

        for config in self.board_id_configs:
            marker_ids = np.arange(
                config["min_id"],
                config["max_id"] + 1,
                dtype=np.int32,
            )

            board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.aruco_dict,
                marker_ids,
            )

            charuco_detector = cv2.aruco.CharucoDetector(board)
            charuco_detector.setDetectorParameters(self.detector_params)

            self.board_entries.append(
                {
                    "config": config,
                    "marker_ids": marker_ids,
                    "board": board,
                    "charuco_detector": charuco_detector,
                }
            )

        self.validate_board_configs()

        # =========================
        # Camera parameters
        # =========================
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.camera_info_event = threading.Event()
        self.detection_lock = threading.Lock()
        self.detection_active_until = 0.0
        self.detection_success_event = threading.Event()
        self.latest_detection_message = ""

        # =========================
        # ROS
        # =========================
        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.saved_world_camera_transforms = {}

        if self.publish_world_camera_tf:
            self.publish_world_camera_transforms()

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
            callback_group=self.callback_group,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
            callback_group=self.callback_group,
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )
        self.detect_service = self.create_service(
            Trigger,
            self.detect_service_name,
            self.handle_detect_once,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Multi ChArUco detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"debug_image_topic: {self.debug_image_topic}")
        self.get_logger().info(
            f"detect_service_name: {self.detect_service_name}, "
            f"detection_window_sec={self.detection_window_sec:.1f}, "
            f"detect_continuously={self.detect_continuously}"
        )
        self.get_logger().info(f"parent_frame: {self.parent_frame}")
        self.get_logger().info(f"board: {self.squares_x}x{self.squares_y}")
        self.get_logger().info(f"square_length: {self.square_length} m")
        self.get_logger().info(f"marker_length: {self.marker_length} m")
        self.get_logger().info(
            f"publish_camera_link_from_cube_tf: "
            f"{self.publish_camera_link_from_cube_tf}, "
            f"cube_camera_link_frames: {self.cube_camera_link_frames}"
        )
        self.get_logger().info(
            f"publish_world_camera_tf: {self.publish_world_camera_tf}, "
            f"world_tf_cache_file: {self.world_tf_cache_file}"
        )

        for entry in self.board_entries:
            config = entry["config"]
            self.get_logger().info(
                f"registered board: {config['name']} "
                f"cube={config['cube_name']} "
                f"face={config['face_name']} "
                f"id={config['min_id']}-{config['max_id']} "
                f"child_frame={config['child_frame']}"
            )

    def validate_board_configs(self):
        """ID数、重複、辞書範囲を確認する。"""

        reference_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict,
        )

        expected_marker_count = len(reference_board.getIds())

        used_ids = []

        for entry in self.board_entries:
            config = entry["config"]
            marker_count = config["max_id"] - config["min_id"] + 1

            if marker_count != expected_marker_count:
                raise ValueError(
                    f"{config['name']} needs {expected_marker_count} marker IDs, "
                    f"but range {config['min_id']}-{config['max_id']} has "
                    f"{marker_count}"
                )

            used_ids.extend(range(config["min_id"], config["max_id"] + 1))

        duplicated_ids = sorted(
            {marker_id for marker_id in used_ids if used_ids.count(marker_id) > 1}
        )

        if duplicated_ids:
            raise ValueError(
                f"Duplicated marker IDs found: {duplicated_ids}"
            )

        max_dictionary_id = 99  # DICT_4X4_100 は 0〜99

        if max(used_ids) > max_dictionary_id:
            raise ValueError(
                f"Marker ID {max(used_ids)} exceeds dictionary limit "
                f"0-{max_dictionary_id}."
            )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        # 各ChArUcoDetectorにカメラ内部パラメータを設定する
        for entry in self.board_entries:
            charuco_detector = entry["charuco_detector"]

            charuco_params = charuco_detector.getCharucoParameters()
            charuco_params.cameraMatrix = self.camera_matrix
            charuco_params.distCoeffs = self.dist_coeffs
            charuco_detector.setCharucoParameters(charuco_params)

        self.camera_info_received = True
        self.camera_info_event.set()

        self.get_logger().info("CameraInfo received")
        self.get_logger().info(f"camera_matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"dist_coeffs: {self.dist_coeffs}")

    def handle_detect_once(self, request, response):
        del request

        if self.detection_window_sec <= 0.0:
            response.success = False
            response.message = "detection_window_sec must be positive"
            return response

        if not self.camera_info_received:
            self.get_logger().info(
                "Waiting for CameraInfo before starting cube detection"
            )
            self.camera_info_event.wait(timeout=min(self.detection_window_sec, 2.0))

        if not self.camera_info_received:
            response.success = False
            response.message = "timed out waiting for CameraInfo"
            self.get_logger().warn(response.message)
            return response

        with self.detection_lock:
            self.latest_detection_message = ""
            self.detection_success_event.clear()
            self.detection_active_until = (
                time.monotonic() + self.detection_window_sec
            )

        self.get_logger().info(
            f"Cube ChArUco detection enabled for "
            f"{self.detection_window_sec:.1f}s"
        )

        detected = self.detection_success_event.wait(
            timeout=self.detection_window_sec
        )

        with self.detection_lock:
            message = self.latest_detection_message
            self.detection_active_until = 0.0

        if detected:
            response.success = True
            response.message = message or "cube ChArUco detection succeeded"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = (
                message
                or f"no cube ChArUco pose detected in "
                f"{self.detection_window_sec:.1f}s"
            )
            self.get_logger().warn(response.message)
        return response

    def detection_is_active(self):
        if self.detect_continuously:
            return True
        with self.detection_lock:
            return time.monotonic() <= self.detection_active_until

    def notify_detection_success(self, message):
        with self.detection_lock:
            self.latest_detection_message = message
        self.detection_success_event.set()

    def image_callback(self, msg: Image):
        if not self.detection_is_active():
            return

        if not self.camera_info_received:
            self.get_logger().warn(
                "Waiting for CameraInfo...",
                throttle_duration_sec=2.0,
            )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================
        # Debug用: 全ArUco marker数を先に見る
        # =========================
        try:
            all_marker_corners, all_marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        except cv2.error as e:
            self.get_logger().debug(
                f"ArUco detection skipped: {e}",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        total_detected_markers = 0 if all_marker_ids is None else len(all_marker_ids)

        if all_marker_ids is not None and total_detected_markers > 0:
            cv2.aruco.drawDetectedMarkers(
                debug_frame,
                all_marker_corners,
                all_marker_ids,
            )

        detected_boards = []

        # =========================
        # 各ChArUco boardを個別に検出
        # =========================
        for entry in self.board_entries:
            result = self.try_detect_one_board(
                entry=entry,
                gray=gray,
                image_msg=msg,
                debug_frame=debug_frame,
            )

            if result is not None:
                detected_boards.append(result)

        self.publish_detection_status(
            debug_frame=debug_frame,
            total_detected_markers=total_detected_markers,
            detected_boards=detected_boards,
        )

        self.publish_debug_image(debug_frame, msg.header)

        ok_boards = [
            result
            for result in detected_boards
            if result.get("pose_success", False)
        ]

        camera_link_transforms = self.publish_cube_transforms(msg, ok_boards)

        if ok_boards:
            board_text = ", ".join(
                [
                    f"{result['cube_name']}:{result['face_name']}"
                    for result in ok_boards
                ]
            )

            self.get_logger().info(
                f"Detected ChArUco boards: {board_text}",
                throttle_duration_sec=1.0,
            )
            if camera_link_transforms:
                child_frames = ", ".join(
                    transform.child_frame_id
                    for transform in camera_link_transforms
                )
                self.notify_detection_success(
                    f"cube ChArUco detection succeeded; updated TF(s): "
                    f"{child_frames}; boards={board_text}"
                )
            else:
                with self.detection_lock:
                    self.latest_detection_message = (
                        f"detected board pose(s), but no camera_link TF was "
                        f"updated; boards={board_text}"
                    )
        else:
            with self.detection_lock:
                self.latest_detection_message = (
                    f"no valid ChArUco board pose; "
                    f"markers={total_detected_markers}"
                )
            self.get_logger().info(
                f"No valid ChArUco board pose. markers={total_detected_markers}",
                throttle_duration_sec=2.0,
            )

    def try_detect_one_board(
        self,
        entry: dict,
        gray,
        image_msg: Image,
        debug_frame,
    ):
        """1つのboard configについて検出と姿勢推定を試す。"""

        config = entry["config"]
        charuco_detector = entry["charuco_detector"]

        # =========================
        # Detect ChArUco board
        # =========================
        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                charuco_detector.detectBoard(gray)
            )
        except cv2.error as e:
            self.get_logger().debug(
                f"{config['name']} detectBoard skipped: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        num_markers = 0 if marker_ids is None else len(marker_ids)

        # そのboardのmarkerが1個も見えていないなら無視
        if marker_ids is None or num_markers == 0:
            return None

        cv2.aruco.drawDetectedMarkers(
            debug_frame,
            marker_corners,
            marker_ids,
        )

        num_corners = 0 if charuco_ids is None else len(charuco_ids)

        if num_corners >= 4:
            cv2.aruco.drawDetectedCornersCharuco(
                debug_frame,
                charuco_corners,
                charuco_ids,
                (0, 255, 255),
            )

        # =========================
        # Estimate pose
        # =========================
        pose = self.estimate_board_pose(
            entry=entry,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
        )

        if pose is None:
            return {
                "name": config["name"],
                "cube_name": config["cube_name"],
                "face_name": config["face_name"],
                "child_frame": config["child_frame"],
                "num_markers": num_markers,
                "num_corners": num_corners,
                "pose_success": False,
            }

        rvec, tvec, pose_source = pose

        # 座標軸を描画
        cv2.drawFrameAxes(
            debug_frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            0.03,
        )

        self.publish_tf(
            image_msg=image_msg,
            child_frame=config["child_frame"],
            rvec=rvec,
            tvec=tvec,
        )

        self.draw_board_label(
            debug_frame=debug_frame,
            config=config,
            charuco_corners=charuco_corners,
            marker_corners=marker_corners,
            tvec=tvec,
            pose_source=pose_source,
        )

        return {
            "name": config["name"],
            "cube_name": config["cube_name"],
            "face_name": config["face_name"],
            "child_frame": config["child_frame"],
            "config": config,
            "num_markers": num_markers,
            "num_corners": num_corners,
            "pose_success": True,
            "pose_source": pose_source,
            "rvec": rvec,
            "tvec": tvec,
        }

    def estimate_board_pose(
        self,
        entry: dict,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ):
        """Estimate the board pose, falling back to the marker corners.

        戻り値は (rvec, tvec, pose_source) か None。

        chessboard: ChArUcoの内部コーナーを使う。精度が高いので優先する。
        marker:     マーカーの角を使う。コーナーが揃わない小さな/斜めの見え方
                    でも、マーカーが読めていれば姿勢を出せる。
        """
        board = entry["board"]
        config = entry["config"]

        if charuco_ids is not None and len(charuco_ids) >= 4:
            object_points = board.getChessboardCorners()[
                charuco_ids.flatten()
            ].astype(np.float32)
            image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

            pose = self.solve_pnp_planar(object_points, image_points, config)
            if pose is not None:
                return pose[0], pose[1], "chessboard"

        if not self.allow_marker_only_pose:
            return None

        object_points, image_points = self.match_marker_points(
            entry,
            marker_corners,
            marker_ids,
        )
        if object_points is None:
            return None

        pose = self.solve_pnp_planar(object_points, image_points, config)
        if pose is None:
            return None

        return pose[0], pose[1], "marker"

    def match_marker_points(self, entry: dict, marker_corners, marker_ids):
        """Pair detected marker corners with their 3D positions on the board."""
        board = entry["board"]

        board_ids = board.getIds().flatten()
        board_object_points = board.getObjPoints()
        id_to_index = {
            int(marker_id): index
            for index, marker_id in enumerate(board_ids)
        }

        object_points = []
        image_points = []

        for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
            index = id_to_index.get(int(marker_id))
            if index is None:
                continue

            object_points.append(
                np.asarray(board_object_points[index], dtype=np.float32).reshape(4, 3)
            )
            image_points.append(
                np.asarray(corners, dtype=np.float32).reshape(4, 2)
            )

        if len(object_points) < self.min_markers_for_pose:
            return None, None

        return (
            np.concatenate(object_points, axis=0),
            np.concatenate(image_points, axis=0),
        )

    def solve_pnp_planar(self, object_points, image_points, config: dict):
        """Solve PnP for coplanar points.

        ボード面上の点は同一平面なので、平面専用のIPPEを使う。
        ITERATIVEは平面かつ点数が少ないと不安定になりやすい。
        """
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error as e:
            self.get_logger().debug(
                f"{config['name']} solvePnP skipped: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        if not success:
            return None

        return rvec, tvec

    def draw_board_label(
        self,
        debug_frame,
        config: dict,
        charuco_corners,
        marker_corners,
        tvec,
        pose_source: str,
    ):
        """検出したboard名を画像上に描画する。"""

        # マーカーのみで解いた場合、ChArUcoコーナーは無いこともある。
        if charuco_corners is not None and len(charuco_corners) > 0:
            points = np.asarray(charuco_corners).reshape(-1, 2)
        else:
            points = np.concatenate(
                [np.asarray(corners).reshape(-1, 2) for corners in marker_corners],
                axis=0,
            )

        x = int(np.mean(points[:, 0]))
        y = int(np.mean(points[:, 1]))

        text = (
            f"{config['cube_name']} {config['face_name']} "
            f"z={float(tvec[2][0]):.3f}m [{pose_source}]"
        )

        cv2.putText(
            debug_frame,
            text,
            (max(10, x - 120), max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def publish_detection_status(
        self,
        debug_frame,
        total_detected_markers: int,
        detected_boards: list,
    ):
        ok_boards = [
            result
            for result in detected_boards
            if result.get("pose_success", False)
        ]

        text1 = (
            f"markers: {total_detected_markers}, "
            f"boards pose OK: {len(ok_boards)}"
        )

        cv2.putText(
            debug_frame,
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if ok_boards else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if ok_boards:
            text2 = " / ".join(
                [
                    f"{result['cube_name']}:{result['face_name']}"
                    for result in ok_boards[:3]
                ]
            )

            cv2.putText(
                debug_frame,
                text2,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def publish_tf(self, image_msg: Image, child_frame: str, rvec, tvec):
        self.tf_broadcaster.sendTransform(
            self.camera_transform_from_rt(image_msg, child_frame, rvec, tvec)
        )

    def camera_transform_from_rt(self, image_msg: Image, child_frame: str, rvec, tvec):
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()  # x, y, z, w

        # 基本はImageのframe_idを使う方が安全
        parent_frame = image_msg.header.frame_id or self.parent_frame

        return self.create_transform(
            image_msg.header.stamp,
            parent_frame,
            child_frame,
            tvec.flatten(),
            quat,
        )

    def create_transform(self, stamp, parent_frame, child_frame, translation, quat):
        transform = TransformStamped()

        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

        return transform

    @staticmethod
    def transform_to_pose(transform: TransformStamped):
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=np.float64)
        rotation = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])
        return translation, rotation

    @staticmethod
    def _compute_T_front_in_face(face_name: str, L: float) -> np.ndarray:
        """各面のボード座標系からfront面ボード座標系への4x4変換行列を返す。

        キューブ座標系 (前面左上隅が原点):
          +X: 右, +Y: 下, +Z: 奥（front面から離れる方向）

        各面ボードの配置前提:
          - 全面でボードの-Y軸（上方向）がキューブ上方向（-Y_cube）を向く
            ただしtop面はボードの-Y軸がfront方向（-Z_cube）を向く
          - ボード原点は各面をキューブ外から見た時の「左上コーナー」
          - 面間オフセットでは +X=右, +Y=下, +Z=奥 として扱う

        面ごとの回転はキューブ形状から決める。並進は実測した面間関係を使う。
        実測したfront原点（各面を正面から見た各面ボード座標）:
          top:   下4.5 cm, 奥0.5 cm
          left:  右4.5 cm, 奥0.5 cm
          right: 奥4.5 cm, 左0.5 cm
          back:  奥5.0 cm, 右4.0 cm

        参考: キューブ形状だけで置いた場合の面ごとの原点（キューブ座標）:
          front: (0,  0,  0)
          top:   (0,  0,  0)  ← front-left-top角を共有
          right: (L,  0,  0)
          left:  (0,  0,  L)
          back:  (L,  0,  L)
        """
        # 各面のボード座標軸（キューブ座標系で表現）と原点
        # R の列 = [X_board, Y_board, Z_board] in cube coords
        face_params = {
            "front": (
                np.array([[1,  0,  0], [0,  1,  0], [0,  0,  1]], dtype=float),
                np.array([0.0, 0.0, 0.0]),
            ),
            "top": (
                np.array([[1,  0,  0], [0,  0,  1], [0, -1,  0]], dtype=float),
                np.array([0.0, 0.0, 0.0]),
            ),
            "right": (
                np.array([[0,  0, -1], [0,  1,  0], [1,  0,  0]], dtype=float),
                np.array([L,  0.0, 0.0]),
            ),
            "left": (
                np.array([[0,  0,  1], [0,  1,  0], [-1, 0,  0]], dtype=float),
                np.array([0.0, 0.0, L]),
            ),
            "back": (
                np.array([[-1, 0,  0], [0,  1,  0], [0,  0, -1]], dtype=float),
                np.array([L,  0.0, L]),
            ),
        }

        R_face_to_cube, p_face = face_params[face_name]
        R_front_to_cube = np.eye(3)
        p_front = np.zeros(3)

        # front board frame → face board frame
        R_front_in_face = R_face_to_cube.T @ R_front_to_cube
        geometric_t_front_in_face = R_face_to_cube.T @ (p_front - p_face)
        t_front_in_face = MultiCharucoDetectorNode._front_translation_in_face(
            face_name,
            geometric_t_front_in_face,
        )

        T = np.eye(4)
        T[:3, :3] = R_front_in_face
        T[:3,  3] = t_front_in_face
        return T

    @staticmethod
    def _front_translation_in_face(face_name: str, fallback: np.ndarray) -> np.ndarray:
        """実測値からfront原点の位置を各面ボード座標で返す。

        +X=右, +Y=下, +Z=奥。したがって「手前」は-Z。
        """
        measured_offsets = {
            "front": np.array([0.0, 0.0, 0.0], dtype=float),
            "top": np.array([0.0, 0.045, 0.005], dtype=float),
            "left": np.array([0.045, 0.0, 0.005], dtype=float),
            "right": np.array([-0.005, 0.0, 0.045], dtype=float),
            "back": np.array([0.040, 0.0, 0.050], dtype=float),
        }
        return measured_offsets.get(face_name, fallback)

    @staticmethod
    def _compute_T_camera_link_in_front() -> np.ndarray:
        """front面ボード座標系からcube camera_link相当への変換を返す。

        実測関係: front -> camera_link は 下7.0 cm, 右4.0 cm, 奥2.0 cm。
        回転は現在のcamera_link軸を基準に、緑軸(+Y)まわり反時計回り90度、
        続けて赤軸(+X)まわり時計回り90度、さらに赤軸(+X)まわり180度を
        合成する。
        """
        T = np.eye(4)
        angle = np.deg2rad(90.0)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        R_y_ccw = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])
        R_x_cw = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ])
        R_x_180 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        T[:3, :3] = R_y_ccw @ R_x_cw @ R_x_180
        T[:3, 3] = np.array([0.040, 0.070, 0.020], dtype=float)
        return T

    def publish_cube_transforms(self, image_msg: Image, ok_boards: list):
        """検出できた面からキューブTFを発信し、キャッシュを更新する。

        同じキューブの複数面が同時に見えることがあるので、カメラに近い
        1面だけを使ってキューブごとに1つのTFを出す。
        """
        best_by_cube = {}

        for result in ok_boards:
            cube_frame = result["config"]["cube_frame"]
            current = best_by_cube.get(cube_frame)

            if current is None or self.pose_priority(result) > self.pose_priority(current):
                best_by_cube[cube_frame] = result

        transforms = []
        camera_link_transforms = []

        for cube_frame, result in best_by_cube.items():
            config = result["config"]
            front_frame = self.front_frame_for_cube(cube_frame)
            camera_link_frame = self.cube_camera_link_frames.get(cube_frame)

            if not camera_link_frame:
                continue

            self.get_logger().info(
                f"Using {cube_frame}:{result['face_name']} for "
                f"{camera_link_frame} "
                f"(distance={self.pose_distance_m(result):.3f} m, "
                f"source={result.get('pose_source')}, "
                f"corners={result.get('num_corners', 0)}, "
                f"markers={result.get('num_markers', 0)})",
                throttle_duration_sec=1.0,
            )

            rvec_camera_link, t_camera_link = self.pose_in_camera_from_face(
                config,
                result["rvec"],
                result["tvec"],
                config["T_camera_link_in_face"],
            )

            transforms.append(
                self.transform_from_matrix(
                    image_msg.header.stamp,
                    config["child_frame"],
                    front_frame,
                    config["T_front_in_face"],
                )
            )

            camera_transform = self.camera_transform_from_rt(
                image_msg,
                camera_link_frame,
                rvec_camera_link,
                t_camera_link,
            )

            world_transform = self.to_world_frame(camera_transform)
            if world_transform is None:
                continue

            camera_link_transforms.append(world_transform)

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        if camera_link_transforms:
            if self.publish_camera_link_from_cube_tf:
                self.static_tf_broadcaster.sendTransform(camera_link_transforms)
            self.save_world_camera_transforms(camera_link_transforms)

        return camera_link_transforms

    def front_frame_for_cube(self, cube_frame: str):
        return f"{cube_frame}_front"

    def transform_from_matrix(
        self,
        stamp,
        parent_frame: str,
        child_frame: str,
        transform_matrix: np.ndarray,
    ):
        return self.create_transform(
            stamp,
            parent_frame,
            child_frame,
            transform_matrix[:3, 3],
            R.from_matrix(transform_matrix[:3, :3]).as_quat(),
        )

    @staticmethod
    def pose_distance_m(result: dict):
        return float(np.linalg.norm(result["tvec"].flatten()))

    @staticmethod
    def pose_priority(result: dict):
        """カメラに近い面を優先し、同距離なら検出品質で比べる。"""
        distance_m = MultiCharucoDetectorNode.pose_distance_m(result)
        return (
            -distance_m,
            1 if result.get("pose_source") == "chessboard" else 0,
            result.get("num_corners", 0),
            result.get("num_markers", 0),
        )

    def cube_target_pose_in_camera(self, config: dict, rvec, tvec):
        """検出した面の姿勢からcube camera_link相当の姿勢を計算する。"""
        return self.pose_in_camera_from_face(
            config,
            rvec,
            tvec,
            config["T_camera_link_in_face"],
        )

    def pose_in_camera_from_face(
        self,
        config: dict,
        rvec,
        tvec,
        target_in_face: np.ndarray,
    ):
        """検出した面の姿勢から任意の面内ターゲット姿勢を計算する。"""
        R_face, _ = cv2.Rodrigues(rvec)
        T_face_in_cam = np.eye(4)
        T_face_in_cam[:3, :3] = R_face
        T_face_in_cam[:3,  3] = tvec.flatten()

        T_target_in_cam = T_face_in_cam @ target_in_face

        R_target = T_target_in_cam[:3, :3]
        t_target = T_target_in_cam[:3, 3].reshape(3, 1)
        rvec_target, _ = cv2.Rodrigues(R_target)

        return rvec_target, t_target

    def to_world_frame(self, camera_transform: TransformStamped):
        """画像フレーム基準のcamera_link TFを world_frame 基準に直す。"""
        try:
            world_to_camera = self.tf_buffer.lookup_transform(
                self.world_frame,
                camera_transform.header.frame_id,
                Time(),
                timeout=Duration(seconds=self.world_lookup_timeout),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"Camera link TF not updated; lookup failed: {self.world_frame} -> "
                f"{camera_transform.header.frame_id}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        world_t_camera, world_r_camera = self.transform_to_pose(world_to_camera)
        camera_t_link, camera_r_link = self.transform_to_pose(camera_transform)

        world_t_link = world_t_camera + world_r_camera.apply(camera_t_link)
        world_r_link = world_r_camera * camera_r_link

        return self.create_transform(
            camera_transform.header.stamp,
            self.world_frame,
            camera_transform.child_frame_id,
            world_t_link,
            world_r_link.as_quat(),
        )

    def resolve_cache_file(self, cache_file):
        if os.path.isabs(cache_file):
            return os.path.realpath(os.path.expanduser(cache_file))

        package_relative_prefix = "charuco_ros2/"
        if cache_file.startswith(package_relative_prefix):
            cache_file = cache_file[len(package_relative_prefix):]

        return os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", cache_file)
        )

    def publish_world_camera_transforms(self):
        """保存済みのworld基準カメラリンクTFを静的TFとして配信する。

        値は detect_charuco が検出して world_tf_cache_file に保存したもの。
        multi_cube_charuco_detector は検出したcube姿勢から、この回転を更新する。
        """
        transforms = self.load_world_camera_transforms()

        if not transforms:
            self.get_logger().warn(
                f"No world camera link TF published from "
                f"{self.world_tf_cache_file}; point clouds from the left/right "
                f"cameras cannot be transformed into {self.world_frame}. "
                f"Run detect_charuco once to detect and save the camera poses."
            )
            return

        self.static_tf_broadcaster.sendTransform(transforms)

        for transform in transforms:
            self.get_logger().info(
                f"published static TF: {transform.header.frame_id} -> "
                f"{transform.child_frame_id}"
            )

    def load_world_camera_transforms(self):
        if not os.path.exists(self.world_tf_cache_file):
            self.get_logger().warn(
                f"World camera TF cache not found: {self.world_tf_cache_file}"
            )
            return []

        try:
            with open(self.world_tf_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn(
                f"Failed to load world camera TF cache "
                f"{self.world_tf_cache_file}: {e}"
            )
            return []

        cached_transforms = data.get("transforms", {})
        if not isinstance(cached_transforms, dict):
            self.get_logger().warn(
                f"World camera TF cache has invalid format: "
                f"{self.world_tf_cache_file}"
            )
            return []

        stamp = self.get_clock().now().to_msg()

        transforms = []
        for child_frame_id, value in cached_transforms.items():
            try:
                transform = self.create_transform(
                    stamp,
                    value.get("parent_frame", self.world_frame),
                    child_frame_id,
                    self.xyz_from_dict(value["translation"]),
                    self.xyzw_from_dict(value["rotation"]),
                )
                transforms.append(transform)
                self.saved_world_camera_transforms[child_frame_id] = transform
            except (KeyError, TypeError, ValueError) as e:
                self.get_logger().warn(
                    f"Skipping invalid world camera TF for {child_frame_id}: {e}"
                )

        return transforms

    def save_world_camera_transforms(self, transforms):
        if not transforms:
            return

        for transform in transforms:
            self.saved_world_camera_transforms[transform.child_frame_id] = transform

        if not self.save_camera_link_from_cube_tf:
            return

        data = {
            "world_frame": self.world_frame,
            "transforms": {
                child_frame_id: self.transform_to_cache(saved_transform)
                for child_frame_id, saved_transform
                in self.saved_world_camera_transforms.items()
            },
        }

        cache_dir = os.path.dirname(self.world_tf_cache_file)
        try:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            tmp_file = f"{self.world_tf_cache_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_file, self.world_tf_cache_file)
        except OSError as e:
            self.get_logger().warn(
                f"Failed to save world camera TF cache "
                f"{self.world_tf_cache_file}: {e}",
                throttle_duration_sec=2.0,
            )
            return

        child_frames = ", ".join(transform.child_frame_id for transform in transforms)
        self.get_logger().info(
            f"Saved camera link TF(s) from cube pose: {child_frames}",
            throttle_duration_sec=2.0,
        )

    def transform_to_cache(self, transform: TransformStamped):
        return {
            "parent_frame": transform.header.frame_id,
            "translation": self.xyz_to_dict(transform.transform.translation),
            "rotation": self.xyzw_to_dict(transform.transform.rotation),
        }

    @staticmethod
    def xyz_to_dict(value):
        return {"x": value.x, "y": value.y, "z": value.z}

    @staticmethod
    def xyzw_to_dict(value):
        return {"x": value.x, "y": value.y, "z": value.z, "w": value.w}

    @staticmethod
    def xyz_from_dict(value):
        return [float(value[key]) for key in ("x", "y", "z")]

    @staticmethod
    def xyzw_from_dict(value):
        return [float(value[key]) for key in ("x", "y", "z", "w")]

    def publish_debug_image(self, frame, header):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiCharucoDetectorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
