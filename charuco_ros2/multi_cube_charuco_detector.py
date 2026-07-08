#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation as R


class MultiCharucoDetectorNode(Node):
    def __init__(self):
        super().__init__("multi_cube_charuco_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/charuco/debug_image")

        self.declare_parameter("parent_frame", "camera_color_optical_frame")

        # =========================
        # ChArUco board parameters
        # =========================
        # 生成コードと必ず合わせる
        self.declare_parameter("squares_x", 4)
        self.declare_parameter("squares_y", 4)
        self.declare_parameter("square_length", 0.010)  # [m]
        self.declare_parameter("marker_length", 0.007)  # [m]
        self.declare_parameter("cube_size", 0.050)  # [m] キューブ一辺の長さ

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.parent_frame = self.get_parameter("parent_frame").value

        self.squares_x = self.get_parameter("squares_x").value
        self.squares_y = self.get_parameter("squares_y").value
        self.square_length = self.get_parameter("square_length").value
        self.marker_length = self.get_parameter("marker_length").value
        self.cube_size = self.get_parameter("cube_size").value

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

        # 各面ボード座標系からfront面ボード座標系への変換行列を計算して追加
        for config in self.board_id_configs:
            config["T_front_in_face"] = self._compute_T_front_in_face(
                config["face_name"], self.cube_size
            )

        # =========================
        # ArUco dictionary
        # =========================
        # 生成コード側が ID 0-79 を使うため DICT_4X4_100 にする
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_100
        )

        # =========================
        # Detector parameters
        # =========================
        self.detector_params = cv2.aruco.DetectorParameters()

        if hasattr(self.detector_params, "useAruco3Detection"):
            self.detector_params.useAruco3Detection = True
            self.get_logger().info("Aruco3 detection: enabled")
        else:
            self.get_logger().warn(
                "Aruco3 detection parameter is not available in this OpenCV version"
            )

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

        # =========================
        # ROS
        # =========================
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )

        self.get_logger().info("Multi ChArUco detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"debug_image_topic: {self.debug_image_topic}")
        self.get_logger().info(f"parent_frame: {self.parent_frame}")
        self.get_logger().info(f"board: {self.squares_x}x{self.squares_y}")
        self.get_logger().info(f"square_length: {self.square_length} m")
        self.get_logger().info(f"marker_length: {self.marker_length} m")

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

        self.get_logger().info("CameraInfo received")
        self.get_logger().info(f"camera_matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"dist_coeffs: {self.dist_coeffs}")

    def image_callback(self, msg: Image):
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
        else:
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
        board = entry["board"]
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

        # solvePnPには最低4点は欲しい
        if charuco_ids is None or len(charuco_ids) < 4:
            return {
                "name": config["name"],
                "cube_name": config["cube_name"],
                "face_name": config["face_name"],
                "child_frame": config["child_frame"],
                "num_markers": num_markers,
                "num_corners": 0 if charuco_ids is None else len(charuco_ids),
                "pose_success": False,
            }

        cv2.aruco.drawDetectedCornersCharuco(
            debug_frame,
            charuco_corners,
            charuco_ids,
            (0, 255, 255),
        )

        # =========================
        # Estimate pose
        # =========================
        object_points = board.getChessboardCorners()[
            charuco_ids.flatten()
        ].astype(np.float32)

        image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as e:
            self.get_logger().debug(
                f"{config['name']} solvePnP skipped: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        if not success:
            return {
                "name": config["name"],
                "cube_name": config["cube_name"],
                "face_name": config["face_name"],
                "child_frame": config["child_frame"],
                "num_markers": num_markers,
                "num_corners": len(charuco_ids),
                "pose_success": False,
            }

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

        self.publish_cube_tf(
            image_msg=image_msg,
            config=config,
            rvec=rvec,
            tvec=tvec,
        )

        self.draw_board_label(
            debug_frame=debug_frame,
            config=config,
            charuco_corners=charuco_corners,
            tvec=tvec,
        )

        return {
            "name": config["name"],
            "cube_name": config["cube_name"],
            "face_name": config["face_name"],
            "child_frame": config["child_frame"],
            "num_markers": num_markers,
            "num_corners": len(charuco_ids),
            "pose_success": True,
            "tvec": tvec,
        }

    def draw_board_label(self, debug_frame, config: dict, charuco_corners, tvec):
        """検出したboard名を画像上に描画する。"""

        points = charuco_corners.reshape(-1, 2)

        x = int(np.mean(points[:, 0]))
        y = int(np.mean(points[:, 1]))

        text = (
            f"{config['cube_name']} {config['face_name']} "
            f"z={float(tvec[2][0]):.3f}m"
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
        transform = TransformStamped()

        transform.header.stamp = image_msg.header.stamp

        # 基本はImageのframe_idを使う方が安全
        if image_msg.header.frame_id:
            transform.header.frame_id = image_msg.header.frame_id
        else:
            transform.header.frame_id = self.parent_frame

        transform.child_frame_id = child_frame

        transform.transform.translation.x = float(tvec[0][0])
        transform.transform.translation.y = float(tvec[1][0])
        transform.transform.translation.z = float(tvec[2][0])

        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()  # x, y, z, w

        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(transform)

    @staticmethod
    def _compute_T_front_in_face(face_name: str, L: float) -> np.ndarray:
        """各面のボード座標系からfront面ボード座標系への4x4変換行列を返す。

        キューブ座標系 (前面左上隅が原点):
          +X: 右, +Y: 下, +Z: 奥（front面から離れる方向）

        各面ボードの配置前提:
          - 全面でボードの-Y軸（上方向）がキューブ上方向（-Y_cube）を向く
            ただしtop面はボードの-Y軸がfront方向（-Z_cube）を向く
          - ボード原点は各面をキューブ外から見た時の「左上コーナー」
          - ボードZ軸はsolvePnP規約に従いカメラ方向（面の外側）を向く

        面ごとの原点（キューブ座標）:
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
        t_front_in_face = R_face_to_cube.T @ (p_front - p_face)

        T = np.eye(4)
        T[:3, :3] = R_front_in_face
        T[:3,  3] = t_front_in_face
        return T

    def publish_cube_tf(self, image_msg: Image, config: dict, rvec, tvec):
        """検出した面のボード姿勢からfront面相当のTFを計算して発信する。"""
        R_face, _ = cv2.Rodrigues(rvec)
        T_face_in_cam = np.eye(4)
        T_face_in_cam[:3, :3] = R_face
        T_face_in_cam[:3,  3] = tvec.flatten()

        T_front_in_cam = T_face_in_cam @ config["T_front_in_face"]

        R_front = T_front_in_cam[:3, :3]
        t_front = T_front_in_cam[:3, 3].reshape(3, 1)
        rvec_front, _ = cv2.Rodrigues(R_front)

        self.publish_tf(
            image_msg=image_msg,
            child_frame=config["cube_frame"],
            rvec=rvec_front,
            tvec=t_front,
        )

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

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()