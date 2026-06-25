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


class MultiArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("multi_aruco_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/aruco/debug_image")

        self.declare_parameter("parent_frame", "camera_color_optical_frame")

        # =========================
        # ArUco marker parameters
        # =========================
        # 生成コード側の marker_size_mm = 45 に合わせる
        self.declare_parameter("marker_length", 0.045)  # [m]

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.parent_frame = self.get_parameter("parent_frame").value
        self.marker_length = float(self.get_parameter("marker_length").value)

        # =========================
        # ArUco ID configs
        # =========================
        # 生成コード側と対応
        #
        # left_cube
        #   front : ID 0
        #   left  : ID 1
        #   right : ID 2
        #   back  : ID 3
        #   top   : ID 4
        #
        # right_cube
        #   front : ID 5
        #   left  : ID 6
        #   right : ID 7
        #   back  : ID 8
        #   top   : ID 9
        self.aruco_id_configs = [
            {
                "name": "left_cube_front_aruco",
                "cube_name": "left_cube",
                "face_name": "front",
                "marker_id": 0,
                "child_frame": "left_cube_front_aruco",
            },
            {
                "name": "left_cube_left_aruco",
                "cube_name": "left_cube",
                "face_name": "left",
                "marker_id": 1,
                "child_frame": "left_cube_left_aruco",
            },
            {
                "name": "left_cube_right_aruco",
                "cube_name": "left_cube",
                "face_name": "right",
                "marker_id": 2,
                "child_frame": "left_cube_right_aruco",
            },
            {
                "name": "left_cube_back_aruco",
                "cube_name": "left_cube",
                "face_name": "back",
                "marker_id": 3,
                "child_frame": "left_cube_back_aruco",
            },
            {
                "name": "left_cube_top_aruco",
                "cube_name": "left_cube",
                "face_name": "top",
                "marker_id": 4,
                "child_frame": "left_cube_top_aruco",
            },
            {
                "name": "right_cube_front_aruco",
                "cube_name": "right_cube",
                "face_name": "front",
                "marker_id": 5,
                "child_frame": "right_cube_front_aruco",
            },
            {
                "name": "right_cube_left_aruco",
                "cube_name": "right_cube",
                "face_name": "left",
                "marker_id": 6,
                "child_frame": "right_cube_left_aruco",
            },
            {
                "name": "right_cube_right_aruco",
                "cube_name": "right_cube",
                "face_name": "right",
                "marker_id": 7,
                "child_frame": "right_cube_right_aruco",
            },
            {
                "name": "right_cube_back_aruco",
                "cube_name": "right_cube",
                "face_name": "back",
                "marker_id": 8,
                "child_frame": "right_cube_back_aruco",
            },
            {
                "name": "right_cube_top_aruco",
                "cube_name": "right_cube",
                "face_name": "top",
                "marker_id": 9,
                "child_frame": "right_cube_top_aruco",
            },
        ]

        self.id_to_config = {
            config["marker_id"]: config
            for config in self.aruco_id_configs
        }

        self.validate_aruco_configs()

        # =========================
        # ArUco dictionary
        # =========================
        # 生成コード側が DICT_4X4_50 / ID 0-9 を使う想定
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # =========================
        # ArUco detector parameters
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

        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params,
        )

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

        self.get_logger().info("Multi ArUco detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"debug_image_topic: {self.debug_image_topic}")
        self.get_logger().info(f"parent_frame: {self.parent_frame}")
        self.get_logger().info(f"marker_length: {self.marker_length} m")

        for config in self.aruco_id_configs:
            self.get_logger().info(
                f"registered marker: {config['name']} "
                f"cube={config['cube_name']} "
                f"face={config['face_name']} "
                f"id={config['marker_id']} "
                f"child_frame={config['child_frame']}"
            )

    def validate_aruco_configs(self):
        """ID重複と辞書範囲を確認する。"""

        used_ids = [
            config["marker_id"]
            for config in self.aruco_id_configs
        ]

        duplicated_ids = sorted(
            {marker_id for marker_id in used_ids if used_ids.count(marker_id) > 1}
        )

        if duplicated_ids:
            raise ValueError(
                f"Duplicated marker IDs found: {duplicated_ids}"
            )

        max_dictionary_id = 49  # DICT_4X4_50 は 0〜49

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
        # Detect ArUco markers
        # =========================
        try:
            marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        except cv2.error as e:
            self.get_logger().debug(
                f"ArUco detection skipped: {e}",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        total_detected_markers = 0 if marker_ids is None else len(marker_ids)

        if marker_ids is not None and total_detected_markers > 0:
            cv2.aruco.drawDetectedMarkers(
                debug_frame,
                marker_corners,
                marker_ids,
            )

        detected_results = []

        if marker_ids is not None and total_detected_markers > 0:
            for marker_corner, marker_id_array in zip(marker_corners, marker_ids):
                marker_id = int(marker_id_array[0])

                if marker_id not in self.id_to_config:
                    continue

                config = self.id_to_config[marker_id]

                result = self.estimate_and_publish_one_marker(
                    config=config,
                    marker_corner=marker_corner,
                    image_msg=msg,
                    debug_frame=debug_frame,
                )

                if result is not None:
                    detected_results.append(result)

        self.publish_detection_status(
            debug_frame=debug_frame,
            total_detected_markers=total_detected_markers,
            detected_results=detected_results,
        )

        self.publish_debug_image(debug_frame, msg.header)

        ok_markers = [
            result
            for result in detected_results
            if result.get("pose_success", False)
        ]

        if ok_markers:
            marker_text = ", ".join(
                [
                    f"{result['cube_name']}:{result['face_name']}"
                    for result in ok_markers
                ]
            )

            self.get_logger().info(
                f"Detected ArUco markers: {marker_text}",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().info(
                f"No valid ArUco marker pose. markers={total_detected_markers}",
                throttle_duration_sec=2.0,
            )

    def estimate_and_publish_one_marker(
        self,
        config: dict,
        marker_corner,
        image_msg: Image,
        debug_frame,
    ):
        """1つのArUco markerについて姿勢推定してTFを出す。"""

        success, rvec, tvec = self.estimate_marker_pose(marker_corner)

        if not success:
            return {
                "name": config["name"],
                "cube_name": config["cube_name"],
                "face_name": config["face_name"],
                "child_frame": config["child_frame"],
                "marker_id": config["marker_id"],
                "pose_success": False,
            }

        cv2.drawFrameAxes(
            debug_frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            min(0.03, self.marker_length * 0.5),
        )

        self.publish_tf(
            image_msg=image_msg,
            child_frame=config["child_frame"],
            rvec=rvec,
            tvec=tvec,
        )

        self.draw_marker_label(
            debug_frame=debug_frame,
            config=config,
            marker_corner=marker_corner,
            tvec=tvec,
        )

        return {
            "name": config["name"],
            "cube_name": config["cube_name"],
            "face_name": config["face_name"],
            "child_frame": config["child_frame"],
            "marker_id": config["marker_id"],
            "pose_success": True,
            "tvec": tvec,
        }

    def estimate_marker_pose(self, marker_corner):
        """
        ArUco markerの4隅から姿勢推定する。

        OpenCVのmarker corner順:
            top-left, top-right, bottom-right, bottom-left

        child_frameはmarker中心に置く。
        """

        half = self.marker_length / 2.0

        object_points = np.array(
            [
                [-half,  half, 0.0],  # top-left
                [ half,  half, 0.0],  # top-right
                [ half, -half, 0.0],  # bottom-right
                [-half, -half, 0.0],  # bottom-left
            ],
            dtype=np.float32,
        )

        image_points = marker_corner.reshape(4, 2).astype(np.float32)

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
                f"ArUco pose estimation skipped: {e}",
                throttle_duration_sec=2.0,
            )
            return False, None, None

        return success, rvec, tvec

    def draw_marker_label(self, debug_frame, config: dict, marker_corner, tvec):
        """検出したmarker名を画像上に描画する。"""

        points = marker_corner.reshape(4, 2)

        x = int(np.mean(points[:, 0]))
        y = int(np.mean(points[:, 1]))

        text = (
            f"{config['cube_name']} {config['face_name']} "
            f"ID:{config['marker_id']} "
            f"z={float(tvec[2][0]):.3f}m"
        )

        cv2.putText(
            debug_frame,
            text,
            (max(10, x - 160), max(30, y - 10)),
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
        detected_results: list,
    ):
        ok_markers = [
            result
            for result in detected_results
            if result.get("pose_success", False)
        ]

        text1 = (
            f"markers: {total_detected_markers}, "
            f"pose OK: {len(ok_markers)}"
        )

        cv2.putText(
            debug_frame,
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if ok_markers else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if ok_markers:
            text2 = " / ".join(
                [
                    f"{result['cube_name']}:{result['face_name']}"
                    for result in ok_markers[:3]
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

    def publish_debug_image(self, frame, header):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiArucoDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()