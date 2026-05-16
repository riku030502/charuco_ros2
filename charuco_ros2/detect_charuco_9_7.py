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


class CharucoDetectorNode(Node):
    def __init__(self):
        super().__init__("charuco_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/charuco/debug_image")

        self.declare_parameter("parent_frame", "camera_color_optical_frame")
        self.declare_parameter("child_frame", "charuco_board")

        self.declare_parameter("squares_x", 9)
        self.declare_parameter("squares_y", 7)
        self.declare_parameter("square_length", 0.010)  # [m]
        self.declare_parameter("marker_length", 0.007)  # [m]

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value

        self.squares_x = self.get_parameter("squares_x").value
        self.squares_y = self.get_parameter("squares_y").value
        self.square_length = self.get_parameter("square_length").value
        self.marker_length = self.get_parameter("marker_length").value

        # =========================
        # ChArUco board
        # =========================
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.charuco_detector.setDetectorParameters(self.detector_params)

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
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10
        )

        self.get_logger().info("ChArUco detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"board: {self.squares_x}x{self.squares_y}")
        self.get_logger().info(f"square_length: {self.square_length} m")
        self.get_logger().info(f"marker_length: {self.marker_length} m")

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        charuco_params = self.charuco_detector.getCharucoParameters()
        charuco_params.cameraMatrix = self.camera_matrix
        charuco_params.distCoeffs = self.dist_coeffs
        self.charuco_detector.setCharucoParameters(charuco_params)

        self.camera_info_received = True

        self.get_logger().info("CameraInfo received")
        self.get_logger().info(f"camera_matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"dist_coeffs: {self.dist_coeffs}")

    def image_callback(self, msg: Image):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for CameraInfo...", throttle_duration_sec=2.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================
        # Detect ArUco markers and interpolate ChArUco corners
        # =========================
        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self.charuco_detector.detectBoard(gray)
            )
        except cv2.error as e:
            self.get_logger().debug(
                f"ChArUco detection skipped: {e}",
                throttle_duration_sec=2.0
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_frame, marker_corners, marker_ids)

        if charuco_ids is None or len(charuco_ids) < 2:
            self.publish_debug_image(debug_frame, msg.header)
            return

        cv2.aruco.drawDetectedCornersCharuco(
            debug_frame,
            charuco_corners,
            charuco_ids
        )

        # =========================
        # Estimate pose
        # =========================
        object_points = self.board.getChessboardCorners()[
            charuco_ids.flatten()
        ].astype(np.float32)
        image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
        except cv2.error as e:
            self.get_logger().debug(
                f"ChArUco pose estimation skipped: {e}",
                throttle_duration_sec=2.0
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        if not success:
            self.publish_debug_image(debug_frame, msg.header)
            return

        # 座標軸を描画
        cv2.drawFrameAxes(
            debug_frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            0.03
        )

        self.publish_tf(msg, rvec, tvec)

        self.get_logger().info(
            f"Detected ChArUco: tvec = "
            f"[{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]",
            throttle_duration_sec=1.0
        )

        self.publish_debug_image(debug_frame, msg.header)

    def publish_tf(self, image_msg: Image, rvec, tvec):
        transform = TransformStamped()

        transform.header.stamp = image_msg.header.stamp

        # 基本はCameraInfo/Imageのframe_idを使う方が安全
        if image_msg.header.frame_id:
            transform.header.frame_id = image_msg.header.frame_id
        else:
            transform.header.frame_id = self.parent_frame

        transform.child_frame_id = self.child_frame

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
    node = CharucoDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
