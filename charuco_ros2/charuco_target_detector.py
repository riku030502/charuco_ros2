#!/usr/bin/env python3
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from charuco_ros2.charuco_board_utils import (
    CUBE_CHARUCO_DEFAULTS,
    create_charuco_board,
    get_aruco_dictionary,
    load_board_config,
    validate_board_config,
)


class CharucoTargetDetectorNode(Node):
    def __init__(self):
        super().__init__("charuco_target_detector")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )
        self.declare_parameter("board_config_path", "")
        self.declare_parameter("output_frame_id", "target_pose")
        self.declare_parameter("parent_frame", "")
        self.declare_parameter("pose_topic", "~/pose")
        self.declare_parameter("debug_image_topic", "~/debug_image")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_pose", True)
        self.declare_parameter("allow_marker_only_pose", True)
        self.declare_parameter("min_markers_for_pose", 2)
        self.declare_parameter("min_charuco_corners", 4)
        self.declare_parameter("max_detection_rate", 10.0)
        self.declare_parameter("axis_length", 0.03)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.board_config_path = self.get_parameter("board_config_path").value
        self.output_frame_id = self.get_parameter("output_frame_id").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.publish_tf_enabled = bool(self.get_parameter("publish_tf").value)
        self.publish_pose_enabled = bool(
            self.get_parameter("publish_pose").value
        )
        self.allow_marker_only_pose = bool(
            self.get_parameter("allow_marker_only_pose").value
        )
        self.min_markers_for_pose = int(
            self.get_parameter("min_markers_for_pose").value
        )
        self.min_charuco_corners = int(
            self.get_parameter("min_charuco_corners").value
        )
        self.max_detection_rate = float(
            self.get_parameter("max_detection_rate").value
        )
        self.axis_length = float(self.get_parameter("axis_length").value)

        self.board_config = self.load_config()
        self.aruco_dict = get_aruco_dictionary(self.board_config["dictionary"])
        self.board = create_charuco_board(self.board_config)

        self.detector_params = cv2.aruco.DetectorParameters()
        if hasattr(self.detector_params, "useAruco3Detection"):
            self.detector_params.useAruco3Detection = False
        self.detector_params.minSideLengthCanonicalImg = 16
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 4
        self.detector_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.charuco_detector.setDetectorParameters(self.detector_params)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_detection_time = None
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
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )

        ids = self.board_config["marker_ids"]
        self.get_logger().info("ChArUco target detector started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(
            f"board_config_path: {self.board_config_path or '<defaults>'}"
        )
        self.get_logger().info(f"output_frame_id: {self.output_frame_id}")
        self.get_logger().info(
            f"board: {self.board_config['squares_x']}x"
            f"{self.board_config['squares_y']}, "
            f"{self.board_config['dictionary']}, IDs {min(ids)}-{max(ids)}"
        )

    def load_config(self):
        if self.board_config_path:
            path = os.path.expanduser(self.board_config_path)
            return load_board_config(path)

        config = dict(CUBE_CHARUCO_DEFAULTS)
        config.update(
            {
                "name": "validation_charuco",
                "marker_id_start": 80,
            }
        )
        return validate_board_config(config)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        charuco_params = self.charuco_detector.getCharucoParameters()
        charuco_params.cameraMatrix = self.camera_matrix
        charuco_params.distCoeffs = self.dist_coeffs
        self.charuco_detector.setCharucoParameters(charuco_params)

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn(
                "Waiting for CameraInfo...",
                throttle_duration_sec=2.0,
            )
            return

        if not self.should_detect_now():
            return

        self.last_detection_time = self.get_clock().now()
        self.detect(msg)

    def should_detect_now(self) -> bool:
        if self.max_detection_rate <= 0.0 or self.last_detection_time is None:
            return True

        elapsed = self.get_clock().now() - self.last_detection_time
        return elapsed.nanoseconds >= int(1e9 / self.max_detection_rate)

    def detect(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self.charuco_detector.detectBoard(gray)
            )
        except cv2.error as e:
            self.get_logger().warn(
                f"ChArUco detection skipped: {e}",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        self.draw_detected_features(
            debug_frame,
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        )

        pose = self.estimate_pose(
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        )
        if pose is None:
            marker_count = 0 if marker_ids is None else len(marker_ids)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            self.get_logger().warn(
                f"No valid target pose. markers={marker_count}, "
                f"charuco_corners={corner_count}",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        rvec, tvec, pose_source = pose
        cv2.drawFrameAxes(
            debug_frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            self.axis_length,
        )

        parent_frame = msg.header.frame_id or self.parent_frame
        if not parent_frame:
            self.get_logger().warn(
                "Image header frame_id is empty and parent_frame is unset",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        pose_msg = self.create_pose_msg(
            msg.header.stamp,
            parent_frame,
            rvec,
            tvec,
        )
        if self.publish_pose_enabled:
            self.pose_pub.publish(pose_msg)
        if self.publish_tf_enabled:
            self.tf_broadcaster.sendTransform(
                self.create_transform_msg(pose_msg, self.output_frame_id)
            )

        self.get_logger().info(
            f"Detected {self.output_frame_id}: "
            f"z={float(tvec[2][0]):.3f} m [{pose_source}]",
            throttle_duration_sec=1.0,
        )
        self.publish_debug_image(debug_frame, msg.header)

    def estimate_pose(
        self,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ):
        if (
            charuco_ids is not None
            and len(charuco_ids) >= self.min_charuco_corners
        ):
            object_points = self.board.getChessboardCorners()[
                charuco_ids.flatten()
            ].astype(np.float32)
            image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
            pose = self.solve_pnp(object_points, image_points)
            if pose is not None:
                return pose[0], pose[1], "chessboard"

        if not self.allow_marker_only_pose:
            return None

        object_points, image_points = self.match_marker_points(
            marker_corners,
            marker_ids,
        )
        if object_points is None:
            return None

        pose = self.solve_pnp(object_points, image_points)
        if pose is None:
            return None

        return pose[0], pose[1], "marker"

    def match_marker_points(self, marker_corners, marker_ids):
        if marker_corners is None or marker_ids is None:
            return None, None

        board_ids = self.board.getIds().flatten()
        board_object_points = self.board.getObjPoints()
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
                np.asarray(
                    board_object_points[index],
                    dtype=np.float32,
                ).reshape(4, 3)
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

    def solve_pnp(self, object_points, image_points):
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error as e:
            self.get_logger().warn(
                f"solvePnP skipped: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        if not success:
            return None

        return rvec, tvec

    def create_pose_msg(self, stamp, frame_id: str, rvec, tvec) -> PoseStamped:
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = float(tvec[0][0])
        pose_msg.pose.position.y = float(tvec[1][0])
        pose_msg.pose.position.z = float(tvec[2][0])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        return pose_msg

    def create_transform_msg(
        self,
        pose_msg: PoseStamped,
        child_frame_id: str,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header = pose_msg.header
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        transform.transform.rotation = pose_msg.pose.orientation
        return transform

    def draw_detected_features(
        self,
        frame,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ):
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                frame,
                charuco_corners,
                charuco_ids,
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
    node = CharucoTargetDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
