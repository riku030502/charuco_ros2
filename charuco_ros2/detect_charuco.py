#!/usr/bin/env python3
import cv2
import json
import numpy as np
import os

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformBroadcaster,
    TransformListener,
)

from scipy.spatial.transform import Rotation as R


class CharucoDetectorNode(Node):
    def __init__(self):
        super().__init__("charuco_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/hand_camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/hand_camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/charuco/debug_image")

        self.declare_parameter("parent_frame", "hand_camera_color_optical_frame")
        self.declare_parameter("child_frame", "charuco_board")
        self.declare_parameter("auto_assign_child_frame", True)
        self.declare_parameter("publish_camera_link_tf", True)
        self.declare_parameter("publish_world_camera_tf", True)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("world_lookup_timeout", 0.02)
        self.declare_parameter(
            "world_tf_cache_file",
            "charuco_ros2/config/world_camera_tfs.json"
        )
        self.declare_parameter("saved_world_tf_publish_rate", 10.0)
        self.declare_parameter("camera_link_offset_x", 0.052)  # [m]
        self.declare_parameter("camera_link_offset_y", 0.067)  # [m]
        self.declare_parameter("camera_link_offset_z", 0.0)    # [m]
        self.declare_parameter("camera_link_rotation_z_deg", 90.0)
        self.declare_parameter("camera_link_rotation_x_deg", 90.0)
        self.declare_parameter("camera_link_rotation_y_deg", 180.0)

        self.declare_parameter("squares_x", 7)
        self.declare_parameter("squares_y", 5)
        self.declare_parameter("square_length", 0.010)  # [m]
        self.declare_parameter("marker_length", 0.007)  # [m]

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        self.auto_assign_child_frame = (
            self.get_parameter("auto_assign_child_frame").value
        )
        self.publish_camera_link_tf = (
            self.get_parameter("publish_camera_link_tf").value
        )
        self.publish_world_camera_tf = (
            self.get_parameter("publish_world_camera_tf").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.world_lookup_timeout = float(
            self.get_parameter("world_lookup_timeout").value
        )
        self.world_tf_cache_file = os.path.expanduser(
            self.get_parameter("world_tf_cache_file").value
        )
        self.saved_world_tf_publish_rate = float(
            self.get_parameter("saved_world_tf_publish_rate").value
        )
        self.camera_link_offset = (
            self.get_parameter("camera_link_offset_x").value,
            self.get_parameter("camera_link_offset_y").value,
            self.get_parameter("camera_link_offset_z").value,
        )
        self.camera_link_quat = R.from_euler(
            "zxy",
            [
                self.get_parameter("camera_link_rotation_z_deg").value,
                self.get_parameter("camera_link_rotation_x_deg").value,
                self.get_parameter("camera_link_rotation_y_deg").value,
            ],
            degrees=True
        ).as_quat()
        self.squares_x = self.get_parameter("squares_x").value
        self.squares_y = self.get_parameter("squares_y").value
        self.square_length = self.get_parameter("square_length").value
        self.marker_length = self.get_parameter("marker_length").value

        self.board_id_configs = [
            {
                "name": "left_camera_charuco",
                "min_id": 0,
                "max_id": 16,
                "child_frame": "left_camera_charuco",
                "camera_link_frame": "left_camera_link",
            },
            {
                "name": "right_camera_charuco",
                "min_id": 30,
                "max_id": 46,
                "child_frame": "right_camera_charuco",
                "camera_link_frame": "right_camera_link",
            },
        ]

        # =========================
        # ChArUco board
        # =========================
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params
        )

        self.board = self.create_charuco_board()
        self.charuco_detector = self.create_charuco_detector(self.board)

        for config in self.board_id_configs:
            marker_ids = np.arange(
                config["min_id"],
                config["max_id"] + 1,
                dtype=np.int32
            )
            config["board"] = self.create_charuco_board(marker_ids)
            config["detector"] = self.create_charuco_detector(config["board"])

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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.saved_world_transforms = {}
        self.load_saved_world_transforms()

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

        self.saved_tf_timer = None
        if self.saved_world_tf_publish_rate > 0.0:
            self.saved_tf_timer = self.create_timer(
                1.0 / self.saved_world_tf_publish_rate,
                self.publish_saved_world_transforms
            )

        self.get_logger().info("ChArUco detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"board: {self.squares_x}x{self.squares_y}")
        self.get_logger().info(f"square_length: {self.square_length} m")
        self.get_logger().info(f"marker_length: {self.marker_length} m")
        self.get_logger().info(
            f"auto_assign_child_frame: {self.auto_assign_child_frame}"
        )
        self.get_logger().info(
            f"publish_world_camera_tf: {self.publish_world_camera_tf}, "
            f"world_frame: {self.world_frame}"
        )
        self.get_logger().info(
            f"world_tf_cache_file: {self.world_tf_cache_file}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        self.set_camera_parameters_for_detector(self.charuco_detector)
        for config in self.board_id_configs:
            self.set_camera_parameters_for_detector(config["detector"])

        self.camera_info_received = True

        self.get_logger().info("CameraInfo received")
        self.get_logger().info(f"camera_matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"dist_coeffs: {self.dist_coeffs}")

    def image_callback(self, msg: Image):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for CameraInfo...", throttle_duration_sec=2.0)
            self.publish_saved_world_transforms(
                msg.header.stamp,
                "waiting for CameraInfo"
            )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            self.publish_saved_world_transforms(
                msg.header.stamp,
                "cv_bridge conversion failed"
            )
            return

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================
        # Detect ArUco markers and interpolate ChArUco corners
        # =========================
        try:
            if self.auto_assign_child_frame:
                marker_corners, marker_ids, _ = (
                    self.aruco_detector.detectMarkers(gray)
                )

                if marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(
                        debug_frame,
                        marker_corners,
                        marker_ids
                    )

                board_config = self.resolve_board_config_from_marker_ids(
                    marker_ids
                )
                if board_config is None:
                    self.handle_detection_failure(
                        msg,
                        debug_frame,
                        "detected markers did not match any configured marker ID range"
                    )
                    return

                selected_board = board_config["board"]
                selected_detector = board_config["detector"]
                child_frame_id = board_config["child_frame"]
                camera_link_frame_id = board_config["camera_link_frame"]
            else:
                selected_board = self.board
                selected_detector = self.charuco_detector
                child_frame_id = self.child_frame
                camera_link_frame_id = None

            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                selected_detector.detectBoard(gray)
            )
        except cv2.error as e:
            self.handle_detection_failure(
                msg,
                debug_frame,
                f"ChArUco detection skipped: {e}"
            )
            return

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_frame, marker_corners, marker_ids)

        if charuco_ids is None or len(charuco_ids) < 6:
            self.handle_detection_failure(
                msg,
                debug_frame,
                "not enough ChArUco corners"
            )
            return

        cv2.aruco.drawDetectedCornersCharuco(
            debug_frame,
            charuco_corners,
            charuco_ids
        )

        self.get_logger().info(
            f"Detected ChArUco assigned to {child_frame_id}",
            throttle_duration_sec=1.0
        )

        # =========================
        # Estimate pose
        # =========================
        object_points = selected_board.getChessboardCorners()[
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
            self.handle_detection_failure(
                msg,
                debug_frame,
                f"ChArUco pose estimation skipped: {e}"
            )
            return

        if not success:
            self.handle_detection_failure(
                msg,
                debug_frame,
                "solvePnP failed"
            )
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

        self.publish_tf(
            msg,
            rvec,
            tvec,
            child_frame_id,
            camera_link_frame_id,
        )

        self.get_logger().info(
            f"Detected ChArUco: tvec = "
            f"[{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]",
            throttle_duration_sec=1.0
        )

        self.publish_debug_image(debug_frame, msg.header)

    def create_charuco_board(self, marker_ids=None):
        if marker_ids is None:
            return cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.aruco_dict
            )

        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict,
            marker_ids
        )

    def create_charuco_detector(self, board):
        detector = cv2.aruco.CharucoDetector(board)
        detector.setDetectorParameters(self.detector_params)
        return detector

    def set_camera_parameters_for_detector(self, detector):
        charuco_params = detector.getCharucoParameters()
        charuco_params.cameraMatrix = self.camera_matrix
        charuco_params.distCoeffs = self.dist_coeffs
        detector.setCharucoParameters(charuco_params)

    def resolve_board_config_from_marker_ids(self, marker_ids):
        if marker_ids is None or len(marker_ids) == 0:
            return None

        detected_ids = marker_ids.flatten()
        for config in self.board_id_configs:
            min_id = config["min_id"]
            max_id = config["max_id"]
            if np.any((detected_ids >= min_id) & (detected_ids <= max_id)):
                return config

        return None

    def resolve_child_frame_from_marker_ids(self, marker_ids):
        config = self.resolve_board_config_from_marker_ids(marker_ids)
        if config is None:
            return None

        return config["child_frame"]

    def publish_tf(self, image_msg: Image, rvec, tvec, child_frame_id,
                   camera_link_frame_id=None):
        transform = TransformStamped()

        transform.header.stamp = image_msg.header.stamp

        # 基本はCameraInfo/Imageのframe_idを使う方が安全
        if image_msg.header.frame_id:
            transform.header.frame_id = image_msg.header.frame_id
        else:
            transform.header.frame_id = self.parent_frame

        transform.child_frame_id = child_frame_id

        transform.transform.translation.x = float(tvec[0][0])
        transform.transform.translation.y = float(tvec[1][0])
        transform.transform.translation.z = float(tvec[2][0])

        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()  # x, y, z, w

        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

        transforms = [transform]

        if (
            self.publish_camera_link_tf
            and camera_link_frame_id
            and not self.publish_world_camera_tf
        ):
            camera_link_transform = TransformStamped()
            camera_link_transform.header.stamp = image_msg.header.stamp
            camera_link_transform.header.frame_id = child_frame_id
            camera_link_transform.child_frame_id = camera_link_frame_id

            camera_link_transform.transform.translation.x = float(
                self.camera_link_offset[0]
            )
            camera_link_transform.transform.translation.y = float(
                self.camera_link_offset[1]
            )
            camera_link_transform.transform.translation.z = float(
                self.camera_link_offset[2]
            )
            camera_link_transform.transform.rotation.x = float(
                self.camera_link_quat[0]
            )
            camera_link_transform.transform.rotation.y = float(
                self.camera_link_quat[1]
            )
            camera_link_transform.transform.rotation.z = float(
                self.camera_link_quat[2]
            )
            camera_link_transform.transform.rotation.w = float(
                self.camera_link_quat[3]
            )
            transforms.append(camera_link_transform)

        if self.publish_world_camera_tf and camera_link_frame_id:
            world_camera_transform = self.create_world_camera_transform(
                image_msg,
                transform,
                camera_link_frame_id,
            )
            if world_camera_transform is not None:
                transforms.append(world_camera_transform)
                self.save_world_transform(world_camera_transform)

        self.tf_broadcaster.sendTransform(transforms)

    def create_world_camera_transform(self, image_msg: Image,
                                      parent_to_charuco: TransformStamped,
                                      camera_link_frame_id):
        try:
            world_to_parent = self.tf_buffer.lookup_transform(
                self.world_frame,
                parent_to_charuco.header.frame_id,
                Time(),
                timeout=Duration(seconds=self.world_lookup_timeout),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(
                f"World TF lookup skipped: {self.world_frame} -> "
                f"{parent_to_charuco.header.frame_id}: {e}",
                throttle_duration_sec=2.0
            )
            return None

        world_t_parent, world_r_parent = self.transform_to_pose(world_to_parent)
        parent_t_charuco, parent_r_charuco = self.transform_to_pose(
            parent_to_charuco
        )
        world_t_charuco = (
            world_t_parent + world_r_parent.apply(parent_t_charuco)
        )
        world_r_charuco = world_r_parent * parent_r_charuco
        charuco_t_camera = np.array(self.camera_link_offset, dtype=np.float64)
        charuco_r_camera = R.from_quat(self.camera_link_quat)
        world_t_camera = (
            world_t_charuco + world_r_charuco.apply(charuco_t_camera)
        )
        world_r_camera = world_r_charuco * charuco_r_camera

        transform = TransformStamped()
        transform.header.stamp = image_msg.header.stamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = camera_link_frame_id
        transform.transform.translation.x = float(world_t_camera[0])
        transform.transform.translation.y = float(world_t_camera[1])
        transform.transform.translation.z = float(world_t_camera[2])

        quat = world_r_camera.as_quat()
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        return transform

    def transform_to_pose(self, transform: TransformStamped):
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

    def handle_detection_failure(self, image_msg: Image, debug_frame, reason):
        self.publish_saved_world_transforms(image_msg.header.stamp, reason)
        self.publish_debug_image(debug_frame, image_msg.header)

    def load_saved_world_transforms(self):
        if not os.path.exists(self.world_tf_cache_file):
            self.get_logger().info(
                f"No saved world TF cache found: {self.world_tf_cache_file}"
            )
            return

        try:
            with open(self.world_tf_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn(
                f"Failed to load saved world TF cache "
                f"{self.world_tf_cache_file}: {e}"
            )
            return

        transforms = data.get("transforms", {})
        if not isinstance(transforms, dict):
            self.get_logger().warn(
                f"Saved world TF cache has invalid format: "
                f"{self.world_tf_cache_file}"
            )
            return

        loaded_count = 0
        for child_frame_id, value in transforms.items():
            if not self.is_camera_link_frame(child_frame_id):
                continue

            try:
                transform = self.transform_from_cache(child_frame_id, value)
            except (KeyError, TypeError, ValueError) as e:
                self.get_logger().warn(
                    f"Skipping invalid saved TF for {child_frame_id}: {e}"
                )
                continue

            self.saved_world_transforms[child_frame_id] = transform
            loaded_count += 1

        self.get_logger().info(
            f"Loaded {loaded_count} saved world camera link TF(s) from "
            f"{self.world_tf_cache_file}"
        )

    def is_camera_link_frame(self, frame_id):
        return any(
            frame_id == config["camera_link_frame"]
            for config in self.board_id_configs
        )

    def save_world_transform(self, transform: TransformStamped):
        self.save_world_transforms([transform])

    def save_world_transforms(self, transforms):
        if not transforms:
            return

        for transform in transforms:
            self.saved_world_transforms[transform.child_frame_id] = transform

        data = {
            "world_frame": self.world_frame,
            "transforms": {
                child_frame_id: self.transform_to_cache(saved_transform)
                for child_frame_id, saved_transform
                in self.saved_world_transforms.items()
            }
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
                f"Failed to save world TF cache "
                f"{self.world_tf_cache_file}: {e}",
                throttle_duration_sec=2.0
            )
            return

        self.get_logger().info(
            f"Saved {len(transforms)} world TF(s) to "
            f"{self.world_tf_cache_file}",
            throttle_duration_sec=2.0
        )

    def publish_saved_world_transforms(self, stamp=None, reason=None):
        if not self.publish_world_camera_tf:
            return False

        if not self.saved_world_transforms:
            if reason:
                self.get_logger().warn(
                    f"ChArUco detection failed ({reason}); no saved world TF "
                    "is available",
                    throttle_duration_sec=2.0
                )
            return False

        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        transforms = []
        for saved_transform in self.saved_world_transforms.values():
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = saved_transform.header.frame_id
            transform.child_frame_id = saved_transform.child_frame_id
            transform.transform = saved_transform.transform
            transforms.append(transform)

        self.tf_broadcaster.sendTransform(transforms)

        if reason:
            child_frames = ", ".join(sorted(self.saved_world_transforms.keys()))
            self.get_logger().warn(
                f"ChArUco detection failed ({reason}); publishing saved "
                f"world TF(s): {child_frames}",
                throttle_duration_sec=2.0
            )
        return True

    def transform_to_cache(self, transform: TransformStamped):
        return {
            "parent_frame": transform.header.frame_id,
            "translation": {
                "x": transform.transform.translation.x,
                "y": transform.transform.translation.y,
                "z": transform.transform.translation.z,
            },
            "rotation": {
                "x": transform.transform.rotation.x,
                "y": transform.transform.rotation.y,
                "z": transform.transform.rotation.z,
                "w": transform.transform.rotation.w,
            },
        }

    def transform_from_cache(self, child_frame_id, value):
        transform = TransformStamped()
        transform.header.frame_id = value.get("parent_frame", self.world_frame)
        transform.child_frame_id = child_frame_id

        translation = value["translation"]
        transform.transform.translation.x = float(translation["x"])
        transform.transform.translation.y = float(translation["y"])
        transform.transform.translation.z = float(translation["z"])

        rotation = value["rotation"]
        transform.transform.rotation.x = float(rotation["x"])
        transform.transform.rotation.y = float(rotation["y"])
        transform.transform.rotation.z = float(rotation["z"])
        transform.transform.rotation.w = float(rotation["w"])
        return transform

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
