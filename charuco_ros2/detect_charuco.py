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
from std_srvs.srv import Trigger

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
    PARAM_DEFAULTS = {
        "image_topic": "/camera/hand_camera/color/image_raw",
        "camera_info_topic": "/camera/hand_camera/color/camera_info",
        "debug_image_topic": "/charuco/debug_image",
        "detect_service_name": "/charuco/detect_once",
        "parent_frame": "hand_camera_color_optical_frame",
        "child_frame": "charuco_board",
        "auto_assign_child_frame": True,
        "publish_camera_link_tf": True,
        "publish_world_camera_tf": True,
        "world_frame": "world",
        "world_lookup_timeout": 0.02,
        "world_tf_cache_file": "charuco_ros2/config/world_camera_tfs.json",
        "saved_world_tf_publish_rate": 10.0,
        "camera_link_offset_x": 0.052,
        "camera_link_offset_y": 0.067,
        "camera_link_offset_z": 0.0,
        "camera_link_rotation_z_deg": 90.0,
        "camera_link_rotation_x_deg": 90.0,
        "camera_link_rotation_y_deg": 180.0,
        "squares_x": 7,
        "squares_y": 5,
        "square_length": 0.010,
        "marker_length": 0.007,
    }

    BOARD_ID_CONFIGS = [
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

    def __init__(self):
        super().__init__("charuco_detector_node")

        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.image_topic = self.param("image_topic")
        self.camera_info_topic = self.param("camera_info_topic")
        self.debug_image_topic = self.param("debug_image_topic")
        self.detect_service_name = self.param("detect_service_name")
        self.parent_frame = self.param("parent_frame")
        self.child_frame = self.param("child_frame")
        self.auto_assign_child_frame = self.param("auto_assign_child_frame")
        self.publish_camera_link_tf = self.param("publish_camera_link_tf")
        self.publish_world_camera_tf = self.param("publish_world_camera_tf")
        self.world_frame = self.param("world_frame")
        self.world_lookup_timeout = float(self.param("world_lookup_timeout"))
        cache_file = self.param("world_tf_cache_file")
        if not os.path.isabs(cache_file):
            cache_file = os.path.join(
                os.path.dirname(__file__), "..", "config", "world_camera_tfs.json"
            )
        self.world_tf_cache_file = os.path.realpath(os.path.expanduser(cache_file))
        self.saved_world_tf_publish_rate = float(
            self.param("saved_world_tf_publish_rate")
        )
        self.camera_link_offset = self.params(
            "camera_link_offset_x",
            "camera_link_offset_y",
            "camera_link_offset_z",
        )
        self.camera_link_quat = R.from_euler(
            "zxy",
            self.params(
                "camera_link_rotation_z_deg",
                "camera_link_rotation_x_deg",
                "camera_link_rotation_y_deg",
            ),
            degrees=True
        ).as_quat()
        self.squares_x = self.param("squares_x")
        self.squares_y = self.param("squares_y")
        self.square_length = self.param("square_length")
        self.marker_length = self.param("marker_length")
        self.board_id_configs = [dict(config) for config in self.BOARD_ID_CONFIGS]

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.useAruco3Detection = False
        self.detector_params.minSideLengthCanonicalImg = 16
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 4
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

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.latest_image_msg = None

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

        self.detect_service = self.create_service(
            Trigger,
            self.detect_service_name,
            self.detect_service_callback
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

        self.log_startup()

    def param(self, name):
        return self.get_parameter(name).value

    def params(self, *names):
        return tuple(self.param(name) for name in names)

    def log_startup(self):
        for message in (
            "ChArUco detector node started",
            f"image_topic: {self.image_topic}",
            f"camera_info_topic: {self.camera_info_topic}",
            f"detect_service_name: {self.detect_service_name}",
            f"board: {self.squares_x}x{self.squares_y}",
            f"square_length: {self.square_length} m",
            f"marker_length: {self.marker_length} m",
            f"auto_assign_child_frame: {self.auto_assign_child_frame}",
            f"publish_world_camera_tf: {self.publish_world_camera_tf}, "
            f"world_frame: {self.world_frame}",
            f"world_tf_cache_file: {self.world_tf_cache_file}",
        ):
            self.get_logger().info(message)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        detectors = [self.charuco_detector]
        detectors.extend(config["detector"] for config in self.board_id_configs)
        for detector in detectors:
            self.set_camera_parameters_for_detector(detector)

        self.camera_info_received = True

        self.get_logger().info("CameraInfo received")
        self.get_logger().info(f"camera_matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"dist_coeffs: {self.dist_coeffs}")

    def image_callback(self, msg: Image):
        self.latest_image_msg = msg

    def detect_service_callback(self, request, response):
        del request

        if not self.camera_info_received:
            self.get_logger().warn(
                "Waiting for CameraInfo...",
                throttle_duration_sec=2.0
            )
            self.publish_saved_world_transforms(None, "waiting for CameraInfo")
            return self.set_detect_response(response, False, "waiting for CameraInfo")

        if self.latest_image_msg is None:
            self.publish_saved_world_transforms(None, "no image has been received")
            return self.set_detect_response(
                response,
                False,
                "no image has been received"
            )

        success, message = self.detect_charuco(self.latest_image_msg)
        return self.set_detect_response(response, success, message)

    def set_detect_response(self, response, success, detail):
        response.success = success
        response.message = "success" if success else f"false: {detail}"
        return response

    def fail_detection(self, msg, debug_frame, reason):
        self.handle_detection_failure(msg, debug_frame, reason)
        return False, reason

    def detect_charuco(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            self.publish_saved_world_transforms(
                msg.header.stamp,
                "cv_bridge conversion failed"
            )
            return False, f"cv_bridge conversion failed: {e}"

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            selected_board, selected_detector, child_frame_id, camera_link_frame_id = (
                self.select_board(gray, debug_frame)
            )
            if selected_board is None:
                return self.fail_detection(
                    msg,
                    debug_frame,
                    "no markers detected or marker IDs did not match any configured range"
                )

            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                selected_detector.detectBoard(gray)
            )
        except cv2.error as e:
            return self.fail_detection(
                msg,
                debug_frame,
                f"ChArUco detection skipped: {e}"
            )

        self.draw_markers(debug_frame, marker_corners, marker_ids)

        if charuco_ids is None or len(charuco_ids) < 6:
            return self.fail_detection(
                msg,
                debug_frame,
                "not enough ChArUco corners"
            )

        cv2.aruco.drawDetectedCornersCharuco(
            debug_frame,
            charuco_corners,
            charuco_ids
        )

        self.get_logger().info(
            f"Detected ChArUco assigned to {child_frame_id}",
            throttle_duration_sec=1.0
        )

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
            return self.fail_detection(
                msg,
                debug_frame,
                f"ChArUco pose estimation skipped: {e}"
            )

        if not success:
            return self.fail_detection(msg, debug_frame, "solvePnP failed")

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
        return True, f"detected ChArUco assigned to {child_frame_id}"

    def select_board(self, gray, debug_frame):
        if not self.auto_assign_child_frame:
            return self.board, self.charuco_detector, self.child_frame, None

        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        self.draw_markers(debug_frame, marker_corners, marker_ids)

        if marker_ids is None or len(marker_ids) == 0:
            self.get_logger().warn(
                "No ArUco markers detected in the image",
                throttle_duration_sec=2.0,
            )
            return None, None, None, None

        self.get_logger().info(
            f"Detected marker IDs: {sorted(marker_ids.flatten().tolist())}",
            throttle_duration_sec=2.0,
        )

        config = self.resolve_board_config_from_marker_ids(marker_ids)
        if config is None:
            return None, None, None, None

        return (
            config["board"],
            config["detector"],
            config["child_frame"],
            config["camera_link_frame"],
        )

    def draw_markers(self, frame, marker_corners, marker_ids):
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    def create_charuco_board(self, marker_ids=None):
        args = [
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict,
        ]
        if marker_ids is not None:
            args.append(marker_ids)
        return cv2.aruco.CharucoBoard(*args)

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

    def create_transform(self, stamp, parent_frame, child_frame, translation, quat):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        self.set_translation(transform, translation)
        self.set_rotation(transform, quat)
        return transform

    def set_translation(self, transform, translation):
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

    def set_rotation(self, transform, quat):
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

    def publish_tf(self, image_msg: Image, rvec, tvec, child_frame_id,
                   camera_link_frame_id=None):
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()  # x, y, z, w
        parent_frame = image_msg.header.frame_id or self.parent_frame
        transform = self.create_transform(
            image_msg.header.stamp,
            parent_frame,
            child_frame_id,
            tvec.flatten(),
            quat,
        )

        transforms = [transform]

        if (
            self.publish_camera_link_tf
            and camera_link_frame_id
            and not self.publish_world_camera_tf
        ):
            transforms.append(
                self.create_transform(
                    image_msg.header.stamp,
                    child_frame_id,
                    camera_link_frame_id,
                    self.camera_link_offset,
                    self.camera_link_quat,
                )
            )

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

        return self.create_transform(
            image_msg.header.stamp,
            self.world_frame,
            camera_link_frame_id,
            world_t_camera,
            world_r_camera.as_quat(),
        )

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
            "translation": self.xyz_to_dict(transform.transform.translation),
            "rotation": self.xyzw_to_dict(transform.transform.rotation),
        }

    def transform_from_cache(self, child_frame_id, value):
        return self.create_transform(
            self.get_clock().now().to_msg(),
            value.get("parent_frame", self.world_frame),
            child_frame_id,
            self.xyz_from_dict(value["translation"]),
            self.xyzw_from_dict(value["rotation"]),
        )

    def xyz_to_dict(self, value):
        return {"x": value.x, "y": value.y, "z": value.z}

    def xyzw_to_dict(self, value):
        return {"x": value.x, "y": value.y, "z": value.z, "w": value.w}

    def xyz_from_dict(self, value):
        return [float(value[key]) for key in ("x", "y", "z")]

    def xyzw_from_dict(self, value):
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
    node = CharucoDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
