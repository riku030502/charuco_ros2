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
from apriltag import apriltag as AprilTagDetector


class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__("apriltag_detector_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/apriltag/debug_image")

        self.declare_parameter("parent_frame", "camera_color_optical_frame")
        self.declare_parameter("child_frame", "apriltag")

        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("tag_size", 0.072)  # [m] 7.2 cm
        self.declare_parameter("target_tag_id", -1)
        self.declare_parameter("publish_all_tags", False)

        self.declare_parameter("threads", 1)
        self.declare_parameter("max_hamming", 1)
        self.declare_parameter("decimate", 1.0)
        self.declare_parameter("blur", 0.0)
        self.declare_parameter("refine_edges", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value

        self.tag_family = self.get_parameter("tag_family").value
        self.tag_size = self.get_parameter("tag_size").value
        self.target_tag_id = self.get_parameter("target_tag_id").value
        self.publish_all_tags = self.get_parameter("publish_all_tags").value

        self.detector = AprilTagDetector(
            self.tag_family,
            threads=self.get_parameter("threads").value,
            maxhamming=self.get_parameter("max_hamming").value,
            decimate=self.get_parameter("decimate").value,
            blur=self.get_parameter("blur").value,
            refine_edges=self.get_parameter("refine_edges").value,
        )

        half = self.tag_size / 2.0
        # apriltag.detect() returns corners in lb, rb, rt, lt order.
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
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

        self.get_logger().info("AprilTag detector node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"tag_family: {self.tag_family}")
        self.get_logger().info(f"tag_size: {self.tag_size} m")
        self.get_logger().info(f"target_tag_id: {self.target_tag_id}")

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
            self.get_logger().warn("Waiting for CameraInfo...", throttle_duration_sec=2.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            detections = self.detector.detect(gray)
        except Exception as e:
            self.get_logger().debug(
                f"AprilTag detection skipped: {e}",
                throttle_duration_sec=2.0,
            )
            self.publish_debug_image(debug_frame, msg.header)
            return

        detections = self.filter_detections(detections)
        if len(detections) == 0:
            self.publish_debug_image(debug_frame, msg.header)
            return

        for detection in detections:
            self.draw_detection(debug_frame, detection)

            image_points = np.array(detection["lb-rb-rt-lt"], dtype=np.float32)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except cv2.error as e:
                self.get_logger().debug(
                    f"AprilTag pose estimation skipped: {e}",
                    throttle_duration_sec=2.0,
                )
                continue

            if not success:
                continue

            cv2.drawFrameAxes(
                debug_frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.tag_size * 0.5,
            )

            tag_id = int(detection["id"])
            self.publish_tf(msg, rvec, tvec, tag_id)
            self.get_logger().info(
                f"Detected AprilTag id={tag_id}: tvec = "
                f"[{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]",
                throttle_duration_sec=1.0,
            )

        self.publish_debug_image(debug_frame, msg.header)

    def filter_detections(self, detections):
        if self.target_tag_id >= 0:
            return [d for d in detections if int(d["id"]) == self.target_tag_id]

        if self.publish_all_tags:
            return detections

        return sorted(
            detections,
            key=lambda d: float(d.get("margin", 0.0)),
            reverse=True,
        )[:1]

    def draw_detection(self, frame, detection):
        corners = np.array(detection["lb-rb-rt-lt"], dtype=np.int32)
        center = tuple(np.array(detection["center"], dtype=np.int32))
        tag_id = int(detection["id"])

        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"id:{tag_id}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def publish_tf(self, image_msg: Image, rvec, tvec, tag_id):
        transform = TransformStamped()

        transform.header.stamp = image_msg.header.stamp

        if image_msg.header.frame_id:
            transform.header.frame_id = image_msg.header.frame_id
        else:
            transform.header.frame_id = self.parent_frame

        if self.publish_all_tags and self.target_tag_id < 0:
            transform.child_frame_id = f"{self.child_frame}_{tag_id}"
        else:
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
    node = AprilTagDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
