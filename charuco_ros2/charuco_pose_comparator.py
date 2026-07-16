#!/usr/bin/env python3
import csv
import os
from datetime import datetime, timezone

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException
from tf2_ros import LookupException, TransformListener


class CharucoPoseComparatorNode(Node):
    CSV_FIELDS = [
        "timestamp_utc",
        "base_frame",
        "ext_frame",
        "hand_frame",
        "translation_error_mm",
        "rotation_error_deg",
        "ext_x_m",
        "ext_y_m",
        "ext_z_m",
        "hand_x_m",
        "hand_y_m",
        "hand_z_m",
        "ext_age_sec",
        "hand_age_sec",
    ]

    def __init__(self):
        super().__init__("charuco_pose_comparator")

        self.declare_parameter("input_mode", "tf")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ext_target_frame", "target_pose_ext")
        self.declare_parameter("hand_target_frame", "target_pose_hand")
        self.declare_parameter("ext_pose_topic", "/target_pose_ext/pose")
        self.declare_parameter("hand_pose_topic", "/target_pose_hand/pose")
        self.declare_parameter("compare_rate", 1.0)
        self.declare_parameter("lookup_timeout", 0.05)
        self.declare_parameter("max_pose_age_sec", 2.0)
        self.declare_parameter("csv_path", "")

        self.input_mode = self.get_parameter("input_mode").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ext_target_frame = self.get_parameter("ext_target_frame").value
        self.hand_target_frame = self.get_parameter("hand_target_frame").value
        self.ext_pose_topic = self.get_parameter("ext_pose_topic").value
        self.hand_pose_topic = self.get_parameter("hand_pose_topic").value
        self.compare_rate = float(self.get_parameter("compare_rate").value)
        self.lookup_timeout = float(self.get_parameter("lookup_timeout").value)
        self.max_pose_age_sec = float(
            self.get_parameter("max_pose_age_sec").value
        )
        self.csv_path = os.path.expanduser(
            self.get_parameter("csv_path").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ext_pose = None
        self.hand_pose = None

        if self.input_mode not in ("tf", "topics"):
            raise ValueError("input_mode must be 'tf' or 'topics'")

        if self.input_mode == "topics":
            self.ext_sub = self.create_subscription(
                PoseStamped,
                self.ext_pose_topic,
                self.ext_pose_callback,
                10,
            )
            self.hand_sub = self.create_subscription(
                PoseStamped,
                self.hand_pose_topic,
                self.hand_pose_callback,
                10,
            )

        if self.csv_path:
            self.ensure_csv_header()

        self.timer = self.create_timer(
            1.0 / self.compare_rate,
            self.compare_once,
        )
        self.get_logger().info("ChArUco pose comparator started")
        self.get_logger().info(f"input_mode: {self.input_mode}")
        self.get_logger().info(f"base_frame: {self.base_frame}")
        self.get_logger().info(
            f"external target: {self.ext_target_frame}, hand target: "
            f"{self.hand_target_frame}"
        )
        if self.csv_path:
            self.get_logger().info(f"csv_path: {self.csv_path}")

    def ext_pose_callback(self, msg: PoseStamped):
        self.ext_pose = msg

    def hand_pose_callback(self, msg: PoseStamped):
        self.hand_pose = msg

    def compare_once(self):
        if self.input_mode == "tf":
            ext = self.lookup_target_pose(self.ext_target_frame)
            hand = self.lookup_target_pose(self.hand_target_frame)
        else:
            ext = self.pose_to_base(self.ext_pose, self.ext_target_frame)
            hand = self.pose_to_base(self.hand_pose, self.hand_target_frame)

        if ext is None or hand is None:
            return

        ext_t, ext_r, ext_age = ext
        hand_t, hand_r, hand_age = hand

        if not self.poses_are_fresh(ext_age, hand_age):
            return

        delta_t = ext_t - hand_t
        translation_error_mm = float(np.linalg.norm(delta_t) * 1000.0)
        delta_r = hand_r.inv() * ext_r
        rotation_error_deg = float(delta_r.magnitude() * 180.0 / np.pi)

        self.get_logger().info(
            "ChArUco validation error: "
            f"translation={translation_error_mm:.2f} mm, "
            f"rotation={rotation_error_deg:.3f} deg",
            throttle_duration_sec=1.0,
        )

        if self.csv_path:
            self.append_csv(
                translation_error_mm,
                rotation_error_deg,
                ext_t,
                hand_t,
                ext_age,
                hand_age,
            )

    def lookup_target_pose(self, target_frame: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                target_frame,
                Time(),
                timeout=Duration(seconds=self.lookup_timeout),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"Waiting for TF {self.base_frame} -> {target_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        t, r = self.transform_to_pose(transform)
        return t, r, self.message_age_sec(transform)

    def pose_to_base(self, pose_msg: PoseStamped | None, target_name: str):
        if pose_msg is None:
            self.get_logger().warn(
                f"Waiting for pose topic for {target_name}",
                throttle_duration_sec=2.0,
            )
            return None

        frame_id = pose_msg.header.frame_id
        pose_t = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        pose_r = R.from_quat(
            [
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ]
        )

        if frame_id == self.base_frame:
            return pose_t, pose_r, self.message_age_sec(pose_msg)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                Time(),
                timeout=Duration(seconds=self.lookup_timeout),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"Waiting for TF {self.base_frame} -> {frame_id}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        base_t_frame, base_r_frame = self.transform_to_pose(transform)
        base_t_target = base_t_frame + base_r_frame.apply(pose_t)
        base_r_target = base_r_frame * pose_r
        return base_t_target, base_r_target, self.message_age_sec(pose_msg)

    def poses_are_fresh(self, ext_age, hand_age) -> bool:
        if self.max_pose_age_sec <= 0.0:
            return True

        stale = []
        if ext_age is not None and ext_age > self.max_pose_age_sec:
            stale.append(f"external age={ext_age:.2f}s")
        if hand_age is not None and hand_age > self.max_pose_age_sec:
            stale.append(f"hand age={hand_age:.2f}s")

        if stale:
            self.get_logger().warn(
                "Skipping comparison because pose/TF is stale: "
                + ", ".join(stale),
                throttle_duration_sec=2.0,
            )
            return False

        return True

    def message_age_sec(self, msg):
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return None

        now = self.get_clock().now()
        msg_time = Time.from_msg(stamp)
        return (now - msg_time).nanoseconds / 1e9

    def transform_to_pose(self, transform: TransformStamped):
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        rotation = R.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        )
        return translation, rotation

    def ensure_csv_header(self):
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        if (
            os.path.exists(self.csv_path)
            and os.path.getsize(self.csv_path) > 0
        ):
            return

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

    def append_csv(
        self,
        translation_error_mm,
        rotation_error_deg,
        ext_t,
        hand_t,
        ext_age,
        hand_age,
    ):
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_frame": self.base_frame,
            "ext_frame": self.ext_target_frame,
            "hand_frame": self.hand_target_frame,
            "translation_error_mm": f"{translation_error_mm:.6f}",
            "rotation_error_deg": f"{rotation_error_deg:.6f}",
            "ext_x_m": f"{ext_t[0]:.9f}",
            "ext_y_m": f"{ext_t[1]:.9f}",
            "ext_z_m": f"{ext_t[2]:.9f}",
            "hand_x_m": f"{hand_t[0]:.9f}",
            "hand_y_m": f"{hand_t[1]:.9f}",
            "hand_z_m": f"{hand_t[2]:.9f}",
            "ext_age_sec": "" if ext_age is None else f"{ext_age:.6f}",
            "hand_age_sec": "" if hand_age is None else f"{hand_age:.6f}",
        }
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = CharucoPoseComparatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
