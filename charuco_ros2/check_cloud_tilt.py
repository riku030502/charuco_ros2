#!/usr/bin/env python3
"""点群の主平面がworld水平面からどれだけ傾いているかを測る。

床や机は本来world座標で水平（法線が +Z）なので、点群から平面を当てて
その法線の傾きを見れば、`world -> *_camera_link` の回転誤差がそのまま出る。

  ros2 run charuco_ros2 check_cloud_tilt \
    --ros-args -p cloud_topic:=/camera/left_camera/depth/color/points

測るだけで、TFの修正はしない。出た角度を detect_charuco の
camera_link_rotation_* に反映するか、カメラを置き直して detect_charuco を
やり直すかは、値を見てから決めること。
"""
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from scipy.spatial.transform import Rotation as R


class CloudTiltCheckerNode(Node):
    def __init__(self):
        super().__init__("cloud_tilt_checker_node")

        self.declare_parameter(
            "cloud_topic",
            "/camera/left_camera/depth/color/points",
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("max_points", 20000)
        # 平面フィットの許容誤差[m]。RealSenseの奥行きノイズより少し大きく取る。
        self.declare_parameter("ransac_threshold", 0.01)
        self.declare_parameter("ransac_iterations", 200)
        self.declare_parameter("report_interval", 2.0)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.max_points = int(self.get_parameter("max_points").value)
        self.ransac_threshold = float(self.get_parameter("ransac_threshold").value)
        self.ransac_iterations = int(self.get_parameter("ransac_iterations").value)
        self.report_interval = float(self.get_parameter("report_interval").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rng = np.random.default_rng(0)
        self.last_report_time = None

        # 点群はBEST_EFFORTで飛んでくる。
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos,
        )

        self.get_logger().info(f"cloud_topic: {self.cloud_topic}")
        self.get_logger().info(f"world_frame: {self.world_frame}")

    def cloud_callback(self, msg: PointCloud2):
        if not self.should_report():
            return

        points = self.read_points(msg)
        if points is None or len(points) < 100:
            self.get_logger().warn(
                "Not enough finite points in the cloud",
                throttle_duration_sec=2.0,
            )
            return

        world_points = self.to_world(points, msg.header.frame_id)
        if world_points is None:
            return

        plane = self.fit_plane(world_points)
        if plane is None:
            self.get_logger().warn("Plane fit failed", throttle_duration_sec=2.0)
            return

        normal, offset, inlier_count = plane
        self.report(normal, offset, inlier_count, len(world_points))

    def should_report(self):
        now = self.get_clock().now()

        if self.last_report_time is not None:
            elapsed = (now - self.last_report_time).nanoseconds * 1e-9
            if elapsed < self.report_interval:
                return False

        self.last_report_time = now
        return True

    def read_points(self, msg: PointCloud2):
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        if len(points) == 0:
            return None

        if len(points) > self.max_points:
            index = self.rng.choice(len(points), self.max_points, replace=False)
            points = points[index]

        return points.astype(np.float64)

    def to_world(self, points, cloud_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                cloud_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF lookup failed: {self.world_frame} -> {cloud_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        rotation = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])

        return rotation.apply(points) + translation

    def fit_plane(self, points):
        """RANSACで主平面を当てる。戻り値は (法線, 原点からの距離, インライア数)。"""
        best_inliers = None
        best_count = 0

        for _ in range(self.ransac_iterations):
            sample = points[self.rng.choice(len(points), 3, replace=False)]

            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm

            distance = np.abs((points - sample[0]) @ normal)
            inliers = distance < self.ransac_threshold
            count = int(np.count_nonzero(inliers))

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < 50:
            return None

        # インライアだけで最小二乗フィットし直す（3点サンプルより安定する）。
        inlier_points = points[best_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid)
        normal = vh[2]

        # 法線は上向きに揃えておく。
        if normal[2] < 0.0:
            normal = -normal

        return normal, float(centroid @ normal), best_count

    def report(self, normal, offset, inlier_count, total_count):
        tilt_deg = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))

        # 傾きを打ち消す回転（この分だけ world -> camera_link がずれている）。
        axis = np.cross(normal, [0.0, 0.0, 1.0])
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-9:
            correction = np.zeros(3)
        else:
            correction = R.from_rotvec(
                axis / axis_norm * np.radians(tilt_deg)
            ).as_euler("xyz", degrees=True)

        self.get_logger().info(
            f"plane inliers={inlier_count}/{total_count} | "
            f"normal=({normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}) | "
            f"tilt from horizontal = {tilt_deg:.2f} deg | "
            f"height at centroid = {offset:+.3f} m | "
            f"correction rpy(xyz,deg) = "
            f"({correction[0]:+.2f}, {correction[1]:+.2f}, {correction[2]:+.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CloudTiltCheckerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
