#!/usr/bin/env python3
import os

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformBroadcaster,
    TransformListener,
)


class FindCubeNode(Node):
    def __init__(self):
        super().__init__("find_cube_node")

        self.declare_parameter("image_topic", "/camera/hand_camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/hand_camera/color/camera_info")
        self.declare_parameter(
            "depth_image_topic",
            "/camera/hand_camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter("debug_image_topic", "/find_cube/debug_image")
        self.declare_parameter("parent_frame", "")
        self.declare_parameter("right_child_frame", "right_cube_color_frame")
        self.declare_parameter("left_child_frame", "left_cube_color_frame")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_base_tf", True)
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter(
            "base_right_child_frame",
            "right_cube_color_link_base_frame",
        )
        self.declare_parameter(
            "base_left_child_frame",
            "left_cube_color_link_base_frame",
        )
        self.declare_parameter("base_lookup_timeout", 0.02)
        self.declare_parameter("save_detected_pose", True)
        self.declare_parameter(
            "cube_pose_cache_file",
            "charuco_ros2/config/cube_target_poses.yaml",
        )
        self.declare_parameter("pose_save_interval_sec", 0.5)
        self.declare_parameter("use_depth_distance", True)
        self.declare_parameter("fallback_to_size_distance", True)
        self.declare_parameter("depth_sample_radius", 3)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        # 距離推定に使う、実物の色付き正方形フレームの一辺の長さ[m]。
        # 画像上の見かけの大きさ(pixel_size)とこの実寸から、カメラまでの距離を計算する。
        self.declare_parameter("frame_size_m", 0.05)

        # 輪郭検出後のフィルタ条件。
        # 小さいノイズ、細長い領域、正方形から外れた領域をここで落とす。
        self.declare_parameter("min_contour_area", 50.0)
        self.declare_parameter("max_contour_area", 80000.0)
        self.declare_parameter("min_rectangularity", 0.15)
        self.declare_parameter("square_aspect_tolerance", 0.25)
        self.declare_parameter("max_frame_aspect_ratio", 2.8)
        self.declare_parameter("polygon_epsilon_ratio", 0.05)
        self.declare_parameter("log_interval_sec", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_image_topic = self.get_parameter("depth_image_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.right_child_frame = self.get_parameter("right_child_frame").value
        self.left_child_frame = self.get_parameter("left_child_frame").value
        self.publish_tf = self.get_parameter("publish_tf").value
        self.publish_base_tf = self.get_parameter("publish_base_tf").value
        self.base_frame = self.get_parameter("base_frame").value
        self.base_right_child_frame = self.get_parameter(
            "base_right_child_frame"
        ).value
        self.base_left_child_frame = self.get_parameter(
            "base_left_child_frame"
        ).value
        self.base_lookup_timeout = self.get_parameter("base_lookup_timeout").value
        self.save_detected_pose = bool(
            self.get_parameter("save_detected_pose").value
        )
        self.cube_pose_cache_file = self.resolve_cache_file(
            self.get_parameter("cube_pose_cache_file").value
        )
        self.pose_save_interval_sec = float(
            self.get_parameter("pose_save_interval_sec").value
        )
        self.use_depth_distance = self.get_parameter("use_depth_distance").value
        self.fallback_to_size_distance = self.get_parameter(
            "fallback_to_size_distance"
        ).value
        self.depth_sample_radius = self.get_parameter("depth_sample_radius").value
        self.depth_scale = self.get_parameter("depth_scale").value
        self.min_depth_m = self.get_parameter("min_depth_m").value
        self.max_depth_m = self.get_parameter("max_depth_m").value
        self.frame_size_m = self.get_parameter("frame_size_m").value
        self.min_contour_area = self.get_parameter("min_contour_area").value
        self.max_contour_area = self.get_parameter("max_contour_area").value
        self.min_rectangularity = self.get_parameter("min_rectangularity").value
        self.square_aspect_tolerance = self.get_parameter(
            "square_aspect_tolerance"
        ).value
        self.max_frame_aspect_ratio = self.get_parameter(
            "max_frame_aspect_ratio"
        ).value
        self.polygon_epsilon_ratio = self.get_parameter(
            "polygon_epsilon_ratio"
        ).value
        self.log_interval_sec = self.get_parameter("log_interval_sec").value

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CameraInfo が届くまではカメラ内部パラメータが分からないため、距離と3次元位置は計算しない。
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.latest_depth_image = None

        # 左右それぞれで最後にログを出した時刻を保持し、ログ出力を間引く。
        self.last_log_time = {
            "right": None,
            "left": None,
        }
        self.last_pose_save_time = {
            "right": None,
            "left": None,
        }

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

        self.depth_image_sub = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_image_callback,
            10,
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )

        self.get_logger().info("find_cube node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"depth_image_topic: {self.depth_image_topic}")
        self.get_logger().info(f"debug_image_topic: {self.debug_image_topic}")
        self.get_logger().info(f"parent_frame: {self.parent_frame or '<image header>'}")
        self.get_logger().info(f"right_child_frame: {self.right_child_frame}")
        self.get_logger().info(f"left_child_frame: {self.left_child_frame}")
        self.get_logger().info(f"publish_tf: {self.publish_tf}")
        self.get_logger().info(f"publish_base_tf: {self.publish_base_tf}")
        self.get_logger().info(f"base_frame: {self.base_frame}")
        self.get_logger().info(f"use_depth_distance: {self.use_depth_distance}")
        self.get_logger().info(
            f"fallback_to_size_distance: {self.fallback_to_size_distance}"
        )
        self.get_logger().info(
            f"base_right_child_frame: {self.base_right_child_frame}"
        )
        self.get_logger().info(
            f"base_left_child_frame: {self.base_left_child_frame}"
        )
        self.get_logger().info(f"save_detected_pose: {self.save_detected_pose}")
        self.get_logger().info(f"cube_pose_cache_file: {self.cube_pose_cache_file}")
        self.get_logger().info(f"frame_size_m: {self.frame_size_m}")

    def resolve_cache_file(self, cache_file):
        if os.path.isabs(cache_file):
            return os.path.realpath(os.path.expanduser(cache_file))

        package_relative_prefix = "charuco_ros2/"
        if cache_file.startswith(package_relative_prefix):
            cache_file = cache_file[len(package_relative_prefix):]

        return os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", cache_file)
        )

    def camera_info_callback(self, msg: CameraInfo):
        # カメラ内部パラメータ行列 K の (0, 0) が x方向の焦点距離 fx。
        # K の (1, 1) が y方向の焦点距離 fy、(0, 2)/(1, 2) が画像中心(cx, cy)。
        # 距離推定と、画像上の中心点をカメラ座標へ戻す計算に使う。
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_image_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )
        except CvBridgeError as error:
            self.get_logger().warn(f"failed to convert depth image: {error}")
            return

        self.latest_depth_image = depth_image

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warn(f"failed to convert image: {error}")
            return

        # BGRのまま色をしきい値処理すると明るさの影響を受けやすい。
        # HSVに変換して、色相(H)を中心にマゼンタ枠の領域を取り出す。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        magenta_mask = self.make_magenta_mask(hsv)

        # マスク画像から「正方形フレームらしい」候補だけを抽出する。
        # 両方ともマゼンタなので、色ではなく画像内の左右位置で分類する。
        magenta_detections = self.find_frames(magenta_mask)
        # CameraInfo から fx が取得できていれば、各候補に distance_m を追加する。
        self.add_distances(magenta_detections)
        left_detections, right_detections = self.split_detections_by_image_side(
            magenta_detections,
            frame.shape[1],
        )

        if right_detections:
            self.log_detection(
                "right",
                "色枠を検出しました",
                right_detections,
            )
            self.publish_detection_tf(
                "right",
                self.right_child_frame,
                self.base_right_child_frame,
                right_detections,
                msg.header,
            )

        if left_detections:
            self.log_detection(
                "left",
                "色枠を検出しました",
                left_detections,
            )
            self.publish_detection_tf(
                "left",
                self.left_child_frame,
                self.base_left_child_frame,
                left_detections,
                msg.header,
            )

        # デバッグ画像には、元の輪郭・近似した四角形・外接矩形・距離を重ねて描画する。
        # さらにマゼンタマスクを半透明で重ねてマスク段階の問題を視覚確認できるようにする。
        debug_frame = frame.copy()
        magenta_overlay = np.zeros_like(frame)
        magenta_overlay[:, :, 0] = magenta_mask
        magenta_overlay[:, :, 2] = magenta_mask
        debug_frame = cv2.addWeighted(debug_frame, 1.0, magenta_overlay, 0.4, 0)
        self.draw_detections(
            debug_frame,
            right_detections,
            color=(255, 0, 255),
            label="color frame",
        )
        self.draw_detections(
            debug_frame,
            left_detections,
            color=(255, 0, 255),
            label="color frame",
        )
        self.publish_debug_image(debug_frame, msg)

    def make_magenta_mask(self, hsv):
        # OpenCVのHSVではマゼンタはH=150付近。
        # 照明で赤寄り/紫寄りに振れても拾えるよう少し広めに取る。
        lower_magenta = np.array([135, 80, 50], dtype=np.uint8)
        upper_magenta = np.array([170, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower_magenta, upper_magenta)

    def split_detections_by_image_side(self, detections, image_width):
        left_detections = []
        right_detections = []
        image_center_x = image_width / 2.0

        for detection in detections:
            center = detection.get("center")
            if center is None:
                continue

            center_x, _ = center
            if center_x < image_center_x:
                left_detections.append(detection)
            else:
                right_detections.append(detection)

        return left_detections, right_detections

    def find_frames(self, mask):
        # OPENは細い枠線を消してしまうため使わない。
        # CLOSEで途切れた枠線の隙間を埋めて4辺が繋がった輪郭にする。
        kernel_large = np.ones((9, 9), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

        # 外側の輪郭だけを使う。内側の穴や細かい内部輪郭は距離推定には不要。
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections = []

        for contour in contours:
            # 面積が小さい輪郭は、遠方の誤検出や照明ノイズである可能性が高い。
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                self.get_logger().warn(
                    f"contour rejected: area={area:.0f} < {self.min_contour_area}",
                    throttle_duration_sec=1.0,
                )
                continue

            if area > self.max_contour_area:
                self.get_logger().warn(
                    f"contour rejected: area={area:.0f} > {self.max_contour_area}",
                    throttle_duration_sec=1.0,
                )
                continue

            # bboxは描画用に軸平行の外接矩形を取る。
            x, y, width, height = cv2.boundingRect(contour)

            # rectangularity は「矩形に対して輪郭がどれだけ詰まっているか」。
            # 軸平行のboundingRectだと斜め45度の正方形で面積比が0.5まで落ちてしまうため、
            # 回転を考慮したminAreaRectで測って斜めの枠でも正しく評価する。
            rot_rect = cv2.minAreaRect(contour)
            rot_w, rot_h = rot_rect[1]
            rot_area = rot_w * rot_h
            if rot_area == 0:
                continue

            # 細い線、円弧、欠けた領域は値が低くなるのでここで除外する。
            rectangularity = area / rot_area
            if rectangularity < self.min_rectangularity:
                self.get_logger().warn(
                    f"contour rejected: rect={rectangularity:.2f} < {self.min_rectangularity}",
                    throttle_duration_sec=1.0,
                )
                continue

            # 輪郭点をそのまま見ると細かすぎるため、多角形近似で頂点数を減らす。
            # 近似後に「4〜8頂点の四角形に近いか」を判定する。
            epsilon = self.polygon_epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if not self.is_square_like(approx):
                rect = cv2.minAreaRect(approx)
                w, h = rect[1]
                aspect = (max(w, h) / min(w, h)) if min(w, h) > 0 else 999
                self.get_logger().warn(
                    f"contour rejected: vertices={len(approx)} aspect={aspect:.2f}",
                    throttle_duration_sec=1.0,
                )
                continue

            # 後段の描画・距離計算で必要な情報をまとめて保持する。
            detections.append(
                {
                    "contour": contour,
                    "approx": approx,
                    "bbox": (x, y, width, height),
                    "center": self.get_square_center(approx),
                    "pixel_size": self.get_square_pixel_size(approx),
                    "area": area,
                }
            )

        return detections

    def get_square_center(self, approx):
        # 回転外接矩形の中心を使うことで、斜めに写った枠でも中心位置を安定して取る。
        rect = cv2.minAreaRect(approx)
        center_x, center_y = rect[0]
        return center_x, center_y

    def get_square_pixel_size(self, approx):
        # minAreaRectを使うと、画像内で傾いた正方形でも回転を考慮したサイズが取れる。
        # 幅と高さの平均を代表サイズにして、片方向だけの揺れの影響を小さくする。
        rect = cv2.minAreaRect(approx)
        width, height = rect[1]
        if width == 0 or height == 0:
            return None
        return (width + height) / 2.0

    def add_distances(self, detections):
        if not detections:
            return

        if not self.has_camera_intrinsics():
            # CameraInfoを受け取る前は内部パラメータがないため、検出だけ行い距離は未設定にする。
            self.get_logger().warn(
                "Waiting for CameraInfo. Distance is not available.",
                throttle_duration_sec=2.0,
            )
            return

        for detection in detections:
            distance_m = self.get_depth_distance(detection)
            source = "depth"

            if distance_m is None and self.fallback_to_size_distance:
                distance_m = self.get_size_distance(detection)
                source = "size"

            detection["distance_m"] = distance_m
            detection["distance_source"] = source if distance_m is not None else None
            detection["position"] = self.project_detection_to_camera(detection)

    def get_depth_distance(self, detection):
        if not self.use_depth_distance or self.latest_depth_image is None:
            return None

        center = detection.get("center")
        if center is None:
            return None

        center_x, center_y = center
        col = int(round(center_x))
        row = int(round(center_y))
        depth = self.latest_depth_image
        if row < 0 or col < 0 or row >= depth.shape[0] or col >= depth.shape[1]:
            return None

        radius = max(int(self.depth_sample_radius), 0)
        row_min = max(row - radius, 0)
        row_max = min(row + radius + 1, depth.shape[0])
        col_min = max(col - radius, 0)
        col_max = min(col + radius + 1, depth.shape[1])
        patch = np.asarray(depth[row_min:row_max, col_min:col_max], dtype=np.float32)
        if patch.size == 0:
            return None

        values = patch[np.isfinite(patch)]
        values = values[values > 0.0]
        if values.size == 0:
            return None

        # RealSense の 16UC1 depth は通常 mm。32FC1 は m のことが多い。
        if np.issubdtype(depth.dtype, np.integer):
            values = values * float(self.depth_scale)

        values = values[
            (values >= float(self.min_depth_m))
            & (values <= float(self.max_depth_m))
        ]
        if values.size == 0:
            return None

        return float(np.median(values))

    def get_size_distance(self, detection):
        pixel_size = detection["pixel_size"]
        if pixel_size is None or pixel_size <= 0:
            return None

        # フォールバック用。depth が取れないときだけ、見かけサイズから距離を推定する。
        return self.fx * self.frame_size_m / pixel_size

    def has_camera_intrinsics(self):
        return all(
            value is not None
            for value in [self.fx, self.fy, self.cx, self.cy]
        )

    def project_detection_to_camera(self, detection):
        distance_m = detection.get("distance_m")
        center = detection.get("center")
        if distance_m is None or center is None:
            return None

        center_x, center_y = center

        # カメラ光学座標系では、xが画像右方向、yが画像下方向、zがカメラ前方。
        # 画像中心からのずれを焦点距離で割り、距離zを掛けて3次元位置に戻す。
        x = (center_x - self.cx) * distance_m / self.fx
        y = (center_y - self.cy) * distance_m / self.fy
        z = distance_m
        return x, y, z

    def publish_detection_tf(
        self,
        key,
        child_frame,
        base_child_frame,
        detections,
        image_header,
    ):
        if not self.publish_tf:
            return

        detection = self.get_nearest_detection(detections)
        if detection is None:
            return

        position = detection.get("position")
        if position is None:
            return

        parent_frame = self.parent_frame or image_header.frame_id
        if not parent_frame:
            self.get_logger().warn(
                f"{key} frame TF skipped because parent frame is empty.",
                throttle_duration_sec=2.0,
            )
            return

        camera_transform = self.create_transform(
            image_header.stamp,
            parent_frame,
            child_frame,
            position,
        )

        transforms = [camera_transform]
        base_transform = self.create_base_transform(
            key,
            image_header.stamp,
            parent_frame,
            base_child_frame,
            position,
        )
        if base_transform is not None:
            transforms.append(base_transform)
            self.save_detected_pose_to_yaml(key, base_transform)

        self.tf_broadcaster.sendTransform(transforms)

    def create_transform(self, stamp, parent_frame, child_frame, position):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])

        # 色枠検出からは向きまでは安定して推定しないため、位置だけを持つidentity姿勢として配信する。
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        return transform

    def create_base_transform(
        self,
        key,
        stamp,
        parent_frame,
        base_child_frame,
        position,
    ):
        if not self.publish_base_tf or not self.base_frame or not base_child_frame:
            return None

        try:
            base_to_parent = self.tf_buffer.lookup_transform(
                self.base_frame,
                parent_frame,
                Time(),
                timeout=Duration(seconds=float(self.base_lookup_timeout)),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(
                f"{key} base TF lookup skipped: "
                f"{self.base_frame} -> {parent_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        # detect_charuco.py と同じく、画像の親フレームから基準フレームへの
        # TFを引いて、検出点をロボット基準(link_base)の座標へ変換して配信する。
        parent_position = np.array(position, dtype=np.float64)
        base_position = self.transform_point(base_to_parent, parent_position)
        return self.create_transform(
            stamp,
            self.base_frame,
            base_child_frame,
            base_position,
        )

    def save_detected_pose_to_yaml(self, key, transform):
        if not self.save_detected_pose:
            return

        now = self.get_clock().now()
        last_time = self.last_pose_save_time.get(key)
        if last_time is not None and self.pose_save_interval_sec > 0.0:
            elapsed = (now - last_time).nanoseconds / 1e9
            if elapsed < self.pose_save_interval_sec:
                return

        self.last_pose_save_time[key] = now

        data = self.load_cube_pose_cache()
        data.setdefault("cubes", {})
        pose_record = {
            "cube_pose": self.transform_to_cache_pose(transform),
            "source": "find_cube_detection",
        }
        data["cubes"][key] = pose_record
        data["cubes"]["latest"] = pose_record

        os.makedirs(os.path.dirname(self.cube_pose_cache_file), exist_ok=True)
        with open(self.cube_pose_cache_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False)

    def load_cube_pose_cache(self):
        if not os.path.exists(self.cube_pose_cache_file):
            return {"cubes": {}}

        with open(self.cube_pose_cache_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            return {"cubes": {}}
        data.setdefault("cubes", {})
        return data

    def transform_to_cache_pose(self, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "base_frame": transform.header.frame_id,
            "child_frame": transform.child_frame_id,
            "stamp": {
                "sec": int(transform.header.stamp.sec),
                "nanosec": int(transform.header.stamp.nanosec),
            },
            "translation_m": {
                "x": float(translation.x),
                "y": float(translation.y),
                "z": float(translation.z),
            },
            "rotation_xyzw": {
                "x": float(rotation.x),
                "y": float(rotation.y),
                "z": float(rotation.z),
                "w": float(rotation.w),
            },
        }

    def transform_point(self, transform, point):
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=np.float64)
        rotation = self.quaternion_to_matrix([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])
        return translation + rotation.dot(point)

    def quaternion_to_matrix(self, quat):
        x, y, z, w = quat
        norm = x * x + y * y + z * z + w * w
        if norm == 0.0:
            return np.identity(3)

        scale = 2.0 / norm
        xx = x * x * scale
        yy = y * y * scale
        zz = z * z * scale
        xy = x * y * scale
        xz = x * z * scale
        yz = y * z * scale
        wx = w * x * scale
        wy = w * y * scale
        wz = w * z * scale

        return np.array([
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ], dtype=np.float64)

    def is_square_like(self, approx):
        # キューブを斜めから見ると複数面のマゼンタ枠が合わさって4〜8頂点の輪郭になる。
        # 4頂点のみでは斜め視点を取り逃すため、4〜8頂点を許容する。
        if not (4 <= len(approx) <= 8):
            return False

        rect = cv2.minAreaRect(approx)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False

        # 斜め姿勢や複数面がつながったマゼンタ枠では、見かけの外接矩形が
        # 2倍以上に伸びることがあるため、透視歪み用の上限まで許容する。
        aspect_ratio = max(width, height) / min(width, height)
        max_aspect_ratio = max(
            1.0 + self.square_aspect_tolerance,
            self.max_frame_aspect_ratio,
        )
        return aspect_ratio <= max_aspect_ratio

    def draw_detections(self, frame, detections, color, label):
        # labelは候補ごとに距離を付け足すので、元の文字列を保持しておく。
        base_label = label

        for detection in detections:
            x, y, width, height = detection["bbox"]
            contour = detection["contour"]
            approx = detection["approx"]
            distance_m = detection.get("distance_m")

            # contour: 実際に検出した色領域。approx: 四角形に近似した輪郭。
            # bbox: 外接矩形。3つを重ねることで、どの条件で拾われたか確認しやすくする。
            cv2.drawContours(frame, [contour], -1, color, 2)
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            label = base_label
            if distance_m is not None:
                source = detection.get("distance_source") or "unknown"
                label = f"{label} {distance_m:.2f}m {source}"

            # 文字が画像上端からはみ出さないよう、最低でもy=20に置く。
            text_y = max(y - 8, 20)
            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    def publish_debug_image(self, frame, source_msg):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warn(f"failed to convert debug image: {error}")
            return

        # 元画像のheaderを引き継ぎ、時刻とframe_idをデバッグ画像にも残す。
        debug_msg.header = source_msg.header
        self.debug_pub.publish(debug_msg)

    def log_detection(self, key, message, detections):
        # カメラ画像の周期で毎回ログを出すと読みにくいため、色ごとに出力間隔を制限する。
        now = self.get_clock().now()
        last_time = self.last_log_time[key]

        if last_time is not None:
            elapsed = (now - last_time).nanoseconds / 1e9
            if elapsed < self.log_interval_sec:
                return

        self.last_log_time[key] = now

        # 複数候補がある場合は、カメラに最も近い枠を代表値としてログに出す。
        nearest_detection = self.get_nearest_detection(detections)
        if nearest_detection is None:
            self.get_logger().info(f"{message} 距離: unknown")
        else:
            source = nearest_detection.get("distance_source") or "unknown"
            self.get_logger().info(
                f"{message} 距離: {nearest_detection['distance_m']:.3f} m "
                f"({source})"
            )

    def get_nearest_distance(self, detections):
        # 距離が計算できた候補だけを対象にする。
        # fx未取得やサイズ異常の候補は distance_m が無いので除外する。
        distances = [
            detection["distance_m"]
            for detection in detections
            if detection.get("distance_m") is not None
        ]
        if not distances:
            return None
        return min(distances)

    def get_nearest_detection(self, detections):
        valid_detections = [
            detection
            for detection in detections
            if detection.get("distance_m") is not None
        ]
        if not valid_detections:
            return None
        return min(valid_detections, key=lambda detection: detection["distance_m"])


def main(args=None):
    rclpy.init(args=args)
    node = FindCubeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
