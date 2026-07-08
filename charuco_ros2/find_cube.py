#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image


class FindCubeNode(Node):
    def __init__(self):
        super().__init__("find_cube_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/find_cube/debug_image")
        # 距離推定に使う、実物の色付き正方形フレームの一辺の長さ[m]。
        # 画像上の見かけの大きさ(pixel_size)とこの実寸から、カメラまでの距離を計算する。
        self.declare_parameter("frame_size_m", 0.05)

        # 輪郭検出後のフィルタ条件。
        # 小さいノイズ、細長い領域、正方形から外れた領域をここで落とす。
        self.declare_parameter("min_contour_area", 1500.0)
        self.declare_parameter("min_rectangularity", 0.45)
        self.declare_parameter("square_aspect_tolerance", 0.25)
        self.declare_parameter("polygon_epsilon_ratio", 0.04)
        self.declare_parameter("log_interval_sec", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.frame_size_m = self.get_parameter("frame_size_m").value
        self.min_contour_area = self.get_parameter("min_contour_area").value
        self.min_rectangularity = self.get_parameter("min_rectangularity").value
        self.square_aspect_tolerance = self.get_parameter(
            "square_aspect_tolerance"
        ).value
        self.polygon_epsilon_ratio = self.get_parameter(
            "polygon_epsilon_ratio"
        ).value
        self.log_interval_sec = self.get_parameter("log_interval_sec").value

        self.bridge = CvBridge()
        # CameraInfo が届くまでは焦点距離 fx が分からないため、距離は計算しない。
        self.fx = None

        # 赤・青それぞれで最後にログを出した時刻を保持し、ログ出力を間引く。
        self.last_log_time = {
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

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )

        self.get_logger().info("find_cube node started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"debug_image_topic: {self.debug_image_topic}")
        self.get_logger().info(f"frame_size_m: {self.frame_size_m}")

    def camera_info_callback(self, msg: CameraInfo):
        # カメラ内部パラメータ行列 K の (0, 0) が x方向の焦点距離 fx。
        # 距離推定では横方向・縦方向の平均pixel_sizeに対して、この fx を使う。
        self.fx = msg.k[0]

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warn(f"failed to convert image: {error}")
            return

        # BGRのまま色をしきい値処理すると明るさの影響を受けやすい。
        # HSVに変換して、色相(H)を中心に赤枠・青枠の領域を取り出す。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 赤はH=0付近とH=180付近にまたがるため専用関数でマスクを作る。
        red_mask = self.make_red_mask(hsv)
        # 青はHSVのH=95〜130付近を対象にする。
        # SとVの下限を設けて、白っぽい領域や暗いノイズを拾いにくくする。
        blue_mask = cv2.inRange(
            hsv,
            np.array([95, 80, 50], dtype=np.uint8),
            np.array([130, 255, 255], dtype=np.uint8),
        )

        # マスク画像から「正方形フレームらしい」候補だけを抽出する。
        # 赤は右、青は左として扱い、それぞれ同じ手順で検出する。
        red_detections = self.find_frames(red_mask)
        blue_detections = self.find_frames(blue_mask)
        # CameraInfo から fx が取得できていれば、各候補に distance_m を追加する。
        self.add_distances(red_detections)
        self.add_distances(blue_detections)

        if red_detections:
            self.log_detection("right", "赤い枠を検出: 右", red_detections)

        if blue_detections:
            self.log_detection("left", "青い枠を検出: 左", blue_detections)

        # デバッグ画像には、元の輪郭・近似した四角形・外接矩形・距離を重ねて描画する。
        debug_frame = frame.copy()
        self.draw_detections(
            debug_frame,
            red_detections,
            color=(0, 0, 255),
            label="red: right",
        )
        self.draw_detections(
            debug_frame,
            blue_detections,
            color=(255, 0, 0),
            label="blue: left",
        )
        self.publish_debug_image(debug_frame, msg)

    def make_red_mask(self, hsv):
        # OpenCVのHSVでは赤がH=0付近とH=180付近に分かれる。
        # 片側だけを見ると赤い物体の一部を取り逃すため、2つの範囲をORで結合する。
        lower_red_1 = np.array([0, 80, 50], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 80, 50], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        return cv2.bitwise_or(mask_1, mask_2)

    def find_frames(self, mask):
        # OPENで孤立した小さな白ノイズを消し、CLOSEで色領域内の小さな穴を埋める。
        # これにより、後段の輪郭検出でフレームが分断されにくくなる。
        kernel = np.ones((5, 5), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

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
                continue

            # 外接矩形は、面積比によるざっくりした形状チェックと描画に使う。
            x, y, width, height = cv2.boundingRect(contour)
            bounding_area = width * height
            if bounding_area == 0:
                continue

            # rectangularity は「外接矩形に対して輪郭がどれだけ詰まっているか」。
            # 細い線、円弧、欠けた領域は値が低くなるのでここで除外する。
            rectangularity = area / bounding_area
            if rectangularity < self.min_rectangularity:
                continue

            # 輪郭点をそのまま見ると細かすぎるため、多角形近似で頂点数を減らす。
            # 近似後に「凸な四角形か」「正方形に近いか」を判定する。
            epsilon = self.polygon_epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if not self.is_square_like(approx):
                continue

            # 後段の描画・距離計算で必要な情報をまとめて保持する。
            detections.append(
                {
                    "contour": contour,
                    "approx": approx,
                    "bbox": (x, y, width, height),
                    "pixel_size": self.get_square_pixel_size(approx),
                    "area": area,
                }
            )

        return detections

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

        if self.fx is None:
            # CameraInfoを受け取る前は焦点距離がないため、検出だけ行い距離は未設定にする。
            self.get_logger().warn(
                "Waiting for CameraInfo. Distance is not available.",
                throttle_duration_sec=2.0,
            )
            return

        for detection in detections:
            pixel_size = detection["pixel_size"]
            if pixel_size is None or pixel_size <= 0:
                # サイズが取れない候補は距離計算できないので、明示的にNoneを入れる。
                detection["distance_m"] = None
                continue

            # ピンホールカメラモデルを使う。
            # 画像上の大きさ[pixel] = fx[pixel] * 実寸[m] / 距離[m]
            # これを変形して、距離[m] = fx * 実寸 / 画像上の大きさ として計算する。
            detection["distance_m"] = self.fx * self.frame_size_m / pixel_size

    def is_square_like(self, approx):
        # 検出対象は正方形フレームなので、まず4頂点の凸四角形だけを残す。
        # その後、回転外接矩形の縦横比が1に近いかを確認する。
        if len(approx) != 4:
            return False

        if not cv2.isContourConvex(approx):
            return False

        rect = cv2.minAreaRect(approx)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False

        # 完全な正方形だけにすると傾きや検出誤差で落ちやすい。
        # square_aspect_tolerance の分だけ縦横比のずれを許容する。
        aspect_ratio = max(width, height) / min(width, height)
        max_aspect_ratio = 1.0 + self.square_aspect_tolerance
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
                label = f"{label} {distance_m:.2f}m"

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
        nearest_distance = self.get_nearest_distance(detections)
        if nearest_distance is None:
            self.get_logger().info(f"{message} 距離: unknown")
        else:
            self.get_logger().info(f"{message} 距離: {nearest_distance:.3f} m")

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
