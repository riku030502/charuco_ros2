#!/usr/bin/env python3
import math
import os
import threading
import time

import numpy as np
import yaml
import rclpy
from geometry_msgs.msg import PoseStamped

try:
    from moveit_msgs.msg import MoveItErrorCodes
    from moveit_msgs.srv import GetPositionIK
except ImportError:
    MoveItErrorCodes = None
    GetPositionIK = None

try:
    from xarm_utils_py import XArmUtils
    from xarm_utils_py import Node as XArmNode
except ImportError:
    XArmUtils = None
    XArmNode = None

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import SetBool, Trigger
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from xarm_msgs.srv import MoveJoint


RIGHT_JOINT_DEGREES = [-53.0, 55.0, -110.0, 0.0, -52.0, -8.0]
LEFT_JOINT_DEGREES = [53.0, 55.0, -110.0, 0.0, -62.0, -8.0]
DEFAULT_JOINT_DEGREES = LEFT_JOINT_DEGREES
PRE_MARKER_JOINT_DEGREES = [0.0, -15.0, 0.0, 0.0, -90.0, 0.0]
DEFAULT_SPEED = 0.035  # 0.1x of the xArm README example 0.35 rad/s
DEFAULT_ACC = 1.0  # 0.1x of the xArm README example 10 rad/s^2
JOINT1_LIMIT_RAD = 2.0 * math.pi


class MoveToCharucoPoseNode(Node):
    def __init__(self):
        super().__init__("move_to_charuco_pose")

        self.declare_parameter("server_service_name", "/move_to_charuco")
        self.declare_parameter("execution_mode", "real")
        self.declare_parameter("backend", "moveit")
        self.declare_parameter("move_group_name", "xarm6")
        self.declare_parameter("moveit_planning_pipeline", "")
        self.declare_parameter("move_group_ready_timeout", 10.0)
        self.declare_parameter("xarm_service_name", "auto")
        self.declare_parameter("controller_name", "xarm6_traj_controller")
        self.declare_parameter("joint_prefix", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("joint_state_sync_timeout", 3.0)
        self.declare_parameter("joint_state_tolerance", 0.02)
        self.declare_parameter("publish_warmup_time", 1.0)
        self.declare_parameter("input_unit", "deg")
        self.declare_parameter("default_speed", DEFAULT_SPEED)
        self.declare_parameter("default_acc", DEFAULT_ACC)
        self.declare_parameter("default_move_duration", 10.0)
        self.declare_parameter("default_wait", True)
        self.declare_parameter("default_timeout", 60.0)
        self.declare_parameter("default_radius", -1.0)
        self.declare_parameter("camera_side", "left")
        self.declare_parameter("left_joint_degrees", LEFT_JOINT_DEGREES)
        self.declare_parameter("right_joint_degrees", RIGHT_JOINT_DEGREES)
        self.declare_parameter("prepare_before_marker_move", True)
        self.declare_parameter(
            "pre_marker_joint_degrees",
            PRE_MARKER_JOINT_DEGREES,
        )
        self.declare_parameter("pre_marker_detection_window_sec", 0.0)
        self.declare_parameter("manage_find_cube_detection", True)
        self.declare_parameter(
            "find_cube_detection_service_name",
            "/find_cube/set_detection_enabled",
        )
        self.declare_parameter("find_cube_detection_service_timeout", 2.0)
        self.declare_parameter("restore_find_cube_detection_after_move", False)
        self.declare_parameter("detect_multi_cube_after_approach", True)
        self.declare_parameter(
            "multi_cube_detect_service_name",
            "/multi_cube_charuco_detector/detect_once",
        )
        self.declare_parameter("multi_cube_detect_service_timeout", 8.0)
        self.declare_parameter("multi_cube_detect_required", True)
        self.declare_parameter("return_to_pre_marker_after_success", True)
        self.declare_parameter("cube_base_frame", "link_base")
        self.declare_parameter(
            "left_cube_tf_frame",
            "left_cube_color_link_base_frame",
        )
        self.declare_parameter(
            "right_cube_tf_frame",
            "right_cube_color_link_base_frame",
        )
        self.declare_parameter(
            "cube_pose_cache_file",
            "charuco_ros2/config/cube_target_poses.yaml",
        )
        self.declare_parameter("use_cached_cube_pose", True)
        self.declare_parameter("cube_lookup_timeout", 0.5)
        # xArm6の可動範囲端を避けるため、キューブから30 cm手前を使用する。
        self.declare_parameter("cube_approach_offset_m", 0.30)
        # radial: link_base->cube方向へ後退する。cubeが横にあれば横から接近する。
        # x_axis: cubeのY/Zに合わせ、X方向にだけ後退する（従来動作）。
        self.declare_parameter("cube_approach_mode", "radial")
        # link_baseから手先目標までの最大距離。
        # xArm6の姿勢制約を考慮し、公称最大リーチより内側に制限する。
        self.declare_parameter("cartesian_max_target_distance_m", 1.5)
        # 手先目標の横位置Yをcube TFへ合わせる。
        # 高さZはcube TFを基準に相対調整する。
        # デフォルトはcubeより5 cm下。
        self.declare_parameter("cube_target_z_offset_m", -0.05)
        self.declare_parameter("cartesian_service_name", "auto")
        self.declare_parameter("cartesian_speed", 50.0)
        self.declare_parameter("cartesian_acc", 500.0)
        self.declare_parameter("cartesian_wait", False)
        self.declare_parameter("cartesian_timeout", 60.0)
        self.declare_parameter("cartesian_radius", -1.0)
        self.declare_parameter("cartesian_rpy_rad", [3.14, 0.0, 0.0])
        # 手先前方軸（ローカルZ軸）まわりの追加回転。
        # カメラ方向は下のcartesian_camera_up_directionで直接決めるため、
        # 通常は0 radのまま使用する。
        self.declare_parameter("cartesian_roll_rad", 0.0)
        # point_at_cube: 手先前方をキューブへ向ける（キューブに正対する）
        # front_fixed: 手先前方をlink_baseの指定方向へ固定
        # fixed: cartesian_rpy_radをそのまま使用
        self.declare_parameter("cartesian_orientation_mode", "point_at_cube")
        self.declare_parameter(
            "cartesian_front_direction",
            [1.0, 0.0, 0.0],
        )
        # ハンドカメラが向く方向。
        # link_eefのローカルX軸をカメラ方向として扱い、
        # link_baseの+Z（上方向）へ向ける。
        self.declare_parameter(
            "cartesian_camera_up_direction",
            [0.0, 0.0, 1.0],
        )
        # point_at_cubeで手先前方をキューブへ向けるとき、上下の傾きを捨てて
        # 水平に保つ。cube_target_z_offset_mを付けると手先はキューブより下に
        # 来るため、そのまま向けると手先が上を向く。高さオフセットは維持した
        # まま、向きだけ水平にしたい場合にtrueにする。
        self.declare_parameter("cartesian_level_tool_forward", True)
        self.declare_parameter("cartesian_point_axis_sign", 1.0)
        self.declare_parameter("cartesian_motion_type", 0)
        self.declare_parameter("prepare_xarm_before_cartesian", True)
        self.declare_parameter("cartesian_mode", 0)
        self.declare_parameter("cartesian_state", 0)
        # 後方互換のため残しているが、現在はreal/simともMoveIt IK -> MoveGroup
        # plan/executeに固定する。xArmのset_positionへはフォールバックしない。
        self.declare_parameter("real_use_moveit_ik", True)
        self.declare_parameter("sim_ik_service_name", "/compute_ik")
        self.declare_parameter("sim_ik_group_name", "xarm6")
        self.declare_parameter("sim_ik_link_name", "link_eef")
        self.declare_parameter("sim_ik_timeout", 2.0)
        self.declare_parameter("sim_avoid_collisions", True)
        self.declare_parameter("sim_cartesian_move_duration", 5.0)
        # KDLはシード近傍の解へ収束する反復ソルバなので、シードが現在姿勢から離れているほど、現在姿勢から遠い分岐（同じ手先ポーズを実現する別の関節解）が返り、移動量が無駄に大きくなる。
        # current: 現在の関節角をシードにする。移動量が最小の分岐に落ちやすい。
        # side_pose: left/rightの決め打ち定数。現在姿勢と無関係なので非推奨。
        # current_or_side: 現在角が取れなければside_poseへフォールバック。
        self.declare_parameter("sim_ik_seed_mode", "current_or_side")
        # 先にjoint1だけを目標方位へ回してからIKを解く。
        # ただしIKは同じ手先ポーズを実現する別分岐（腕を真逆へ向けて後ろへ反り返る解など）を返すことがあり、その場合joint1は整列させた角度から大きく戻されるため、事前整列はかえって総移動量を増やす。
        # 目標が遠くて正面向きの分岐が存在しない配置では特に起きるので、既定はfalse。
        self.declare_parameter("align_joint1_before_move", False)
        self.declare_parameter("joint1_align_duration", 3.0)
        self.declare_parameter("joint1_align_tolerance_deg", 2.0)

        self.server_service_name = self.get_parameter("server_service_name").value
        self.execution_mode = self.get_parameter("execution_mode").value.lower()
        self.backend = self.get_parameter("backend").value.lower()
        self.move_group_name = self.get_parameter("move_group_name").value
        self.moveit_planning_pipeline = self.get_parameter(
            "moveit_planning_pipeline"
        ).value
        self.move_group_ready_timeout = float(
            self.get_parameter("move_group_ready_timeout").value
        )
        self.xarm_service_name = self.get_parameter("xarm_service_name").value
        self.controller_name = self.get_parameter("controller_name").value
        self.joint_prefix = self.get_parameter("joint_prefix").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.joint_state_sync_timeout = float(
            self.get_parameter("joint_state_sync_timeout").value
        )
        self.joint_state_tolerance = float(
            self.get_parameter("joint_state_tolerance").value
        )
        self.publish_warmup_time = float(
            self.get_parameter("publish_warmup_time").value
        )
        self.input_unit = self.get_parameter("input_unit").value.lower()
        self.default_speed = float(self.get_parameter("default_speed").value)
        self.default_acc = float(self.get_parameter("default_acc").value)
        self.default_move_duration = float(
            self.get_parameter("default_move_duration").value
        )
        self.default_wait = bool(self.get_parameter("default_wait").value)
        self.default_timeout = float(self.get_parameter("default_timeout").value)
        self.default_radius = float(self.get_parameter("default_radius").value)
        self.camera_side = self.get_parameter("camera_side").value.lower()
        self.left_joint_degrees = list(self.get_parameter("left_joint_degrees").value)
        self.right_joint_degrees = list(self.get_parameter("right_joint_degrees").value)
        self.prepare_before_marker_move = bool(
            self.get_parameter("prepare_before_marker_move").value
        )
        self.pre_marker_joint_degrees = list(
            self.get_parameter("pre_marker_joint_degrees").value
        )
        self.pre_marker_detection_window_sec = float(
            self.get_parameter("pre_marker_detection_window_sec").value
        )
        self.manage_find_cube_detection = bool(
            self.get_parameter("manage_find_cube_detection").value
        )
        self.find_cube_detection_service_name = self.get_parameter(
            "find_cube_detection_service_name"
        ).value
        self.find_cube_detection_service_timeout = float(
            self.get_parameter("find_cube_detection_service_timeout").value
        )
        self.restore_find_cube_detection_after_move = bool(
            self.get_parameter("restore_find_cube_detection_after_move").value
        )
        self.detect_multi_cube_after_approach = bool(
            self.get_parameter("detect_multi_cube_after_approach").value
        )
        self.multi_cube_detect_service_name = self.get_parameter(
            "multi_cube_detect_service_name"
        ).value
        self.multi_cube_detect_service_timeout = float(
            self.get_parameter("multi_cube_detect_service_timeout").value
        )
        self.multi_cube_detect_required = bool(
            self.get_parameter("multi_cube_detect_required").value
        )
        self.return_to_pre_marker_after_success = bool(
            self.get_parameter("return_to_pre_marker_after_success").value
        )
        self.cube_base_frame = self.get_parameter("cube_base_frame").value
        self.left_cube_tf_frame = self.get_parameter("left_cube_tf_frame").value
        self.right_cube_tf_frame = self.get_parameter("right_cube_tf_frame").value
        self.cube_pose_cache_file = self.resolve_cache_file(
            self.get_parameter("cube_pose_cache_file").value
        )
        self.use_cached_cube_pose = bool(
            self.get_parameter("use_cached_cube_pose").value
        )
        self.cube_lookup_timeout = float(
            self.get_parameter("cube_lookup_timeout").value
        )
        self.cube_approach_offset_m = float(
            self.get_parameter("cube_approach_offset_m").value
        )
        self.cube_approach_mode = self.get_parameter(
            "cube_approach_mode"
        ).value.lower()
        self.cartesian_max_target_distance_m = float(
            self.get_parameter(
                "cartesian_max_target_distance_m"
            ).value
        )
        self.cube_target_z_offset_m = float(
            self.get_parameter("cube_target_z_offset_m").value
        )
        self.cartesian_service_name = self.get_parameter(
            "cartesian_service_name"
        ).value
        self.cartesian_speed = float(self.get_parameter("cartesian_speed").value)
        self.cartesian_acc = float(self.get_parameter("cartesian_acc").value)
        self.cartesian_wait = bool(self.get_parameter("cartesian_wait").value)
        self.cartesian_timeout = float(
            self.get_parameter("cartesian_timeout").value
        )
        self.cartesian_radius = float(self.get_parameter("cartesian_radius").value)
        self.cartesian_rpy_rad = list(
            self.get_parameter("cartesian_rpy_rad").value
        )
        self.cartesian_roll_rad = float(
            self.get_parameter("cartesian_roll_rad").value
        )
        self.cartesian_orientation_mode = self.get_parameter(
            "cartesian_orientation_mode"
        ).value
        self.cartesian_front_direction = list(
            self.get_parameter("cartesian_front_direction").value
        )
        self.cartesian_camera_up_direction = list(
            self.get_parameter(
                "cartesian_camera_up_direction"
            ).value
        )
        self.cartesian_level_tool_forward = bool(
            self.get_parameter("cartesian_level_tool_forward").value
        )
        self.cartesian_point_axis_sign = float(
            self.get_parameter("cartesian_point_axis_sign").value
        )
        self.cartesian_motion_type = int(
            self.get_parameter("cartesian_motion_type").value
        )
        self.prepare_xarm_before_cartesian = bool(
            self.get_parameter("prepare_xarm_before_cartesian").value
        )
        self.cartesian_mode = int(self.get_parameter("cartesian_mode").value)
        self.cartesian_state = int(self.get_parameter("cartesian_state").value)
        self.real_use_moveit_ik = bool(
            self.get_parameter("real_use_moveit_ik").value
        )
        self.sim_ik_service_name = self.get_parameter(
            "sim_ik_service_name"
        ).value
        self.sim_ik_group_name = self.get_parameter(
            "sim_ik_group_name"
        ).value
        self.sim_ik_link_name = self.get_parameter(
            "sim_ik_link_name"
        ).value
        self.sim_ik_timeout = float(
            self.get_parameter("sim_ik_timeout").value
        )
        self.sim_avoid_collisions = bool(
            self.get_parameter("sim_avoid_collisions").value
        )
        self.sim_cartesian_move_duration = float(
            self.get_parameter("sim_cartesian_move_duration").value
        )
        self.sim_ik_seed_mode = self.get_parameter(
            "sim_ik_seed_mode"
        ).value.lower()
        self.align_joint1_before_move = bool(
            self.get_parameter("align_joint1_before_move").value
        )
        self.joint1_align_duration = float(
            self.get_parameter("joint1_align_duration").value
        )
        self.joint1_align_tolerance_rad = math.radians(
            float(self.get_parameter("joint1_align_tolerance_deg").value)
        )

        if self.camera_side == "left":
            self.side_joint_degrees = self.left_joint_degrees
        elif self.camera_side == "right":
            self.side_joint_degrees = self.right_joint_degrees
        else:
            raise ValueError("camera_side must be 'left' or 'right'")

        if self.execution_mode not in ("real", "sim"):
            raise ValueError("execution_mode must be 'real' or 'sim'")

        valid_backends = (
            "auto",
            "moveit",
            "xarm_api",
            "trajectory",
            "topic",
        )
        if self.backend not in valid_backends:
            raise ValueError(
                "backend must be 'auto', 'moveit', 'xarm_api', 'trajectory', "
                "or 'topic'"
            )
        if self.input_unit not in ("deg", "rad"):
            raise ValueError("input_unit must be 'deg' or 'rad'")
        if len(self.pre_marker_joint_degrees) != 6:
            raise ValueError("pre_marker_joint_degrees must contain 6 values")
        if self.pre_marker_detection_window_sec < 0.0:
            raise ValueError("pre_marker_detection_window_sec must be >= 0")
        if self.cartesian_max_target_distance_m <= 0.0:
            raise ValueError(
                "cartesian_max_target_distance_m must be positive"
            )
        if self.cube_approach_mode not in ("radial", "x_axis"):
            raise ValueError("cube_approach_mode must be 'radial' or 'x_axis'")
        if len(self.cartesian_rpy_rad) != 3:
            raise ValueError("cartesian_rpy_rad must contain roll, pitch, yaw")
        if self.cartesian_orientation_mode not in (
            "fixed",
            "point_at_cube",
            "front_fixed",
        ):
            raise ValueError(
                "cartesian_orientation_mode must be 'fixed', "
                "'point_at_cube', or 'front_fixed'"
            )
        if len(self.cartesian_front_direction) != 3:
            raise ValueError(
                "cartesian_front_direction must contain x, y, z"
            )
        if len(self.cartesian_camera_up_direction) != 3:
            raise ValueError(
                "cartesian_camera_up_direction must contain x, y, z"
            )
        if self.sim_ik_seed_mode not in (
            "side_pose",
            "current",
            "current_or_side",
        ):
            raise ValueError(
                "sim_ik_seed_mode must be 'side_pose', 'current', "
                "or 'current_or_side'"
            )

        self.callback_group = ReentrantCallbackGroup()
        self.xarm_node = None
        self.xarm = None
        self.sim_ik_client = None
        self.find_cube_detection_client = None
        self.multi_cube_detect_client = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joint_names = [
            f"{self.joint_prefix}joint{i}" for i in range(1, 7)
        ]
        self.latest_joint_state = None
        self.latest_joint_state_event = threading.Event()
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.handle_joint_state,
            10,
            callback_group=self.callback_group,
        )
        self.server = self.create_service(
            MoveJoint,
            self.server_service_name,
            self.handle_move_request,
            callback_group=self.callback_group,
        )
        self.left_server = self.create_service(
            MoveJoint,
            self.server_service_name + "/left",
            self.handle_left_move_request,
            callback_group=self.callback_group,
        )
        self.right_server = self.create_service(
            MoveJoint,
            self.server_service_name + "/right",
            self.handle_right_move_request,
            callback_group=self.callback_group,
        )
        self.cube_server = self.create_service(
            Trigger,
            self.server_service_name + "/cube",
            self.handle_cube_move_request,
            callback_group=self.callback_group,
        )
        self.left_cube_server = self.create_service(
            Trigger,
            self.server_service_name + "/cube_left",
            self.handle_left_cube_move_request,
            callback_group=self.callback_group,
        )
        self.right_cube_server = self.create_service(
            Trigger,
            self.server_service_name + "/cube_right",
            self.handle_right_cube_move_request,
            callback_group=self.callback_group,
        )
        self.pre_marker_server = self.create_service(
            Trigger,
            self.server_service_name + "/pre_marker",
            self.handle_pre_marker_move_request,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"Services ready: {self.server_service_name} | "
            f"{self.server_service_name}/left | {self.server_service_name}/right "
            f"(angles unit: {self.input_unit})"
        )
        self.get_logger().info(
            "Pre-marker return service ready: "
            f"{self.server_service_name}/pre_marker"
        )
        self.get_logger().info(
            f"execution_mode={self.execution_mode}, "
            f"backend={self.backend} "
            f"(effective={self.get_effective_joint_backend()}), "
            f"move_group={self.move_group_name}, "
            f"planning_pipeline={self.moveit_planning_pipeline or 'default'}"
        )
        self.get_logger().info(
            f"left pose:  {self.left_joint_degrees} deg\n"
            f"right pose: {self.right_joint_degrees} deg"
        )
        self.get_logger().info(
            "pre-marker move: "
            f"enabled={self.prepare_before_marker_move}, "
            f"return_after_success={self.return_to_pre_marker_after_success}, "
            f"detection_window={self.pre_marker_detection_window_sec:.1f}s, "
            f"pose={self.pre_marker_joint_degrees} deg"
        )
        self.get_logger().info(
            "find_cube detection control: "
            f"manage={self.manage_find_cube_detection}, "
            f"service={self.find_cube_detection_service_name}, "
            f"timeout={self.find_cube_detection_service_timeout:.1f}s, "
            f"restore_after_move={self.restore_find_cube_detection_after_move}"
        )
        self.get_logger().info(
            "multi-cube detection after approach: "
            f"enabled={self.detect_multi_cube_after_approach}, "
            f"required={self.multi_cube_detect_required}, "
            f"service={self.multi_cube_detect_service_name}, "
            f"timeout={self.multi_cube_detect_service_timeout:.1f}s"
        )
        self.get_logger().info(
            "Cartesian orientation: "
            f"mode={self.cartesian_orientation_mode}, "
            f"tool_axis_roll={math.degrees(self.cartesian_roll_rad):.1f} deg, "
            f"front_direction={self.cartesian_front_direction}, "
            f"camera_direction={self.cartesian_camera_up_direction}, "
            f"level_tool_forward={self.cartesian_level_tool_forward} "
            "(real and simulation)"
        )
        self.get_logger().info(
            f"IK seed mode: {self.sim_ik_seed_mode}"
        )
        if self.backend in ("topic", "trajectory", "xarm_api"):
            self.get_logger().warn(
                f"backend:={self.backend} is deprecated and ignored for "
                "execution. move_to_charuco now always uses MoveIt "
                "MoveGroup plan -> execute."
            )
        if self.execution_mode == "real" and not self.real_use_moveit_ik:
            self.get_logger().warn(
                "real_use_moveit_ik:=false is deprecated and ignored. "
                "Cartesian moves still use MoveIt IK followed by MoveGroup "
                "plan -> execute; xArm set_position fallback is disabled."
            )
        self.get_logger().info(
            "joint1 alignment before Cartesian moves: "
            f"{self.align_joint1_before_move} "
            f"(duration={self.joint1_align_duration:.1f}s, "
            "tolerance="
            f"{math.degrees(self.joint1_align_tolerance_rad):.1f} deg)"
        )
        self.get_logger().info(
            f"Cube TF move services ready: {self.server_service_name}/cube | "
            f"{self.server_service_name}/cube_left | "
            f"{self.server_service_name}/cube_right"
        )
        self.get_logger().info(
            f"cube_base_frame={self.cube_base_frame}, "
            f"cache={self.cube_pose_cache_file}, "
            f"approach_mode={self.cube_approach_mode}, "
            f"approach_offset={self.cube_approach_offset_m:.3f} m, "
            f"max_target_distance="
            f"{self.cartesian_max_target_distance_m:.3f} m, "
            f"cube_height_offset={self.cube_target_z_offset_m:.3f} m"
        )

    def resolve_cache_file(self, cache_file):
        if os.path.isabs(cache_file):
            return os.path.realpath(os.path.expanduser(cache_file))

        package_relative_prefix = "charuco_ros2/"
        if cache_file.startswith(package_relative_prefix):
            cache_file = cache_file[len(package_relative_prefix):]

        return os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", cache_file)
        )

    def get_effective_joint_backend(self):
        """Resolve the actual joint command path."""
        return "moveit"

    def get_xarm_utils(self):
        if XArmUtils is None or XArmNode is None:
            return (
                None,
                "xarm_utils_py is not available. Build/source xarm_utils_cpp "
                "before using move_to_charuco.",
            )

        if self.xarm is None:
            if not self.wait_for_move_group_parameter_service():
                return (
                    None,
                    "MoveIt move_group parameter service is not available: "
                    "/move_group/get_parameters. Start/source the xArm MoveIt "
                    "launch before calling move_to_charuco.",
                )
            try:
                self.xarm_node = XArmNode("move_to_charuco_xarm_utils")
                self.xarm = XArmUtils(self.xarm_node, self.move_group_name)
                if self.moveit_planning_pipeline:
                    self.xarm.set_planning_pipeline(
                        self.moveit_planning_pipeline
                    )
                self.get_logger().info(
                    "MoveGroup interface ready via xarm_utils_py: "
                    f"group={self.move_group_name}, "
                    f"pipeline={self.moveit_planning_pipeline or 'default'}"
                )
            except Exception as error:
                self.xarm = None
                return None, f"failed to initialize xarm_utils_py: {error}"

        return self.xarm, "success"

    def wait_for_move_group_parameter_service(self):
        deadline = time.monotonic() + max(self.move_group_ready_timeout, 0.0)
        while time.monotonic() <= deadline:
            for service_name, _ in self.get_service_names_and_types():
                if service_name == "/move_group/get_parameters":
                    return True
            time.sleep(0.1)
        return False

    def handle_joint_state(self, msg):
        self.latest_joint_state = msg
        self.latest_joint_state_event.set()

    def wait_for_joint_state_target(self, target_positions):
        if self.joint_state_sync_timeout <= 0.0:
            return True

        deadline = self.get_clock().now() + Duration(
            seconds=self.joint_state_sync_timeout
        )
        while self.get_clock().now() < deadline:
            msg = self.latest_joint_state
            if msg is not None:
                positions = dict(zip(msg.name, msg.position))
                if all(name in positions for name in self.joint_names):
                    errors = [
                        abs(positions[name] - target)
                        for name, target in zip(self.joint_names, target_positions)
                    ]
                    if max(errors) <= self.joint_state_tolerance:
                        return True

            self.latest_joint_state_event.clear()
            remaining = (deadline - self.get_clock().now()).nanoseconds / 1e9
            self.latest_joint_state_event.wait(timeout=min(max(remaining, 0.0), 0.1))

        return False

    def get_current_joint_positions(self):
        msg = self.latest_joint_state
        if msg is None:
            return None

        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in self.joint_names):
            return None

        return [positions[name] for name in self.joint_names]

    def move_joints_with_moveit(self, target_positions, label="joint target"):
        xarm, message = self.get_xarm_utils()
        if xarm is None:
            return False, -20, message

        target_positions = [float(value) for value in target_positions]
        self.get_logger().info(
            "Planning MoveGroup motion: "
            f"{label}, target_deg="
            f"{[round(math.degrees(value), 1) for value in target_positions]}"
        )

        try:
            target_ok = xarm.set_joint_value_target(target_positions)
        except Exception as error:
            return (
                False,
                -21,
                f"MoveGroup set_joint_value_target failed: {error}",
            )
        if not target_ok:
            return False, -21, "MoveGroup rejected the joint target"

        try:
            plan_success, _, plan_duration, error_code = xarm.plan()
        except Exception as error:
            return (
                False,
                -22,
                f"MoveGroup planning threw an exception: {error}",
            )

        error_value = getattr(error_code, "val", 0)
        if not plan_success:
            ret = error_value if error_value not in (0, None) else -22
            return (
                False,
                ret,
                "MoveGroup planning failed: "
                f"error_code={error_value}, duration={plan_duration:.3f}s",
            )

        self.get_logger().info(
            f"MoveGroup plan succeeded in {plan_duration:.3f}s; executing"
        )

        try:
            execute_success = xarm.execute()
        except Exception as error:
            return False, -23, f"MoveGroup execute threw an exception: {error}"
        if not execute_success:
            return False, -23, "MoveGroup execute failed"

        if not self.wait_for_joint_state_target(target_positions):
            return (
                False,
                -24,
                "/joint_states did not reach the MoveGroup target before "
                "joint_state_sync_timeout",
            )

        return True, 0, "success"

    def get_pre_marker_positions_rad(self):
        return [
            math.radians(float(value))
            for value in self.pre_marker_joint_degrees
        ]

    def move_to_pre_marker_pose(self, force=False):
        if not force and not self.prepare_before_marker_move:
            return True, 0, "pre-marker move disabled"

        self.get_logger().info(
            "Moving to pre-marker pose"
        )
        return self.move_joints_with_moveit(
            self.get_pre_marker_positions_rad(),
            "pre-marker pose",
        )

    def return_to_pre_marker_pose_after_success(self):
        if not self.return_to_pre_marker_after_success:
            return True, 0, "return-to-pre-marker disabled"

        self.get_logger().info(
            "Returning to pre-marker pose after the requested target succeeded"
        )
        return self.move_joints_with_moveit(
            self.get_pre_marker_positions_rad(),
            "return to pre-marker pose",
        )

    def wait_for_detection_after_pre_marker(self):
        if self.pre_marker_detection_window_sec <= 0.0:
            return

        success, message = self.set_find_cube_detection_enabled(
            True,
            required=True,
        )
        if not success:
            raise RuntimeError(message)

        self.get_logger().info(
            "Waiting for cube detection after pre-marker move: "
            f"{self.pre_marker_detection_window_sec:.1f}s"
        )
        try:
            time.sleep(self.pre_marker_detection_window_sec)
        finally:
            self.set_find_cube_detection_enabled(False, required=False)

    @staticmethod
    def wait_for_future(future, timeout):
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        return done_event.wait(timeout=timeout)

    def handle_left_move_request(self, request, response):
        return self._handle_move_request(request, response, self.left_joint_degrees)

    def handle_right_move_request(self, request, response):
        return self._handle_move_request(request, response, self.right_joint_degrees)

    def handle_move_request(self, request, response):
        return self._handle_move_request(request, response, self.side_joint_degrees)

    def handle_cube_move_request(self, request, response):
        del request
        return self._handle_cube_move_request("latest", response)

    def handle_left_cube_move_request(self, request, response):
        del request
        return self._handle_cube_move_request("left", response)

    def handle_right_cube_move_request(self, request, response):
        del request
        return self._handle_cube_move_request("right", response)

    def handle_pre_marker_move_request(self, request, response):
        """Return the arm to the shared pre-marker joint pose."""
        del request
        self.set_find_cube_detection_enabled(False, required=False)
        success, ret, message = self.move_to_pre_marker_pose(force=True)
        if success:
            return self.set_trigger_response(
                response,
                True,
                "returned to pre-marker pose",
            )
        return self.set_trigger_response(
            response,
            False,
            f"return to pre-marker pose failed: ret={ret}, {message}",
        )

    def _handle_cube_move_request(self, side, response):
        self.set_find_cube_detection_enabled(False, required=False)

        success, ret, message = self.move_to_pre_marker_pose()
        if not success:
            return self.set_trigger_response(
                response,
                False,
                f"pre-marker move failed: ret={ret}, {message}",
            )

        try:
            self.wait_for_detection_after_pre_marker()
        except RuntimeError as e:
            return self.set_trigger_response(response, False, str(e))

        try:
            cube_pose, pose_source = self.get_cube_pose_for_move(side)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return self.set_trigger_response(
                response,
                False,
                f"{side} cube pose is not available: {e}",
            )

        try:
            approach_position_m = self.compute_approach_position(cube_pose)
        except ValueError as e:
            return self.set_trigger_response(response, False, str(e))

        cartesian_pose = self.create_cartesian_pose(
            approach_position_m,
            cube_pose,
        )

        self.save_cube_pose(side, cube_pose, approach_position_m, cartesian_pose)

        success, message = self.move_with_cartesian_pose(
            cartesian_pose,
            side,
        )
        if not success:
            return self.set_trigger_response(response, False, message)

        detect_success, detect_message = self.call_multi_cube_detect_once()

        success, ret, message = self.return_to_pre_marker_pose_after_success()
        if not success:
            return self.set_trigger_response(
                response,
                False,
                f"return to pre-marker pose failed: ret={ret}, {message}",
            )

        if not detect_success:
            return self.set_trigger_response(
                response,
                False,
                (
                    f"multi-cube detection failed after approach: "
                    f"{detect_message}; returned to pre-marker pose"
                ),
            )

        if self.restore_find_cube_detection_after_move:
            self.set_find_cube_detection_enabled(True, required=False)

        return self.set_trigger_response(
            response,
            True,
            (
                f"moved to {side} cube approach pose from {pose_source}: "
                f"pose_mm_rad={cartesian_pose}; "
                f"multi_cube_detection={detect_message}; "
                f"returned to pre-marker pose"
            ),
        )

    def set_trigger_response(self, response, success, message):
        response.success = success
        response.message = message
        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)
        return response

    def cube_tf_frame_for_side(self, side):
        if side == "left":
            return self.left_cube_tf_frame
        if side == "right":
            return self.right_cube_tf_frame
        raise ValueError("side must be 'left' or 'right'")

    def get_cube_pose_for_move(self, side):
        if self.use_cached_cube_pose:
            cube_pose = self.load_cached_cube_pose(side)
            if cube_pose is not None:
                return cube_pose, "cache"

        lookup_side = self.camera_side if side == "latest" else side
        return self.lookup_cube_pose(lookup_side), "tf"

    def lookup_cube_pose(self, side):
        cube_frame = self.cube_tf_frame_for_side(side)
        transform = self.tf_buffer.lookup_transform(
            self.cube_base_frame,
            cube_frame,
            Time(),
            timeout=Duration(seconds=self.cube_lookup_timeout),
        )
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

    def compute_approach_position(self, cube_pose):
        if self.cube_approach_mode == "radial":
            return self.compute_radial_approach_position(cube_pose)
        return self.compute_x_axis_approach_position(cube_pose)

    def compute_radial_approach_position(self, cube_pose):
        """Stand off along the link_base->cube direction.

        キューブが横にあれば後退方向も横向きになるため、手先はキューブへ
        正対したまま真横から接近する。cubeが正面(+X)にある場合は
        x_axisモードと同じ結果になる。
        """
        translation = cube_pose["translation_m"]
        cube_x = float(translation["x"])
        cube_y = float(translation["y"])
        cube_z = float(translation["z"])

        # 高さオフセットは従来どおりZへ直接加え、後退は水平方向のみで行う。
        look_at = np.array(
            [cube_x, cube_y, cube_z + self.cube_target_z_offset_m],
            dtype=np.float64,
        )

        horizontal = np.array([cube_x, cube_y, 0.0], dtype=np.float64)
        horizontal_distance = float(np.linalg.norm(horizontal))
        if horizontal_distance <= 1e-6:
            raise ValueError(
                "cube is directly above/below link_base; the approach "
                "direction is undefined"
            )
        approach_direction = horizontal / horizontal_distance

        if horizontal_distance <= self.cube_approach_offset_m:
            raise ValueError(
                "cube_approach_offset_m is larger than the cube horizontal "
                "distance from link_base"
            )

        standoff = self.solve_reachable_standoff(look_at, approach_direction)
        target = look_at - approach_direction * standoff

        approach_position = {
            "x": float(target[0]),
            "y": float(target[1]),
            "z": float(target[2]),
        }

        self.get_logger().info(
            "Cube radial approach: "
            f"cube_position=[{cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}] m, "
            f"target_position=["
            f"{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] m, "
            f"approach_direction=["
            f"{approach_direction[0]:.3f}, "
            f"{approach_direction[1]:.3f}, "
            f"{approach_direction[2]:.3f}], "
            f"requested_standoff={self.cube_approach_offset_m:.3f} m, "
            f"actual_standoff={standoff:.3f} m, "
            f"cube_distance={float(np.linalg.norm(look_at - target)):.3f} m, "
            f"target_distance_from_base={float(np.linalg.norm(target)):.3f} m"
        )

        if standoff > self.cube_approach_offset_m + 1e-6:
            self.get_logger().warn(
                "The requested standoff put the target outside the configured "
                "workspace. Kept the approach direction and increased the "
                f"standoff from {self.cube_approach_offset_m:.3f} m to "
                f"{standoff:.3f} m."
            )

        return approach_position

    def solve_reachable_standoff(self, look_at, approach_direction):
        """Smallest standoff along -approach_direction that stays reachable.

        |look_at - t * u| <= max_distance を満たす最小の t を求める。
        t^2 - 2 (look_at・u) t + (|look_at|^2 - max^2) = 0 の小さい方の根。
        """
        requested = self.cube_approach_offset_m
        max_distance = self.cartesian_max_target_distance_m

        if float(np.linalg.norm(look_at - approach_direction * requested)) <= max_distance:
            return requested

        projection = float(np.dot(look_at, approach_direction))
        discriminant = (
            projection ** 2
            - float(np.dot(look_at, look_at))
            + max_distance ** 2
        )
        if discriminant < 0.0:
            raise ValueError(
                "cube is outside the configured Cartesian workspace along the "
                "approach direction: "
                f"cube_distance_from_base={float(np.linalg.norm(look_at)):.3f} m, "
                f"max_target_distance={max_distance:.3f} m. "
                "Increase cartesian_max_target_distance_m or change the "
                "robot/cube arrangement."
            )

        return max(requested, projection - math.sqrt(discriminant))

    def compute_x_axis_approach_position(self, cube_pose):
        """Match the cube Y/Z position and keep standoff only along X."""
        translation = cube_pose["translation_m"]
        cube_x = float(translation["x"])
        cube_y = float(translation["y"])
        cube_z = float(translation["z"])

        target_y = cube_y
        target_z = cube_z + self.cube_target_z_offset_m

        if abs(cube_x) <= 1e-6:
            raise ValueError(
                "cube X position is too close to the link_base origin"
            )
        if abs(cube_x) <= self.cube_approach_offset_m:
            raise ValueError(
                "cube_approach_offset_m is larger than the cube X distance "
                "from link_base"
            )

        # cubeが+X側なら手前は-X方向、-X側なら手前は+X方向。
        approach_sign = 1.0 if cube_x >= 0.0 else -1.0
        requested_target_x = (
            cube_x - approach_sign * self.cube_approach_offset_m
        )

        # YとZをcubeに固定したまま、到達範囲内に入るXの上限を求める。
        max_distance_sq = self.cartesian_max_target_distance_m ** 2
        fixed_yz_distance_sq = target_y ** 2 + target_z ** 2

        if fixed_yz_distance_sq >= max_distance_sq:
            raise ValueError(
                "cube Y/Z position alone is outside the configured Cartesian "
                "workspace: "
                f"target_y={target_y:.3f} m, "
                f"target_z={target_z:.3f} m, "
                f"max_target_distance="
                f"{self.cartesian_max_target_distance_m:.3f} m. "
                "Increase cartesian_max_target_distance_m or change the "
                "robot/cube arrangement."
            )

        max_abs_x = math.sqrt(max_distance_sq - fixed_yz_distance_sq)

        if cube_x >= 0.0:
            target_x = min(requested_target_x, max_abs_x)
            target_x = max(target_x, 0.0)
        else:
            target_x = max(requested_target_x, -max_abs_x)
            target_x = min(target_x, 0.0)

        approach_position = {
            "x": target_x,
            "y": target_y,
            "z": target_z,
        }

        actual_target_distance = math.sqrt(
            target_x ** 2 + target_y ** 2 + target_z ** 2
        )
        actual_x_standoff = abs(cube_x - target_x)
        actual_cube_distance = math.sqrt(
            (cube_x - target_x) ** 2
            + (cube_y - target_y) ** 2
            + (cube_z - target_z) ** 2
        )

        self.get_logger().info(
            "Cube Y/Z-aligned approach: "
            f"cube_position=[{cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}] m, "
            f"target_position=["
            f"{target_x:.3f}, {target_y:.3f}, {target_z:.3f}] m, "
            f"y_difference={target_y - cube_y:.3f} m, "
            f"z_difference={target_z - cube_z:.3f} m, "
            f"requested_x_standoff="
            f"{self.cube_approach_offset_m:.3f} m, "
            f"actual_x_standoff={actual_x_standoff:.3f} m, "
            f"actual_cube_distance={actual_cube_distance:.3f} m, "
            f"target_distance_from_base="
            f"{actual_target_distance:.3f} m"
        )

        if abs(target_x - requested_target_x) > 1e-6:
            self.get_logger().warn(
                "The requested X approach position was outside the configured "
                "workspace. Preserved the cube Y/Z alignment and clamped only "
                f"X from {requested_target_x:.3f} m to {target_x:.3f} m."
            )

        return approach_position

    def create_cartesian_pose(self, approach_position_m, cube_pose=None):
        rpy = self.get_cartesian_rpy(approach_position_m, cube_pose)

        return [
            float(approach_position_m["x"]) * 1000.0,
            float(approach_position_m["y"]) * 1000.0,
            float(approach_position_m["z"]) * 1000.0,
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ]

    def get_cartesian_rpy(self, approach_position_m, cube_pose):
        if self.cartesian_orientation_mode == "fixed":
            return list(self.cartesian_rpy_rad)

        try:
            if self.cartesian_orientation_mode == "front_fixed":
                tool_forward = np.asarray(
                    self.cartesian_front_direction,
                    dtype=np.float64,
                )
            else:
                if cube_pose is None:
                    raise ValueError(
                        "cube pose is required for point_at_cube mode"
                    )
                tool_forward = self.get_cube_direction(
                    approach_position_m,
                    cube_pose,
                )

            rotation = self.create_tool_rotation(tool_forward)
            rpy = rotation.as_euler("xyz").tolist()

            normalized_forward = (
                tool_forward / np.linalg.norm(tool_forward)
            )
            camera_direction_world = rotation.apply(
                np.array([1.0, 0.0, 0.0], dtype=np.float64)
            )
            self.get_logger().info(
                "Cartesian orientation: "
                f"mode={self.cartesian_orientation_mode}, "
                f"tool_forward=["
                f"{normalized_forward[0]:.3f}, "
                f"{normalized_forward[1]:.3f}, "
                f"{normalized_forward[2]:.3f}], "
                f"camera_direction_world=["
                f"{camera_direction_world[0]:.3f}, "
                f"{camera_direction_world[1]:.3f}, "
                f"{camera_direction_world[2]:.3f}], "
                f"tool_axis_roll="
                f"{math.degrees(self.cartesian_roll_rad):.1f} deg, "
                f"result_rpy_deg=["
                f"{math.degrees(rpy[0]):.1f}, "
                f"{math.degrees(rpy[1]):.1f}, "
                f"{math.degrees(rpy[2]):.1f}]"
            )
            return rpy
        except ValueError as error:
            self.get_logger().warn(
                f"Falling back to fixed Cartesian RPY: {error}"
            )
            return list(self.cartesian_rpy_rad)

    def get_cube_direction(self, approach_position_m, cube_pose):
        cube_translation = cube_pose["translation_m"]
        cube_position = np.array([
            float(cube_translation["x"]),
            float(cube_translation["y"]),
            float(cube_translation["z"]),
        ], dtype=np.float64)
        approach_position = np.array([
            float(approach_position_m["x"]),
            float(approach_position_m["y"]),
            float(approach_position_m["z"]),
        ], dtype=np.float64)

        direction = cube_position - approach_position

        if self.cartesian_level_tool_forward:
            # 上下成分を捨て、手先前方を水平に保つ。
            direction[2] = 0.0

        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            raise ValueError(
                "approach pose and cube pose are too close, or the cube is "
                "directly above/below the approach pose while "
                "cartesian_level_tool_forward is enabled"
            )

        direction /= norm
        if self.cartesian_point_axis_sign < 0.0:
            direction = -direction
        return direction

    def create_tool_rotation(self, tool_forward):
        """Constrain hand forward and hand-camera direction independently.

        link_eef local axes used here:
          Z: hand/tool forward
          X: hand-camera viewing/upward direction
          Y: completes the right-handed coordinate frame

        Therefore:
          local Z -> cartesian_front_direction
          local X -> cartesian_camera_up_direction
        """
        tool_z_axis = np.asarray(tool_forward, dtype=np.float64)
        forward_norm = np.linalg.norm(tool_z_axis)
        if forward_norm <= 1e-9:
            raise ValueError("tool forward direction must not be zero")
        tool_z_axis /= forward_norm

        desired_camera_direction = np.asarray(
            self.cartesian_camera_up_direction,
            dtype=np.float64,
        )
        camera_norm = np.linalg.norm(desired_camera_direction)
        if camera_norm <= 1e-9:
            raise ValueError("camera direction must not be zero")
        desired_camera_direction /= camera_norm

        # カメラ方向から手先前方軸と平行な成分を除去し、
        # link_eefのローカルX軸として使用する。
        tool_x_axis = (
            desired_camera_direction
            - float(np.dot(
                desired_camera_direction,
                tool_z_axis,
            )) * tool_z_axis
        )
        tool_x_norm = np.linalg.norm(tool_x_axis)

        if tool_x_norm <= 1e-6:
            raise ValueError(
                "camera direction is parallel to tool forward direction"
            )
        tool_x_axis /= tool_x_norm

        # X × Y = Z となるようにローカルY軸を構成する。
        tool_y_axis = np.cross(tool_z_axis, tool_x_axis)
        tool_y_axis /= np.linalg.norm(tool_y_axis)

        # 数値誤差を除去するため、X軸を再計算する。
        tool_x_axis = np.cross(tool_y_axis, tool_z_axis)
        tool_x_axis /= np.linalg.norm(tool_x_axis)

        nominal_rotation = R.from_matrix(
            np.column_stack((
                tool_x_axis,
                tool_y_axis,
                tool_z_axis,
            ))
        )

        # 必要な場合だけ、手先前方軸まわりに追加回転を加える。
        tool_axis_roll = R.from_euler(
            "z",
            self.cartesian_roll_rad,
        )
        return nominal_rotation * tool_axis_roll

    def load_cached_cube_pose(self, side):
        data = self.load_cube_pose_cache()
        cube_record = data.get("cubes", {}).get(side)
        if not cube_record:
            return None

        cube_pose = cube_record.get("cube_pose")
        if not cube_pose:
            return None

        if cube_pose.get("base_frame") != self.cube_base_frame:
            self.get_logger().warn(
                f"Cached {side} cube base frame is {cube_pose.get('base_frame')}, "
                f"expected {self.cube_base_frame}"
            )

        return cube_pose

    def save_cube_pose(self, side, cube_pose, approach_position_m, cartesian_pose):
        data = self.load_cube_pose_cache()
        data.setdefault("cubes", {})
        data["cubes"][side] = {
            "cube_pose": cube_pose,
            "approach": {
                "approach_offset_m": float(self.cube_approach_offset_m),
                "target_z_offset_m": float(self.cube_target_z_offset_m),
                "position_m": {
                    "x": float(approach_position_m["x"]),
                    "y": float(approach_position_m["y"]),
                    "z": float(approach_position_m["z"]),
                },
                "moveit_target_pose_mm_rad": [
                    float(value) for value in cartesian_pose
                ],
            },
        }

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

    def move_with_cartesian_pose(self, cartesian_pose, side="latest"):
        return self.move_with_moveit_cartesian_pose(cartesian_pose, side)

    def move_with_moveit_cartesian_pose(self, cartesian_pose, side="latest"):
        """Solve a Cartesian target, then execute via MoveGroup."""
        if self.latest_joint_state is None and self.publish_warmup_time > 0.0:
            self.latest_joint_state_event.wait(timeout=self.publish_warmup_time)

        target_joint1 = None
        if self.align_joint1_before_move:
            current_positions = self.get_current_joint_positions()
            target_joint1 = self.compute_target_joint1_rad(
                cartesian_pose,
                current_positions[0] if current_positions else None,
            )
            if target_joint1 is None:
                self.get_logger().warn(
                    "The target is on the link_base Z axis, so the joint1 "
                    "azimuth is undefined; skipping the alignment move."
                )
            else:
                success, message = self.align_joint1_with_trajectory(
                    target_joint1,
                    side,
                )
                if not success:
                    return False, f"joint1 alignment failed: {message}"

        success, target_positions, message = self.request_ik_solution(
            cartesian_pose,
            side,
            target_joint1,
        )
        if not success:
            return False, message

        success, ret, message = self.move_joints_with_moveit(
            target_positions,
            "Cartesian IK solution",
        )
        if not success:
            return False, message

        return True, "Cartesian move succeeded via MoveIt IK -> MoveGroup"

    def get_sim_ik_client(self):
        if GetPositionIK is None:
            return None

        if self.sim_ik_client is None:
            self.sim_ik_client = self.create_client(
                GetPositionIK,
                self.sim_ik_service_name,
                callback_group=self.callback_group,
            )
        return self.sim_ik_client

    def get_find_cube_detection_client(self):
        if self.find_cube_detection_client is None:
            self.find_cube_detection_client = self.create_client(
                SetBool,
                self.find_cube_detection_service_name,
                callback_group=self.callback_group,
            )
        return self.find_cube_detection_client

    def set_find_cube_detection_enabled(self, enabled, required=False):
        if not self.manage_find_cube_detection:
            return True, "find_cube detection management disabled"

        client = self.get_find_cube_detection_client()
        timeout = max(self.find_cube_detection_service_timeout, 0.0)
        if not client.wait_for_service(timeout_sec=timeout):
            message = (
                f"find_cube detection service is not available: "
                f"{self.find_cube_detection_service_name}"
            )
            if required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        request = SetBool.Request()
        request.data = bool(enabled)
        future = client.call_async(request)
        if not self.wait_for_future(future, timeout + 1.0):
            message = "timed out waiting for find_cube detection service"
            if required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        if future.exception() is not None:
            message = (
                f"find_cube detection service call failed: "
                f"{future.exception()}"
            )
            if required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        result = future.result()
        if not result.success:
            message = result.message or "find_cube detection service returned false"
            if required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        self.get_logger().info(result.message)
        return True, result.message

    def get_multi_cube_detect_client(self):
        if self.multi_cube_detect_client is None:
            self.multi_cube_detect_client = self.create_client(
                Trigger,
                self.multi_cube_detect_service_name,
                callback_group=self.callback_group,
            )
        return self.multi_cube_detect_client

    def call_multi_cube_detect_once(self):
        if not self.detect_multi_cube_after_approach:
            return True, "multi-cube detection after approach disabled"

        client = self.get_multi_cube_detect_client()
        timeout = max(self.multi_cube_detect_service_timeout, 0.0)
        if not client.wait_for_service(timeout_sec=timeout):
            message = (
                f"multi-cube detect service is not available: "
                f"{self.multi_cube_detect_service_name}"
            )
            if self.multi_cube_detect_required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        self.get_logger().info(
            f"Calling multi-cube ChArUco detection: "
            f"{self.multi_cube_detect_service_name}"
        )
        future = client.call_async(Trigger.Request())
        if not self.wait_for_future(future, timeout):
            message = "timed out waiting for multi-cube detect service"
            if self.multi_cube_detect_required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        if future.exception() is not None:
            message = (
                f"multi-cube detect service call failed: "
                f"{future.exception()}"
            )
            if self.multi_cube_detect_required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        result = future.result()
        if not result.success:
            message = result.message or "multi-cube detect service returned false"
            if self.multi_cube_detect_required:
                return False, message
            self.get_logger().warn(message)
            return True, message

        self.get_logger().info(result.message)
        return True, result.message

    def get_side_for_ik_seed(self, side):
        if side == "latest":
            return self.camera_side
        if side in ("left", "right"):
            return side
        return self.camera_side

    def get_side_pose_seed(self, side):
        resolved_side = self.get_side_for_ik_seed(side)
        if resolved_side == "left":
            degrees = self.left_joint_degrees
        else:
            degrees = self.right_joint_degrees
        return [math.radians(value) for value in degrees]

    def get_sim_ik_seed_positions(self, side, joint1_rad=None):
        current_positions = self.get_current_joint_positions()

        if self.sim_ik_seed_mode == "current":
            seed, source = current_positions, "current"
        elif self.sim_ik_seed_mode == "current_or_side":
            if current_positions is not None:
                seed, source = current_positions, "current"
            else:
                seed, source = self.get_side_pose_seed(side), "side_pose"
        else:
            seed, source = self.get_side_pose_seed(side), "side_pose"

        if seed is not None and joint1_rad is not None:
            seed = list(seed)
            seed[0] = float(joint1_rad)
            source = f"{source}+joint1_aligned"

        return seed, source

    def compute_target_joint1_rad(self, cartesian_pose, current_joint1=None):
        """joint1 angle that puts the arm plane on the target azimuth.

        link_baseから見た目標のXY方位をそのままjoint1にする。
        joint1は±2piまで回せるため、現在角に最も近い等価角を選ぶ。
        """
        target_x = float(cartesian_pose[0])
        target_y = float(cartesian_pose[1])
        if math.hypot(target_x, target_y) <= 1e-6:
            return None

        yaw = math.atan2(target_y, target_x)
        if current_joint1 is None:
            return yaw

        best = None
        for turn in (-1, 0, 1):
            candidate = yaw + 2.0 * math.pi * turn
            if abs(candidate) > JOINT1_LIMIT_RAD:
                continue
            if best is None or abs(candidate - current_joint1) < abs(
                best - current_joint1
            ):
                best = candidate
        return yaw if best is None else best

    def execute_joint_positions(self, target_positions, duration):
        del duration
        success, _, message = self.move_joints_with_moveit(
            target_positions,
            "joint1 alignment",
        )
        return success, message

    def align_joint1_with_trajectory(self, target_joint1, side):
        """Rotate joint1 only, keeping the other joints where they are."""
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            current_positions = self.get_side_pose_seed(side)
            self.get_logger().warn(
                "No joint state available for the joint1 alignment; "
                "starting from the side pose instead."
            )

        if abs(target_joint1 - current_positions[0]) <= self.joint1_align_tolerance_rad:
            self.get_logger().info(
                "joint1 is already on the target azimuth "
                f"({math.degrees(current_positions[0]):.1f} deg); "
                "skipping the alignment move."
            )
            return True, "joint1 already aligned"

        aligned_positions = list(current_positions)
        aligned_positions[0] = float(target_joint1)

        self.get_logger().info(
            "Aligning joint1 before the Cartesian move: "
            f"{math.degrees(current_positions[0]):.1f} deg -> "
            f"{math.degrees(target_joint1):.1f} deg, "
            f"duration={self.joint1_align_duration:.1f}s"
        )

        return self.execute_joint_positions(
            aligned_positions,
            self.joint1_align_duration,
        )

    def request_ik_solution(self, cartesian_pose, side="latest", target_joint1=None):
        """Ask MoveIt for the joint solution of a Cartesian pose.

        Returns (success, joint_positions_rad or None, message).
        軌道は送らないので、到達性や選ばれる分岐の確認にも使える。
        """
        ik_client = self.get_sim_ik_client()
        if ik_client is None:
            return (
                False,
                None,
                "moveit_msgs is not installed. Install/source MoveIt before "
                "using execution_mode:=sim.",
            )
        if not ik_client.wait_for_service(timeout_sec=10.0):
            return (
                False,
                None,
                f"MoveIt IK service not available: {self.sim_ik_service_name}. "
                "Start the xArm MoveIt fake/simulation launch.",
            )

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.cube_base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(cartesian_pose[0]) / 1000.0
        pose_stamped.pose.position.y = float(cartesian_pose[1]) / 1000.0
        pose_stamped.pose.position.z = float(cartesian_pose[2]) / 1000.0

        quaternion = R.from_euler(
            "xyz",
            [
                float(cartesian_pose[3]),
                float(cartesian_pose[4]),
                float(cartesian_pose[5]),
            ],
        ).as_quat()
        pose_stamped.pose.orientation.x = float(quaternion[0])
        pose_stamped.pose.orientation.y = float(quaternion[1])
        pose_stamped.pose.orientation.z = float(quaternion[2])
        pose_stamped.pose.orientation.w = float(quaternion[3])

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.sim_ik_group_name
        ik_request.ik_request.ik_link_name = self.sim_ik_link_name
        ik_request.ik_request.pose_stamped = pose_stamped
        ik_request.ik_request.timeout = Duration(
            seconds=self.sim_ik_timeout
        ).to_msg()
        ik_request.ik_request.avoid_collisions = self.sim_avoid_collisions
        ik_request.ik_request.robot_state.is_diff = True

        seed_positions, seed_source = self.get_sim_ik_seed_positions(
            side,
            target_joint1,
        )
        if seed_positions is not None:
            ik_request.ik_request.robot_state.joint_state.name = list(
                self.joint_names
            )
            ik_request.ik_request.robot_state.joint_state.position = list(
                seed_positions
            )

        self.get_logger().info(
            "Requesting simulation IK: "
            f"service={self.sim_ik_service_name}, "
            f"group={self.sim_ik_group_name}, "
            f"link={self.sim_ik_link_name}, "
            f"frame={self.cube_base_frame}, "
            f"seed={seed_source}, "
            f"seed_deg="
            f"{[round(math.degrees(v), 1) for v in seed_positions] if seed_positions is not None else None}, "
            f"pose_mm_rad={cartesian_pose}"
        )

        future = ik_client.call_async(ik_request)
        if not self.wait_for_future(
            future,
            self.sim_ik_timeout + 10.0,
        ):
            return False, None, "Timed out waiting for MoveIt IK response"

        if future.exception() is not None:
            return (
                False,
                None,
                f"MoveIt IK service call failed: {future.exception()}",
            )

        ik_response = future.result()
        if ik_response.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                None,
                "MoveIt IK failed: "
                f"error_code={ik_response.error_code.val}, "
                f"group={self.sim_ik_group_name}, "
                f"link={self.sim_ik_link_name}",
            )

        solution_map = dict(zip(
            ik_response.solution.joint_state.name,
            ik_response.solution.joint_state.position,
        ))
        missing_joints = [
            name for name in self.joint_names if name not in solution_map
        ]
        if missing_joints:
            return (
                False,
                None,
                "IK response does not contain required joints: "
                + ", ".join(missing_joints),
            )

        target_positions = [
            float(solution_map[name]) for name in self.joint_names
        ]

        self.get_logger().info(
            "IK solution: solution_deg="
            f"{[round(math.degrees(v), 1) for v in target_positions]}"
        )

        return True, target_positions, "success"

    def _handle_move_request(self, request, response, default_angles):
        self.set_find_cube_detection_enabled(False, required=False)

        input_angles = (
            list(request.angles) if request.angles else default_angles
        )
        if len(input_angles) != 6:
            response.ret = -1
            response.message = (
                "angles must contain exactly 6 values in joint1..joint6 order"
            )
            self.get_logger().error(response.message)
            return response

        success, ret, message = self.move_to_pre_marker_pose()
        if not success:
            response.ret = ret
            response.message = f"pre-marker move failed: {message}"
            self.get_logger().error(response.message)
            return response

        if self.input_unit == "deg":
            xarm_angles = [math.radians(angle) for angle in input_angles]
        else:
            xarm_angles = [float(angle) for angle in input_angles]

        if request.relative:
            current_positions = self.get_current_joint_positions()
            if current_positions is None:
                response.ret = -25
                response.message = (
                    f"No joint state on {self.joint_state_topic}; cannot "
                    "convert relative joint command to an absolute MoveGroup "
                    "target"
                )
                self.get_logger().error(response.message)
                return response
            xarm_angles = [
                current + delta
                for current, delta in zip(current_positions, xarm_angles)
            ]

        if request.speed > 0.0 or request.acc > 0.0 or request.mvtime > 0.0:
            self.get_logger().warn(
                "speed/acc/mvtime parameters are accepted for service "
                "compatibility, but MoveGroup planning/execution uses the "
                "MoveIt configuration for timing."
            )

        success, ret, message = self.move_joints_with_moveit(
            xarm_angles,
            f"angles_{self.input_unit}={input_angles}",
        )
        if success:
            success, ret, message = self.return_to_pre_marker_pose_after_success()
            if not success:
                message = f"return to pre-marker pose failed: {message}"
            elif self.restore_find_cube_detection_after_move:
                self.set_find_cube_detection_enabled(True, required=False)

        response.ret = ret
        response.message = message
        if success:
            self.get_logger().info(
                "Target pose command succeeded via MoveGroup and returned "
                "to pre-marker pose"
            )
        else:
            self.get_logger().error(message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MoveToCharucoPoseNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
