#!/usr/bin/env python3
import math
import os
import threading
import time

import numpy as np
import yaml
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped

try:
    from moveit_msgs.msg import MoveItErrorCodes
    from moveit_msgs.srv import GetPositionIK
except ImportError:
    MoveItErrorCodes = None
    GetPositionIK = None
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from xarm_msgs.srv import MoveCartesian, MoveJoint, SetInt16, SetInt16ById


RIGHT_JOINT_DEGREES = [-53.0, 55.0, -110.0, 0.0, -52.0, -8.0]
LEFT_JOINT_DEGREES = [53.0, 55.0, -110.0, 0.0, -62.0, -8.0]
DEFAULT_JOINT_DEGREES = LEFT_JOINT_DEGREES
DEFAULT_SPEED = 0.035  # 0.1x of the xArm README example 0.35 rad/s
DEFAULT_ACC = 1.0  # 0.1x of the xArm README example 10 rad/s^2
MOVE_JOINT_TYPE = "xarm_msgs/srv/MoveJoint"
MOVE_CARTESIAN_TYPE = "xarm_msgs/srv/MoveCartesian"
JOINT1_LIMIT_RAD = 2.0 * math.pi


class MoveToCharucoPoseNode(Node):
    def __init__(self):
        super().__init__("move_to_charuco_pose")

        self.declare_parameter("server_service_name", "/move_to_charuco")
        self.declare_parameter("execution_mode", "real")
        self.declare_parameter("backend", "auto")
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
        # realでもsimと同じ経路を通す。
        # falseだとxArmの set_position を使うが、その場合IKを解くのはMoveItでは
        # なくxArmのファームウェアであり、選ばれる分岐も経路生成もsimと一致
        # しない。simで確認した動きをそのまま実機で再現したいならtrueにする。
        # /compute_ik が無い場合は set_position へフォールバックする。
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

        if self.backend not in ("auto", "xarm_api", "trajectory", "topic"):
            raise ValueError(
                "backend must be 'auto', 'xarm_api', 'trajectory', or 'topic'"
            )
        if self.input_unit not in ("deg", "rad"):
            raise ValueError("input_unit must be 'deg' or 'rad'")
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
        self.resolved_xarm_service_name = None
        self.xarm_client = None
        self.resolved_cartesian_service_name = None
        self.cartesian_client = None
        self.sim_ik_client = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joint_names = [
            f"{self.joint_prefix}joint{i}" for i in range(1, 7)
        ]
        self.trajectory_topic = f"/{self.controller_name}/joint_trajectory"
        self.trajectory_action_name = (
            f"/{self.controller_name}/follow_joint_trajectory"
        )
        self.trajectory_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action_name,
            callback_group=self.callback_group,
        )
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.trajectory_topic,
            10,
        )
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

        self.get_logger().info(
            f"Services ready: {self.server_service_name} | "
            f"{self.server_service_name}/left | {self.server_service_name}/right "
            f"(angles unit: {self.input_unit})"
        )
        self.get_logger().info(
            f"execution_mode={self.execution_mode}, "
            f"backend={self.backend} "
            f"(effective={self.get_effective_joint_backend()}), "
            f"xarm_service={self.xarm_service_name}, "
            f"trajectory_topic={self.trajectory_topic}, "
            f"trajectory_action={self.trajectory_action_name}"
        )
        self.get_logger().info(
            f"left pose:  {self.left_joint_degrees} deg\n"
            f"right pose: {self.right_joint_degrees} deg"
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
        if self.execution_mode == "real":
            if self.real_use_moveit_ik:
                self.get_logger().info(
                    "Cartesian moves use MoveIt IK -> set_servo_angle, "
                    "so the motion matches execution_mode:=sim"
                )
            else:
                self.get_logger().warn(
                    "real_use_moveit_ik is false: Cartesian moves use the xArm "
                    "set_position service, which solves the IK and plans the "
                    "motion inside the xArm firmware. The motion will NOT "
                    "match execution_mode:=sim."
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
        """Resolve the actual joint command path for the selected environment."""
        if self.backend != "auto":
            return self.backend
        if self.execution_mode == "real":
            return "xarm_api"
        return "trajectory"

    def resolve_xarm_service_name(self):
        if self.xarm_service_name != "auto":
            return self.xarm_service_name

        matches = []
        for service_name, service_types in self.get_service_names_and_types():
            if service_name.endswith("/set_servo_angle") and MOVE_JOINT_TYPE in service_types:
                matches.append(service_name)

        if not matches:
            return None

        if "/xarm/set_servo_angle" in matches:
            return "/xarm/set_servo_angle"

        return sorted(matches)[0]

    def get_xarm_client(self):
        if self.execution_mode != "real":
            return None

        if self.get_effective_joint_backend() in ("trajectory", "topic"):
            return None

        service_name = self.resolve_xarm_service_name()
        if service_name is None:
            return None

        if (
            self.xarm_client is None
            or service_name != self.resolved_xarm_service_name
        ):
            self.resolved_xarm_service_name = service_name
            self.xarm_client = self.create_client(
                MoveJoint,
                service_name,
                callback_group=self.callback_group,
            )
            self.get_logger().info(f"Forwarding commands to: {service_name}")

        return self.xarm_client

    def resolve_cartesian_service_name(self):
        if self.cartesian_service_name != "auto":
            return self.cartesian_service_name

        matches = []
        for service_name, service_types in self.get_service_names_and_types():
            if (
                service_name.endswith("/set_position")
                and MOVE_CARTESIAN_TYPE in service_types
            ):
                matches.append(service_name)

        if not matches:
            return None

        if "/xarm/set_position" in matches:
            return "/xarm/set_position"

        return sorted(matches)[0]

    def get_cartesian_client(self):
        service_name = self.resolve_cartesian_service_name()
        if service_name is None:
            return None

        if (
            self.cartesian_client is None
            or service_name != self.resolved_cartesian_service_name
        ):
            self.resolved_cartesian_service_name = service_name
            self.cartesian_client = self.create_client(
                MoveCartesian,
                service_name,
                callback_group=self.callback_group,
            )
            self.get_logger().info(f"Forwarding Cartesian commands to: {service_name}")

        return self.cartesian_client

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

    def move_with_joint_trajectory_topic(
        self, input_angles, target_positions, request, response
    ):
        move_duration = request.mvtime if request.mvtime > 0.0 else self.default_move_duration
        if self.latest_joint_state is None and self.publish_warmup_time > 0.0:
            self.latest_joint_state_event.wait(timeout=self.publish_warmup_time)
        current_positions = self.get_current_joint_positions()

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        if current_positions is not None:
            start_point = JointTrajectoryPoint()
            start_point.positions = current_positions
            start_point.time_from_start.sec = 0
            start_point.time_from_start.nanosec = 0
            trajectory.points.append(start_point)
        else:
            self.get_logger().warn(
                f"No usable joint state on {self.joint_state_topic}; "
                "publishing target-only trajectory"
            )

        target_point = JointTrajectoryPoint()
        target_point.positions = target_positions
        target_point.time_from_start.sec = int(move_duration)
        target_point.time_from_start.nanosec = int(
            (move_duration - int(move_duration)) * 1e9
        )
        trajectory.points.append(target_point)

        self.get_logger().info(
            "Publishing trajectory to "
            f"{self.trajectory_topic}: angles_{self.input_unit}={input_angles}, "
            f"duration={move_duration:.3f}s"
        )

        if self.publish_warmup_time > 0.0:
            time.sleep(self.publish_warmup_time)

        self.trajectory_publisher.publish(trajectory)

        time.sleep(move_duration + 1.0)
        response.ret = 0
        response.message = "success"

        if not self.wait_for_joint_state_target(target_positions):
            response.message = (
                "success, but /joint_states did not reach target before timeout; "
                "RViz current state may still be stale"
            )
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Trajectory published and /joint_states reached target")
        return response

    def move_with_joint_trajectory(self, input_angles, target_positions, request, response):
        if self.get_effective_joint_backend() == "xarm_api":
            response.ret = -2
            response.message = "xArm set_servo_angle service not found"
            self.get_logger().error(response.message)
            return response

        if not self.trajectory_action_client.wait_for_server(timeout_sec=10.0):
            response.ret = -5
            response.message = (
                f"Action server not available: {self.trajectory_action_name}. "
                "Start xarm6_moveit_fake.launch.py or xarm6_moveit_realmove.launch.py."
            )
            self.get_logger().error(response.message)
            return response

        move_duration = request.mvtime if request.mvtime > 0.0 else self.default_move_duration
        trajectory = JointTrajectory()
        # stampはゼロのままにする。ゼロ以外を入れると、コントローラへ届く頃には
        # その時刻が過去になっており、軌道が拒否される。
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = Duration(seconds=move_duration).to_msg()
        trajectory.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.get_logger().info(
            "Sending trajectory goal: "
            f"angles_{self.input_unit}={input_angles}, duration={move_duration:.3f}s"
        )

        send_goal_future = self.trajectory_action_client.send_goal_async(goal)
        if not self.wait_for_future(send_goal_future, 10.0):
            response.ret = -6
            response.message = "Timed out sending trajectory goal"
            self.get_logger().error(response.message)
            return response

        if send_goal_future.exception() is not None:
            response.ret = -7
            response.message = (
                f"Failed to send trajectory goal: {send_goal_future.exception()}"
            )
            self.get_logger().error(response.message)
            return response

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            response.ret = -8
            response.message = "Trajectory goal rejected"
            self.get_logger().error(response.message)
            return response

        result_future = goal_handle.get_result_async()
        wait_timeout = move_duration + 10.0
        if not self.wait_for_future(result_future, wait_timeout):
            response.ret = -9
            response.message = "Timed out waiting for trajectory completion"
            self.get_logger().error(response.message)
            return response

        if result_future.exception() is not None:
            response.ret = -10
            response.message = f"Trajectory action failed: {result_future.exception()}"
            self.get_logger().error(response.message)
            return response

        result = result_future.result().result
        response.ret = result.error_code
        response.message = result.error_string if result.error_string else "success"

        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                "Trajectory failed: "
                f"error_code={result.error_code}, message={response.message}"
            )
            return response

        if not self.wait_for_joint_state_target(target_positions):
            response.message = (
                "success, but /joint_states did not reach target before timeout; "
                "RViz current state may still be stale"
            )
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Trajectory completed and /joint_states reached target")
        return response

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

    def _handle_cube_move_request(self, side, response):
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

        return self.set_trigger_response(
            response,
            True,
            (
                f"moved to {side} cube approach pose from {pose_source}: "
                f"pose_mm_rad={cartesian_pose}"
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
                "xarm_set_position_pose_mm_rad": [
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
        if self.execution_mode == "sim":
            return self.move_with_sim_cartesian_pose(
                cartesian_pose,
                side,
            )

        if self.real_use_moveit_ik:
            success, message = self.move_with_real_ik_pose(cartesian_pose, side)
            if success:
                return True, message

            self.get_logger().warn(
                f"Falling back to the xArm set_position path: {message}. "
                "The xArm firmware will solve the IK and plan the motion "
                "itself, so the result will NOT match execution_mode:=sim."
            )

        return self.move_with_real_cartesian_pose(cartesian_pose)

    def move_with_real_ik_pose(self, cartesian_pose, side="latest"):
        """Solve with MoveIt IK, then send the joints to the real xArm.

        simと同じIK・同じ関節指令を使うため、simで確認した動きが実機でも
        そのまま再現される。set_positionはxArm側でIKと経路生成を行うため
        simと一致しない。
        """
        if self.latest_joint_state is None and self.publish_warmup_time > 0.0:
            self.latest_joint_state_event.wait(timeout=self.publish_warmup_time)

        success, target_positions, message = self.request_ik_solution(
            cartesian_pose,
            side,
        )
        if not success:
            return False, message

        return self.move_joints_with_xarm(target_positions)

    def move_joints_with_xarm(self, target_positions):
        """Send joint angles to the real xArm with set_servo_angle."""
        service_name = self.resolve_xarm_service_name()
        if service_name is None:
            return False, "xArm set_servo_angle service not found"

        client = self.create_client(
            MoveJoint,
            service_name,
            callback_group=self.callback_group,
        )
        if not client.wait_for_service(timeout_sec=10.0):
            return False, f"xArm service not available: {service_name}"

        success, message = self.prepare_xarm_for_cartesian()
        if not success:
            return False, message

        request = MoveJoint.Request()
        request.angles = [float(value) for value in target_positions]
        request.speed = self.default_speed
        request.acc = self.default_acc
        request.mvtime = 0.0
        request.wait = True
        request.timeout = self.default_timeout
        request.radius = self.default_radius
        request.relative = False

        self.get_logger().info(
            f"Moving joints via {service_name}: angles_deg="
            f"{[round(math.degrees(v), 1) for v in target_positions]}, "
            f"speed={request.speed} rad/s, acc={request.acc} rad/s^2"
        )

        future = client.call_async(request)
        if not self.wait_for_future(future, request.timeout + 5.0):
            return False, "Timed out waiting for xArm set_servo_angle response"

        if future.exception() is not None:
            return False, f"set_servo_angle failed: {future.exception()}"

        result = future.result()
        if result.ret != 0:
            return (
                False,
                f"set_servo_angle failed: ret={result.ret}, "
                f"message={result.message}",
            )

        self.wait_for_joint_state_target(target_positions)
        return True, "real Cartesian move succeeded via MoveIt IK -> " + service_name

    def move_with_real_cartesian_pose(self, cartesian_pose):
        cartesian_client = self.get_cartesian_client()
        if cartesian_client is None:
            return False, "xArm Cartesian service not found"

        if not cartesian_client.wait_for_service(timeout_sec=10.0):
            return (
                False,
                f"xArm Cartesian service not available: "
                f"{self.resolved_cartesian_service_name}",
            )

        if self.prepare_xarm_before_cartesian:
            success, message = self.prepare_xarm_for_cartesian()
            if not success:
                return False, message

        if self.align_joint1_before_move:
            success, message = self.align_joint1_with_xarm(cartesian_pose)
            if not success:
                return False, f"joint1 alignment failed: {message}"

        request = MoveCartesian.Request()
        request.pose = [float(value) for value in cartesian_pose]
        request.speed = self.cartesian_speed
        request.acc = self.cartesian_acc
        request.mvtime = 0.0
        request.wait = self.cartesian_wait
        request.timeout = self.cartesian_timeout
        request.radius = self.cartesian_radius
        request.is_tool_coord = False
        request.relative = False
        request.motion_type = self.cartesian_motion_type

        self.get_logger().info(
            f"Moving Cartesian pose: pose_mm_rad={request.pose}, "
            f"speed={request.speed} mm/s, acc={request.acc} mm/s^2, "
            f"wait={request.wait}"
        )

        future = cartesian_client.call_async(request)
        wait_timeout = request.timeout + 5.0 if request.wait else 30.0
        if not self.wait_for_future(future, wait_timeout):
            return False, "Timed out waiting for xArm Cartesian service response"

        if future.exception() is not None:
            return False, f"xArm Cartesian service call failed: {future.exception()}"

        result = future.result()
        self.get_logger().info(
            f"set_position response: ret={result.ret}, message={result.message}"
        )
        if result.ret != 0:
            return (
                False,
                f"set_position failed: ret={result.ret}, message={result.message}",
            )

        return True, result.message if result.message else "success"

    def align_joint1_with_xarm(self, cartesian_pose):
        """Rotate joint1 to the target azimuth with set_servo_angle."""
        if self.latest_joint_state is None and self.publish_warmup_time > 0.0:
            self.latest_joint_state_event.wait(timeout=self.publish_warmup_time)

        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            return (
                False,
                f"No joint state on {self.joint_state_topic}; cannot rotate "
                "joint1 before the Cartesian move.",
            )

        target_joint1 = self.compute_target_joint1_rad(
            cartesian_pose,
            current_positions[0],
        )
        if target_joint1 is None:
            self.get_logger().warn(
                "The target is on the link_base Z axis, so the joint1 azimuth "
                "is undefined; skipping the alignment move."
            )
            return True, "joint1 azimuth undefined"

        if abs(target_joint1 - current_positions[0]) <= self.joint1_align_tolerance_rad:
            self.get_logger().info(
                "joint1 is already on the target azimuth "
                f"({math.degrees(current_positions[0]):.1f} deg); "
                "skipping the alignment move."
            )
            return True, "joint1 already aligned"

        service_name = self.resolve_xarm_service_name()
        if service_name is None:
            return False, "xArm set_servo_angle service not found"

        client = self.create_client(
            MoveJoint,
            service_name,
            callback_group=self.callback_group,
        )
        if not client.wait_for_service(timeout_sec=10.0):
            return False, f"xArm service not available: {service_name}"

        aligned_positions = list(current_positions)
        aligned_positions[0] = float(target_joint1)

        request = MoveJoint.Request()
        request.angles = aligned_positions
        request.speed = self.default_speed
        request.acc = self.default_acc
        request.mvtime = 0.0
        request.wait = True
        request.timeout = self.default_timeout
        request.radius = self.default_radius
        request.relative = False

        self.get_logger().info(
            "Aligning joint1 before the Cartesian move: "
            f"{math.degrees(current_positions[0]):.1f} deg -> "
            f"{math.degrees(target_joint1):.1f} deg via {service_name}"
        )

        future = client.call_async(request)
        if not self.wait_for_future(future, request.timeout + 5.0):
            return False, "Timed out waiting for the joint1 alignment response"

        if future.exception() is not None:
            return False, f"set_servo_angle failed: {future.exception()}"

        result = future.result()
        if result.ret != 0:
            return (
                False,
                f"set_servo_angle failed: ret={result.ret}, "
                f"message={result.message}",
            )

        self.wait_for_joint_state_target(aligned_positions)
        return True, result.message if result.message else "success"

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
        request = MoveJoint.Request()
        request.angles = [float(value) for value in target_positions]
        request.mvtime = float(duration)
        request.wait = True
        response = MoveJoint.Response()

        if self.get_effective_joint_backend() == "topic":
            result = self.move_with_joint_trajectory_topic(
                target_positions,
                target_positions,
                request,
                response,
            )
        else:
            result = self.move_with_joint_trajectory(
                target_positions,
                target_positions,
                request,
                response,
            )

        if result.ret != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, result.message
        return True, result.message

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

    def move_with_sim_cartesian_pose(self, cartesian_pose, side="latest"):
        """Convert a Cartesian target to joints with MoveIt IK, then use ros2_control."""
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

        success, message = self.execute_joint_positions(
            target_positions,
            self.sim_cartesian_move_duration,
        )
        if not success:
            return False, message

        return (
            True,
            "simulation Cartesian move succeeded via "
            f"{self.sim_ik_service_name} -> "
            f"{self.trajectory_action_name}",
        )

    def prepare_xarm_for_cartesian(self):
        namespace = self.get_xarm_namespace()
        steps = [
            (
                f"{namespace}/motion_enable",
                SetInt16ById,
                {"id": 8, "data": 1},
            ),
            (
                f"{namespace}/set_mode",
                SetInt16,
                {"data": self.cartesian_mode},
            ),
            (
                f"{namespace}/set_state",
                SetInt16,
                {"data": self.cartesian_state},
            ),
        ]

        for service_name, service_type, values in steps:
            success, message = self.call_prepare_service(
                service_name,
                service_type,
                values,
            )
            if not success:
                return False, message

        return True, "xArm is ready for Cartesian command"

    def get_xarm_namespace(self):
        service_name = self.resolved_cartesian_service_name or "/xarm/set_position"
        if service_name.endswith("/set_position"):
            namespace = service_name[: -len("/set_position")]
            return namespace if namespace else "/xarm"
        return "/xarm"

    def call_prepare_service(self, service_name, service_type, values):
        client = self.create_client(
            service_type,
            service_name,
            callback_group=self.callback_group,
        )
        if not client.wait_for_service(timeout_sec=5.0):
            return False, f"xArm prepare service not available: {service_name}"

        request = service_type.Request()
        for key, value in values.items():
            setattr(request, key, value)

        future = client.call_async(request)
        if not self.wait_for_future(future, 10.0):
            return False, f"Timed out waiting for {service_name}"

        if future.exception() is not None:
            return False, f"{service_name} failed: {future.exception()}"

        result = future.result()
        self.get_logger().info(
            f"{service_name} response: ret={result.ret}, "
            f"message={result.message}"
        )
        if result.ret != 0:
            return (
                False,
                f"{service_name} failed: ret={result.ret}, "
                f"message={result.message}",
            )

        return True, result.message if result.message else "success"

    def _handle_move_request(self, request, response, default_angles):
        input_angles = list(request.angles) if request.angles else default_angles
        if len(input_angles) != 6:
            response.ret = -1
            response.message = (
                "angles must contain exactly 6 values in joint1..joint6 order"
            )
            self.get_logger().error(response.message)
            return response

        if self.input_unit == "deg":
            xarm_angles = [math.radians(angle) for angle in input_angles]
        else:
            xarm_angles = input_angles

        effective_backend = self.get_effective_joint_backend()
        if effective_backend == "topic":
            return self.move_with_joint_trajectory_topic(
                input_angles,
                xarm_angles,
                request,
                response,
            )

        if effective_backend == "trajectory":
            return self.move_with_joint_trajectory(
                input_angles,
                xarm_angles,
                request,
                response,
            )

        xarm_client = self.get_xarm_client()
        if xarm_client is None:
            response.ret = -2
            response.message = (
                "xArm set_servo_angle service not found in real mode"
            )
            self.get_logger().error(response.message)
            return response

        if not xarm_client.wait_for_service(timeout_sec=10.0):
            response.ret = -2
            response.message = (
                f"xArm service not available: {self.resolved_xarm_service_name}"
            )
            self.get_logger().error(response.message)
            return response

        xarm_request = MoveJoint.Request()
        xarm_request.angles = xarm_angles
        xarm_request.speed = request.speed if request.speed > 0.0 else self.default_speed
        xarm_request.acc = request.acc if request.acc > 0.0 else self.default_acc
        xarm_request.mvtime = request.mvtime
        xarm_request.wait = request.wait or self.default_wait
        xarm_request.timeout = (
            request.timeout if request.timeout >= 0.0 else self.default_timeout
        )
        xarm_request.radius = (
            request.radius if request.radius >= 0.0 else self.default_radius
        )
        xarm_request.relative = request.relative

        self.get_logger().info(
            f"Moving joints: angles_{self.input_unit}={input_angles}, "
            f"speed={xarm_request.speed} rad/s, acc={xarm_request.acc} rad/s^2"
        )

        future = xarm_client.call_async(xarm_request)
        wait_timeout = xarm_request.timeout + 5.0 if xarm_request.wait else 30.0

        if not self.wait_for_future(future, wait_timeout):
            response.ret = -4
            response.message = "Timed out waiting for xArm service response"
            self.get_logger().error(response.message)
            return response

        if future.exception() is not None:
            response.ret = -3
            response.message = f"xArm service call failed: {future.exception()}"
            self.get_logger().error(response.message)
            return response

        xarm_response = future.result()
        response.ret = xarm_response.ret
        response.message = xarm_response.message if xarm_response.message else "success"

        if xarm_response.ret != 0:
            self.get_logger().error(
                "set_servo_angle failed: "
                f"ret={xarm_response.ret}, message={xarm_response.message}"
            )
            return response

        if not self.wait_for_joint_state_target(xarm_angles):
            response.message = (
                "success, but /joint_states did not reach target before timeout; "
                "RViz current state may still be stale"
            )
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Target pose command succeeded")
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