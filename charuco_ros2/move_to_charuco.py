#!/usr/bin/env python3
import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from xarm_msgs.srv import MoveJoint


RIGHT_JOINT_DEGREES = [-53.0, 55.0, -110.0, 0.0, -52.0, -8.0]
LEFT_JOINT_DEGREES = [53.0, 55.0, -110.0, 0.0, -62.0, -8.0]
DEFAULT_JOINT_DEGREES = LEFT_JOINT_DEGREES
DEFAULT_SPEED = 0.035  # 0.1x of the xArm README example 0.35 rad/s
DEFAULT_ACC = 1.0  # 0.1x of the xArm README example 10 rad/s^2
MOVE_JOINT_TYPE = "xarm_msgs/srv/MoveJoint"


class MoveToCharucoPoseNode(Node):
    def __init__(self):
        super().__init__("move_to_charuco_pose")

        self.declare_parameter("server_service_name", "/move_to_charuco")
        self.declare_parameter("backend", "topic")
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

        self.server_service_name = self.get_parameter("server_service_name").value
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

        if self.camera_side == "left":
            self.side_joint_degrees = self.left_joint_degrees
        elif self.camera_side == "right":
            self.side_joint_degrees = self.right_joint_degrees
        else:
            raise ValueError("camera_side must be 'left' or 'right'")

        if self.backend not in ("auto", "xarm_api", "trajectory", "topic"):
            raise ValueError(
                "backend must be 'auto', 'xarm_api', 'trajectory', or 'topic'"
            )
        if self.input_unit not in ("deg", "rad"):
            raise ValueError("input_unit must be 'deg' or 'rad'")

        self.callback_group = ReentrantCallbackGroup()
        self.resolved_xarm_service_name = None
        self.xarm_client = None
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

        self.get_logger().info(
            f"Services ready: {self.server_service_name} | "
            f"{self.server_service_name}/left | {self.server_service_name}/right "
            f"(angles unit: {self.input_unit})"
        )
        self.get_logger().info(
            f"backend={self.backend}, xarm_service={self.xarm_service_name}, "
            f"trajectory_topic={self.trajectory_topic}, "
            f"trajectory_action={self.trajectory_action_name}"
        )
        self.get_logger().info(
            f"left pose:  {self.left_joint_degrees} deg\n"
            f"right pose: {self.right_joint_degrees} deg"
        )

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
        if self.backend in ("trajectory", "topic"):
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
        if self.backend == "xarm_api":
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
        trajectory.header.stamp = self.get_clock().now().to_msg()
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

        xarm_client = self.get_xarm_client()
        if self.backend == "topic":
            return self.move_with_joint_trajectory_topic(
                input_angles,
                xarm_angles,
                request,
                response,
            )

        if xarm_client is None:
            return self.move_with_joint_trajectory(
                input_angles,
                xarm_angles,
                request,
                response,
            )

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
