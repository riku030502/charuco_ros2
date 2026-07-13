#!/usr/bin/env python3
"""Sweep cube positions around the arm and report the IK solution for each.

move_to_charucoの本番と同じ計算（radial approach、手先の水平化、IKシード）を
そのまま呼ぶ。既定ではIKを解くだけでアームは動かさない。

このノードの経路は「現在姿勢 -> IK解」の関節空間での線形補間なので、IK解と
現在姿勢との関節移動量を見れば、実際に動かさなくても大回りは検出できる。

判定:
  BACKWARD  joint1が目標方位と大きくずれている（肩越しに後ろへ反り返る分岐）
  FAR       いずれかの関節の移動量が大きい
  WRIST     手首(joint4/6)が大きく捻れている
  NO_IK     解なし

使い方（RVizのxArm MoveIt/simを起動した状態で）:

  ros2 run charuco_ros2 sweep_cube_targets --ros-args \
    -p execution_mode:=sim \
    -p cartesian_max_target_distance_m:=0.5 \
    -p server_service_name:=/move_to_charuco_sweep

  # 実際に動かして目視したい場合（1点ずつ動く。時間がかかる）
  ... -p sweep_execute:=true

  # 1点動かすたびに初期姿勢へ戻す（各点を必ず同じ姿勢から始める）
  ... -p sweep_execute:=true -p sweep_return_home:=true -p sweep_dwell_time:=1.5
"""
import math
import sys
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor

from charuco_ros2.move_to_charuco import MoveToCharucoPoseNode


# 疑わしいと判定する閾値。
BACKWARD_JOINT1_ERROR_DEG = 90.0
LARGE_JOINT_TRAVEL_DEG = 150.0
LARGE_WRIST_TRAVEL_DEG = 120.0


def wrap_to_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def make_cube_pose(x, y, z, base_frame):
    return {
        "base_frame": base_frame,
        "child_frame": "sweep_cube",
        "stamp": {"sec": 0, "nanosec": 0},
        "translation_m": {"x": float(x), "y": float(y), "z": float(z)},
        "rotation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def classify(node, cartesian_pose, seed_rad, solution_rad):
    """Return a list of warning tags for one IK solution."""
    tags = []

    target_azimuth = math.atan2(
        float(cartesian_pose[1]),
        float(cartesian_pose[0]),
    )
    joint1_error = abs(math.degrees(wrap_to_pi(solution_rad[0] - target_azimuth)))
    if joint1_error > BACKWARD_JOINT1_ERROR_DEG:
        tags.append(f"BACKWARD(j1 off by {joint1_error:.0f} deg)")

    # コントローラは関節値を線形補間するだけなので、±360°の等価角へ「回り込む」
    # ことはしない。実際の移動量は生の差であり、wrapしてはいけない。
    travel_deg = [
        abs(math.degrees(s - c))
        for s, c in zip(solution_rad, seed_rad)
    ]
    if max(travel_deg) > LARGE_JOINT_TRAVEL_DEG:
        worst = travel_deg.index(max(travel_deg)) + 1
        tags.append(f"FAR(joint{worst} {max(travel_deg):.0f} deg)")

    wrist_travel = max(travel_deg[3], travel_deg[5])
    if wrist_travel > LARGE_WRIST_TRAVEL_DEG:
        tags.append(f"WRIST({wrist_travel:.0f} deg)")

    return tags, travel_deg


def sweep(node):
    azimuth_step = node.get_parameter("sweep_azimuth_step_deg").value
    radii = list(node.get_parameter("sweep_cube_radii_m").value)
    heights = list(node.get_parameter("sweep_cube_heights_m").value)
    execute = bool(node.get_parameter("sweep_execute").value)
    return_home = bool(node.get_parameter("sweep_return_home").value)
    dwell_time = float(node.get_parameter("sweep_dwell_time").value)
    home_degrees = list(
        node.get_parameter("sweep_home_joint_degrees").value or []
    )

    azimuths = [
        a * float(azimuth_step)
        for a in range(int(round(360.0 / float(azimuth_step))))
    ]

    node.latest_joint_state_event.wait(timeout=5.0)
    seed_rad = node.get_current_joint_positions()
    if seed_rad is None:
        print(
            f"ERROR: no joint state on {node.joint_state_topic}. "
            "Start the xArm sim/MoveIt stack first.",
            file=sys.stderr,
        )
        return 1

    # 初期姿勢。sweep_home_joint_degreesが空なら、起動時の姿勢をそのまま使う。
    if home_degrees:
        home_rad = [math.radians(v) for v in home_degrees]
    else:
        home_rad = list(seed_rad)

    # 各点は必ず初期姿勢から出発するので、IKシードも初期姿勢に固定する。
    if return_home:
        seed_rad = list(home_rad)

    print(f"\nseed (start pose): {[round(math.degrees(v), 1) for v in seed_rad]} deg")
    print(f"home pose:         {[round(math.degrees(v), 1) for v in home_rad]} deg")
    print(
        f"grid: azimuth 0..360 step {azimuth_step} deg, "
        f"cube radii {radii} m, cube heights {heights} m"
    )
    print(
        "max target distance: "
        f"{node.cartesian_max_target_distance_m:.2f} m, "
        f"execute={execute}, return_home={return_home}, "
        f"dwell={dwell_time:.1f}s\n"
    )

    if execute and return_home:
        moved, message = node.execute_joint_positions(
            home_rad,
            node.sim_cartesian_move_duration,
        )
        if not moved:
            print(f"ERROR: could not reach the home pose: {message}", file=sys.stderr)
            return 1

    header = (
        f"{'azim':>6} {'r':>5} {'z':>5} | {'target xyz (m)':>22} | "
        f"{'dist':>5} | {'IK solution (deg)':>40} | {'maxΔ':>6} | note"
    )
    print(header)
    print("-" * len(header))

    suspicious = []
    failures = []
    total = 0

    for azimuth_deg in azimuths:
        for radius in radii:
            for height in heights:
                total += 1
                azimuth = math.radians(azimuth_deg)
                cube_pose = make_cube_pose(
                    radius * math.cos(azimuth),
                    radius * math.sin(azimuth),
                    height,
                    node.cube_base_frame,
                )

                label = f"{azimuth_deg:>6.0f} {radius:>5.2f} {height:>5.2f}"

                try:
                    approach = node.compute_approach_position(cube_pose)
                except ValueError as error:
                    print(f"{label} | {'-':>22} | {'-':>5} | SKIP: {error}")
                    continue

                cartesian_pose = node.create_cartesian_pose(approach, cube_pose)
                distance = math.sqrt(sum(v ** 2 for v in cartesian_pose[:3])) / 1000.0
                position = (
                    f"[{approach['x']:6.3f},{approach['y']:6.3f},{approach['z']:6.3f}]"
                )

                ok, solution_rad, message = node.request_ik_solution(cartesian_pose)
                if not ok:
                    failures.append((label, message))
                    print(
                        f"{label} | {position:>22} | {distance:>5.2f} | "
                        f"{'NO_IK':>40} | {'-':>6} | {message}"
                    )
                    continue

                tags, travel_deg = classify(
                    node,
                    cartesian_pose,
                    seed_rad,
                    solution_rad,
                )
                solution_deg = [round(math.degrees(v)) for v in solution_rad]
                note = ", ".join(tags) if tags else "ok"
                if tags:
                    suspicious.append((label, solution_deg, note))

                print(
                    f"{label} | {position:>22} | {distance:>5.2f} | "
                    f"{str(solution_deg):>40} | {max(travel_deg):>6.0f} | {note}"
                )

                if not execute:
                    continue

                moved, move_message = node.execute_joint_positions(
                    solution_rad,
                    node.sim_cartesian_move_duration,
                )
                if not moved:
                    print(f"       move to target failed: {move_message}")

                if dwell_time > 0.0:
                    time.sleep(dwell_time)

                if not return_home:
                    # 戻さない場合、次の点は今いる場所から解くのでシードを更新する。
                    seed_rad = node.get_current_joint_positions() or seed_rad
                    continue

                moved, move_message = node.execute_joint_positions(
                    home_rad,
                    node.sim_cartesian_move_duration,
                )
                if not moved:
                    print(f"       return to home failed: {move_message}")
                    return 1

                if dwell_time > 0.0:
                    time.sleep(dwell_time)

    print(f"\n{total} targets, {len(failures)} without IK, {len(suspicious)} suspicious")
    if suspicious:
        print("\nsuspicious targets:")
        for label, solution_deg, note in suspicious:
            print(f"  {label} -> {solution_deg}  {note}")

    return 0


def main(args=None):
    rclpy.init(args=args)
    node = MoveToCharucoPoseNode()

    node.declare_parameter("sweep_azimuth_step_deg", 45.0)
    node.declare_parameter("sweep_cube_radii_m", [0.35, 0.45, 0.55])
    node.declare_parameter("sweep_cube_heights_m", [0.2, 0.4, 0.6])
    node.declare_parameter("sweep_execute", False)
    node.declare_parameter("sweep_return_home", True)
    node.declare_parameter("sweep_dwell_time", 1.0)
    # 空なら起動時の姿勢を初期姿勢として使う。
    # 空リストの既定値はBYTE_ARRAYと推論されDOUBLE_ARRAYの指定を弾くため、
    # 動的型付けにして型検査を外す。
    node.declare_parameter(
        "sweep_home_joint_degrees",
        [],
        ParameterDescriptor(dynamic_typing=True),
    )

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        code = sweep(node)
    except KeyboardInterrupt:
        code = 130
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(code)


if __name__ == "__main__":
    main()
