from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_image_topic = LaunchConfiguration("depth_image_topic")
    base_frame = LaunchConfiguration("base_frame")
    cube_pose_cache_file = LaunchConfiguration("cube_pose_cache_file")
    world_tf_cache_file = LaunchConfiguration("world_tf_cache_file")
    publish_camera_link_from_cube_tf = LaunchConfiguration(
        "publish_camera_link_from_cube_tf"
    )
    cube_detect_service_name = LaunchConfiguration("cube_detect_service_name")
    cube_detection_window_sec = LaunchConfiguration("cube_detection_window_sec")
    camera_side = LaunchConfiguration("camera_side")
    find_cube_log_level = LaunchConfiguration("find_cube_log_level")
    pre_marker_detection_window_sec = LaunchConfiguration(
        "pre_marker_detection_window_sec"
    )
    move_backend = LaunchConfiguration("move_backend")
    cartesian_service_name = LaunchConfiguration("cartesian_service_name")

    return LaunchDescription([
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/hand_camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/hand_camera/color/camera_info",
        ),
        DeclareLaunchArgument(
            "depth_image_topic",
            default_value="/camera/hand_camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument("base_frame", default_value="link_base"),
        DeclareLaunchArgument(
            "cube_pose_cache_file",
            default_value="charuco_ros2/config/cube_target_poses.yaml",
        ),
        DeclareLaunchArgument(
            "world_tf_cache_file",
            default_value="charuco_ros2/config/world_camera_tfs.json",
        ),
        DeclareLaunchArgument(
            "publish_camera_link_from_cube_tf",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "cube_detect_service_name",
            default_value="/multi_cube_charuco_detector/detect_once",
        ),
        DeclareLaunchArgument("cube_detection_window_sec", default_value="2.0"),
        DeclareLaunchArgument("camera_side", default_value="left"),
        DeclareLaunchArgument("find_cube_log_level", default_value="warn"),
        DeclareLaunchArgument(
            "pre_marker_detection_window_sec",
            default_value="0.0",
        ),
        DeclareLaunchArgument("move_backend", default_value="moveit"),
        DeclareLaunchArgument("cartesian_service_name", default_value="auto"),

        Node(
            package="charuco_ros2",
            executable="multi_cube_charuco_detector",
            name="multi_cube_charuco_detector_node",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "world_tf_cache_file": world_tf_cache_file,
                "publish_camera_link_from_cube_tf": (
                    publish_camera_link_from_cube_tf
                ),
                "save_camera_link_from_cube_tf": True,
                "detect_service_name": cube_detect_service_name,
                "detection_window_sec": cube_detection_window_sec,
                "detect_continuously": False,
            }],
        ),
        Node(
            package="charuco_ros2",
            executable="find_cube",
            name="find_cube_node",
            output="screen",
            arguments=[
                "--ros-args",
                "--log-level",
                ["find_cube_node:=", find_cube_log_level],
            ],
            parameters=[{
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "depth_image_topic": depth_image_topic,
                "base_frame": base_frame,
                "cube_pose_cache_file": cube_pose_cache_file,
                "save_detected_pose": True,
            }],
        ),
        Node(
            package="charuco_ros2",
            executable="move_to_charuco",
            name="move_to_charuco_pose",
            output="screen",
            parameters=[{
                "camera_side": camera_side,
                "backend": move_backend,
                "cube_base_frame": base_frame,
                "cube_pose_cache_file": cube_pose_cache_file,
                "pre_marker_detection_window_sec": (
                    pre_marker_detection_window_sec
                ),
                "multi_cube_detect_service_name": cube_detect_service_name,
                "cartesian_service_name": cartesian_service_name,
            }],
        ),
    ])
