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
    camera_side = LaunchConfiguration("camera_side")
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
        DeclareLaunchArgument("camera_side", default_value="left"),
        DeclareLaunchArgument("move_backend", default_value="topic"),
        DeclareLaunchArgument("cartesian_service_name", default_value="auto"),

        Node(
            package="charuco_ros2",
            executable="detect_charuco",
            name="charuco_detector_node",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
            }],
        ),
        Node(
            package="charuco_ros2",
            executable="find_cube",
            name="find_cube_node",
            output="screen",
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
                "cartesian_service_name": cartesian_service_name,
            }],
        ),
    ])
