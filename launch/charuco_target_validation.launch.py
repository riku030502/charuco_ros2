from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    board_config_path = LaunchConfiguration("board_config_path")
    base_frame = LaunchConfiguration("base_frame")
    csv_path = LaunchConfiguration("csv_path")

    ext_image_topic = LaunchConfiguration("ext_image_topic")
    ext_camera_info_topic = LaunchConfiguration("ext_camera_info_topic")
    ext_parent_frame = LaunchConfiguration("ext_parent_frame")
    ext_output_frame_id = LaunchConfiguration("ext_output_frame_id")

    hand_image_topic = LaunchConfiguration("hand_image_topic")
    hand_camera_info_topic = LaunchConfiguration("hand_camera_info_topic")
    hand_parent_frame = LaunchConfiguration("hand_parent_frame")
    hand_output_frame_id = LaunchConfiguration("hand_output_frame_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument("board_config_path", default_value=""),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("csv_path", default_value=""),
            DeclareLaunchArgument(
                "ext_image_topic",
                default_value="/left_camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "ext_camera_info_topic",
                default_value="/left_camera/color/camera_info",
            ),
            DeclareLaunchArgument("ext_parent_frame", default_value=""),
            DeclareLaunchArgument(
                "ext_output_frame_id",
                default_value="target_pose_ext_left",
            ),
            DeclareLaunchArgument(
                "hand_image_topic",
                default_value="/camera/hand_camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "hand_camera_info_topic",
                default_value="/camera/hand_camera/color/camera_info",
            ),
            DeclareLaunchArgument(
                "hand_parent_frame",
                default_value="hand_camera_color_optical_frame",
            ),
            DeclareLaunchArgument(
                "hand_output_frame_id",
                default_value="target_pose_hand",
            ),
            Node(
                package="charuco_ros2",
                executable="charuco_target_detector",
                name="charuco_target_detector_ext",
                output="screen",
                parameters=[
                    {
                        "image_topic": ext_image_topic,
                        "camera_info_topic": ext_camera_info_topic,
                        "board_config_path": board_config_path,
                        "parent_frame": ext_parent_frame,
                        "output_frame_id": ext_output_frame_id,
                        "pose_topic": "/charuco_validation/ext_pose",
                        "debug_image_topic": (
                            "/charuco_validation/ext_debug_image"
                        ),
                    }
                ],
            ),
            Node(
                package="charuco_ros2",
                executable="charuco_target_detector",
                name="charuco_target_detector_hand",
                output="screen",
                parameters=[
                    {
                        "image_topic": hand_image_topic,
                        "camera_info_topic": hand_camera_info_topic,
                        "board_config_path": board_config_path,
                        "parent_frame": hand_parent_frame,
                        "output_frame_id": hand_output_frame_id,
                        "pose_topic": "/charuco_validation/hand_pose",
                        "debug_image_topic": (
                            "/charuco_validation/hand_debug_image"
                        ),
                    }
                ],
            ),
            Node(
                package="charuco_ros2",
                executable="charuco_pose_comparator",
                name="charuco_pose_comparator",
                output="screen",
                parameters=[
                    {
                        "input_mode": "tf",
                        "base_frame": base_frame,
                        "ext_target_frame": ext_output_frame_id,
                        "hand_target_frame": hand_output_frame_id,
                        "csv_path": csv_path,
                    }
                ],
            ),
        ]
    )
