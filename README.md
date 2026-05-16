# ChArUco Generator / Detector

## Files

- `detect_charuco.py`
  - Current ROS 2 ChArUco detector.
  - Supports automatic TF child frame assignment from detected ArUco marker IDs.
  - Publishes fixed camera-link TFs from detected ChArUco frames:
    - `left_camera_charuco` -> `left_camera_link`
    - `right_camera_charuco` -> `right_camera_link`
    - These are only published when `publish_world_camera_tf:=false`.
  - Also publishes world-frame camera-link TFs when `world` can be looked up:
    - `world` -> `left_camera_link`
    - `world` -> `right_camera_link`
    - RealSense publishes the internal camera frames from each `*_camera_link`.
  - Saves detected world-frame camera TFs and reloads them at startup.
    - Default cache: `charuco_ros2/config/world_camera_tfs.json`
    - When detection fails, saved TFs are published instead.
    - Offset: `x=0.052 m`, `y=0.067 m`, `z=0.0 m` from the ChArUco frame.
    - Rotation: `z=90 deg`, then `x=90 deg`, then `y=180 deg` from the ChArUco frame.
- `generate_charuco_5_7.py`
  - Current generator for two 7x5 ChArUco boards:
    - `left_camera_charuco`, marker IDs `0-16`
    - `right_camera_charuco`, marker IDs `30-46`
- `detect_apriltag.py`
  - AprilTag detector script.
- `detect_charuco_9_7.py`
  - Older 9x7 ChArUco detector variant kept for reference.
- `generate_charuco_9_7.py`
  - Older 9x7 ChArUco generator variant kept for reference.

## Output Directories

- `boards/generated/`
  - Current generated boards used by `detect_charuco.py`.
- `boards/legacy/`
  - Older generated board images/PDFs kept for reference.

## Generate Current Boards

```bash
python3 generate_charuco_5_7.py
```

## Run ROS 2 Nodes

```bash
colcon build --packages-select charuco_ros2
source install/setup.bash

ros2 run charuco_ros2 charuco_detector
```

The detector publishes:

- image frame -> `left_camera_charuco` or `right_camera_charuco`
- `left_camera_charuco` -> `left_camera_link` when `publish_world_camera_tf:=false`
- `right_camera_charuco` -> `right_camera_link` when `publish_world_camera_tf:=false`
- `world` -> `left_camera_link`
- `world` -> `right_camera_link`

With RealSense `publish_tf:=true`, the camera driver publishes internal frames such as:

- `left_camera_link` -> `left_camera_depth_optical_frame`
- `left_camera_link` -> `left_camera_color_optical_frame`
- `right_camera_link` -> `right_camera_depth_optical_frame`
- `right_camera_link` -> `right_camera_color_optical_frame`

The fixed camera-link offset and rotation can be changed with ROS 2 parameters:

```bash
ros2 run charuco_ros2 charuco_detector --ros-args \
  -p camera_link_offset_x:=0.052 \
  -p camera_link_offset_y:=0.067 \
  -p camera_link_offset_z:=0.0 \
  -p camera_link_rotation_z_deg:=90.0 \
  -p camera_link_rotation_x_deg:=90.0 \
  -p camera_link_rotation_y_deg:=180.0
```

World-frame TF publishing can be changed with ROS 2 parameters:

```bash
ros2 run charuco_ros2 charuco_detector --ros-args \
  -p world_frame:=world \
  -p publish_world_camera_tf:=true \
  -p world_tf_cache_file:=charuco_ros2/config/world_camera_tfs.json \
  -p saved_world_tf_publish_rate:=10.0
```

On startup, the detector loads saved TFs from `world_tf_cache_file`.
When a ChArUco board is detected, the detected `world` TF overwrites the cache.
When detection fails, the detector logs the failure reason and publishes the saved TFs.

Check the published world camera TFs:

```bash
ros2 run tf2_ros tf2_echo world left_camera_link
ros2 run tf2_ros tf2_echo world left_camera_depth_optical_frame
ros2 run tf2_ros tf2_echo world right_camera_link
ros2 run tf2_ros tf2_echo world right_camera_depth_optical_frame
```

Optional detector nodes:

```bash
ros2 run charuco_ros2 charuco_detector_9_7
ros2 run charuco_ros2 apriltag_detector
```
