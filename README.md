# ChArUco Generator / Detector

## Files

- `detect_charuco.py`
  - Current ROS 2 ChArUco detector.
  - Supports automatic TF child frame assignment from detected ArUco marker IDs.
  - Publishes fixed camera-link TFs from detected ChArUco frames:
    - `left_camera_charuco` -> `left_camera_link`
    - `right_camera_charuco` -> `right_camera_link`
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
- `left_camera_charuco` -> `left_camera_link`
- `right_camera_charuco` -> `right_camera_link`

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

Optional detector nodes:

```bash
ros2 run charuco_ros2 charuco_detector_9_7
ros2 run charuco_ros2 apriltag_detector
```
