"""RGB intrinsic calibration and per-serial camera model selection (ARGUS Plan028).

ROS-independent modules: ``board`` (ChArUco board definition and detection),
``quality`` (per-pose admission and the data-sufficiency gate), ``solver``
(OpenCV calibration, bootstrap, held-out evaluation and model selection) and
``registry`` (validated model YAMLs keyed by serial and image profile).
``cli`` is the ``calibrate_intrinsics`` command.
"""
