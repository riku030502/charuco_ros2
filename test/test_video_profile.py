import pytest

from charuco_ros2.multi_cube_charuco_detector import (
    MultiCharucoDetectorNode,
    parse_video_profile,
)


class _Logger:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _ProfileHarness:
    prepare_detection_color_profile = (
        MultiCharucoDetectorNode.prepare_detection_color_profile
    )

    def __init__(self, profile, enabled=True):
        self.state = {"profile": profile, "enabled": enabled}
        self.detection_color_profile = "1280x720x30"
        self.detection_profile = (1280, 720, 30)
        self.detection_profile_size = (1280, 720)
        self.color_profile_settle_sec = 0.0
        self.applied = []
        self.waited_for = []
        self.logger = _Logger()

    def get_realsense_color_state(self):
        return dict(self.state)

    def apply_realsense_color_state(self, profile, enabled):
        self.applied.append((profile, enabled))

    def wait_for_camera_info_size(self, width, height):
        self.waited_for.append((width, height))

    def get_logger(self):
        return self.logger


@pytest.mark.parametrize(
    "profile, expected",
    [
        ("1280x720x30", (1280, 720, 30)),
        ("1280X720X30", (1280, 720, 30)),
        ("1280,720,30", (1280, 720, 30)),
        (" 424 x 240 x 30 ", (424, 240, 30)),
        ("0,0,0", (0, 0, 0)),
    ],
)
def test_parse_video_profile(profile, expected):
    assert parse_video_profile(profile) == expected


@pytest.mark.parametrize(
    "profile",
    [
        "",
        "1280x720",
        "1280x720x30xRGB8",
        "wide",
        "-1x720x30",
    ],
)
def test_parse_video_profile_rejects_invalid_values(profile):
    with pytest.raises(ValueError):
        parse_video_profile(profile)


def test_detection_profile_is_unchanged_when_full_profile_matches():
    harness = _ProfileHarness("1280x720x30")

    assert harness.prepare_detection_color_profile() is None
    assert harness.applied == []
    assert harness.waited_for == []


def test_detection_profile_switches_when_only_fps_differs():
    harness = _ProfileHarness("1280x720x15")

    original = harness.prepare_detection_color_profile()

    assert original == {
        "profile": "1280x720x15",
        "enabled": True,
    }
    assert harness.applied == [("1280x720x30", True)]
    assert harness.waited_for == [(1280, 720)]
