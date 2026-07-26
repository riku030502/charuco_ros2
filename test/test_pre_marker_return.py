from std_srvs.srv import Trigger

from charuco_ros2.move_to_charuco import MoveToCharucoPoseNode


class _ReturnHarness:
    handle_pre_marker_move_request = (
        MoveToCharucoPoseNode.handle_pre_marker_move_request
    )

    def __init__(self, success=True):
        self.success = success
        self.force = None
        self.find_cube_enabled = True

    def set_find_cube_detection_enabled(self, enabled, required=False):
        del required
        self.find_cube_enabled = enabled
        return True, "ok"

    def move_to_pre_marker_pose(self, force=False):
        self.force = force
        if self.success:
            return True, 0, "success"
        return False, -24, "execution failed"

    @staticmethod
    def set_trigger_response(response, success, message):
        response.success = success
        response.message = message
        return response


def test_pre_marker_service_forces_return_pose():
    harness = _ReturnHarness(success=True)

    response = harness.handle_pre_marker_move_request(
        Trigger.Request(),
        Trigger.Response(),
    )

    assert response.success is True
    assert response.message == "returned to pre-marker pose"
    assert harness.force is True
    assert harness.find_cube_enabled is False


def test_pre_marker_service_reports_motion_failure():
    harness = _ReturnHarness(success=False)

    response = harness.handle_pre_marker_move_request(
        Trigger.Request(),
        Trigger.Response(),
    )

    assert response.success is False
    assert "ret=-24" in response.message
    assert "execution failed" in response.message
