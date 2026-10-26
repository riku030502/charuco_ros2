#!/usr/bin/env python3
"""calibrate_intrinsics: one-command RGB intrinsic calibration of a running ROS 2 camera.

    ros2 run charuco_ros2 calibrate_intrinsics --camera-name hand_camera

1. Audit (read-only): camera node, device-reported serial (RealSense
   ``device_info`` service, cross-checked with the ``serial_no`` parameter),
   image/CameraInfo topics and their publisher, raw (not rectified) image,
   CameraInfo K/D/size. Stops when the serial is unknown or inconsistent.
2. Guided capture in a localhost-only browser page: the operator moves the
   ChArUco board by hand; one PNG per independent still pose is saved.
3. When the live gate passes and the target count is reached (or when the
   operator presses "解析"), the offline gate, M0/M2/M3 calibration,
   held-out evaluation, selection and the candidate YAML are produced.
4. Registration into the writable registry needs a click on the page
   (and an explicit replace confirmation for an already registered key).

The tool only subscribes and reads parameters/services. It never changes the
camera profile, restarts nodes, publishes, or talks to a robot.

``--offline SESSION`` re-runs step 3 on a saved session without ROS.
"""
from __future__ import annotations

import os

# One worker (ARGUS experiment PC): OpenBLAS threads made the bootstrap 2.5x slower at
# 16x the CPU time. Must be set before numpy is imported.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import pipeline, solver  # noqa: E402
from .board import DEFAULT_BOARD_FILE, BoardDetector, CalibrationBoard  # noqa: E402
from .capture import Collector  # noqa: E402
from .registry import (  # noqa: E402
    ENV_VAR,
    RegistryError,
    find_model,
    normalise_serial,
    register,
    registry_dirs,
)

DEFAULT_SESSIONS = Path("~/.local/share/charuco_ros2/intrinsic_sessions")


def code_version() -> dict:
    package = Path(__file__).resolve().parents[2]
    info = {
        "package_dir": str(package),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    try:
        info["git_commit"] = subprocess.run(
            ["git", "-C", str(package), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(package), "status", "--porcelain", "--", "charuco_ros2", "setup.py"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        info["git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = ""
    return info


def log(message: str) -> None:
    print(f"[calibrate_intrinsics] {message}", flush=True)


class Session:
    """State shared by the ROS node, the analysis thread and the web page."""

    def __init__(self):
        self.collector = None
        self.phase = "audit"  # audit -> capture -> analysing -> analysed -> registered | failed
        self.log_lines: list = []
        self.report = None
        self.registration = None
        self.lock = threading.Lock()

    def note(self, message: str) -> None:
        log(message)
        with self.lock:
            self.log_lines = (self.log_lines + [time.strftime("%H:%M:%S ") + message])[-200:]


def audit(node, args, session: Session) -> dict:
    """Read-only identification of the camera. Raises RuntimeError on any doubt."""
    from rcl_interfaces.srv import GetParameters

    node_name = (
        args.node_name
        or f"/{args.camera_namespace.strip('/')}/{args.camera_name}".replace("//", "/")
    )
    deadline = time.time() + args.timeout
    names = []
    while time.time() < deadline:
        names = [f"{ns.rstrip('/')}/{n}" for n, ns in node.get_node_names_and_namespaces()]
        if node_name in names:
            break
        time.sleep(0.5)
    if node_name not in names:
        candidates = [n for n in names if args.camera_name in n or "camera" in n]
        raise RuntimeError(f"camera node {node_name} not found; candidates {candidates}")

    clients = node.__dict__.setdefault("_audit_clients", {})

    def client_for(kind, name):
        if name not in clients:
            clients[name] = node.create_client(kind, name)
        return clients[name]

    def call(client, request, what):
        if not client.wait_for_service(timeout_sec=args.timeout):
            return None
        future = client.call_async(request)
        end = time.time() + args.timeout
        while not future.done() and time.time() < end:
            time.sleep(0.05)
        if not future.done():
            raise RuntimeError(f"timed out {what}")
        return future.result()

    request = GetParameters.Request()
    request.names = [
        "serial_no",
        "rgb_camera.color_profile",
        "camera_name",
        "rgb_camera.enable_auto_exposure",
        "rgb_camera.exposure",
        "rgb_camera.gain",
    ]
    response = call(
        client_for(GetParameters, f"{node_name}/get_parameters"),
        request,
        "reading camera parameters",
    )
    params = {}
    if response is not None:
        for name, value in zip(request.names, response.values):
            params[name] = {
                1: value.bool_value,
                2: value.integer_value,
                3: value.double_value,
                4: value.string_value,
            }.get(value.type)
    device = {}
    try:
        from realsense2_camera_msgs.srv import DeviceInfo

        result = call(
            client_for(DeviceInfo, f"{node_name}/device_info"),
            DeviceInfo.Request(),
            "reading device_info",
        )
        if result is not None:
            device = {
                "device_name": result.device_name,
                "serial_number": result.serial_number,
                "firmware_version": result.firmware_version,
                "usb_type": result.usb_type_descriptor,
                "physical_port": result.physical_port,
            }
    except ImportError:
        session.note("realsense2_camera_msgs is not available; using the serial_no parameter only")
    reported = normalise_serial(device.get("serial_number", ""))
    requested = normalise_serial(params.get("serial_no") or "")
    if not reported and not requested:
        raise RuntimeError(
            "camera serial is unknown (no device_info, empty serial_no parameter); refusing"
        )
    if reported and requested and reported != requested:
        raise RuntimeError(
            f"device_info serial {reported} differs from the serial_no parameter {requested}"
        )
    serial = reported or requested
    if args.expected_serial and normalise_serial(args.expected_serial) != serial:
        raise RuntimeError(f"camera reports serial {serial}, expected {args.expected_serial}")
    image_topic = args.image_topic or f"{node_name}/color/image_raw"
    info_topic = args.camera_info_topic or f"{node_name}/color/camera_info"
    if "rect" in image_topic.split("/")[-1]:
        raise RuntimeError(
            f"{image_topic} looks rectified; calibration needs image_raw (no double correction)"
        )
    publishers = {}
    for topic in (image_topic, info_topic):
        end = time.time() + args.timeout
        found = []
        while time.time() < end and not found:
            found = [
                f"{p.node_namespace.rstrip('/')}/{p.node_name}"
                for p in node.get_publishers_info_by_topic(topic)
            ]
            if not found:
                time.sleep(0.3)
        if found != [node_name]:
            raise RuntimeError(
                f"{topic} must be published only by {node_name}; publishers {found}"
            )
        publishers[topic] = found
    return {
        "node_name": node_name,
        "serial_no": serial,
        "serial_source": "device_info" if reported else "serial_no parameter",
        "serial_no_parameter": params.get("serial_no"),
        "device": device,
        "parameters": params,
        "color_profile": params.get("rgb_camera.color_profile") or "",
        "image_topic": image_topic,
        "camera_info_topic": info_topic,
        "publishers": publishers,
    }


def wait_camera_size(node, audit_info, args) -> tuple:
    """First CameraInfo (width, height) on the audited topic."""
    from sensor_msgs.msg import CameraInfo

    seen = []
    subscription = node.create_subscription(
        CameraInfo, audit_info["camera_info_topic"], lambda m: seen.append((m.width, m.height)), 1
    )
    end = time.time() + args.timeout
    while not seen and time.time() < end:
        time.sleep(0.05)
    node.destroy_subscription(subscription)
    if not seen:
        raise RuntimeError(f"no CameraInfo on {audit_info['camera_info_topic']}")
    return int(seen[0][0]), int(seen[0][1])


def run_analysis(session: Session, path: Path, args, provenance: dict) -> None:
    session.phase = "analysing"
    try:
        report = pipeline.analyse(
            path, bootstrap_n=args.bootstrap, log=session.note, provenance=provenance
        )
        session.report = report
        session.phase = "analysed"
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in ("status", "selected", "reason", "heldout", "candidate_model")
                },
                indent=1,
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - report on the page and keep serving
        session.note(f"解析に失敗: {type(error).__name__}: {error}")
        session.phase = "capture"


def do_register(session: Session, path: Path, args, replace: bool) -> dict:
    report = session.report or json.loads(
        (path / "analysis/report.json").read_text(encoding="utf-8")
    )
    if report.get("candidate_status") != "VALIDATED":
        raise RegistryError(
            "NOT_VALIDATED", f"analysis status {report.get('status')}; nothing to register"
        )
    result = register(path / "candidate_model.yaml", args.registry, replace=replace)
    (path / "registration.json").write_text(
        json.dumps(dict(result, registered_unix=time.time()), indent=2) + "\n", encoding="utf-8"
    )
    session.registration = result
    session.phase = "registered"
    session.note(
        f"登録しました: {result['path']}"
        + (
            f"（旧登録を {result['replaced_backups']} へ退避）"
            if result["replaced_backups"]
            else ""
        )
    )
    return result


PAGE = Path(__file__).resolve().parent / "page.html"


def serve(session: Session, path: Path, args, provenance: dict, audit_info: dict) -> None:
    from flask import Flask, jsonify, request, send_file

    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        collector = session.collector
        body = collector.status() if collector else {}
        existing = None
        if collector and collector.camera:
            try:
                existing = find_model(
                    audit_info["serial_no"],
                    collector.camera["width"],
                    collector.camera["height"],
                    dirs=args.registry,
                )["path"]
            except RegistryError:
                existing = None
        body.update(
            phase=session.phase,
            log=session.log_lines[-30:],
            report=session.report,
            registration=session.registration,
            serial_no=audit_info["serial_no"],
            camera=audit_info["node_name"],
            registry=[str(d) for d in registry_dirs(args.registry)],
            existing_model=existing,
        )
        return jsonify(body)

    @app.get("/api/overlay")
    def overlay():
        data = session.collector.jpeg() if session.collector else None
        return (
            ("waiting for camera", 503)
            if data is None
            else send_file(BytesIO(data), mimetype="image/jpeg", max_age=0)
        )

    @app.post("/api/pause")
    def pause():
        session.collector.paused = not session.collector.paused
        return jsonify(paused=session.collector.paused)

    @app.post("/api/analyse")
    def analyse():
        if session.phase not in ("capture", "analysed"):
            return jsonify(error=f"phase {session.phase}"), 409
        start_analysis(session, path, args, provenance, audit_info)
        return jsonify(ok=True)

    @app.post("/api/register")
    def register_route():
        body = request.get_json(silent=True) or {}
        if session.phase != "analysed":
            return jsonify(error=f"phase {session.phase}"), 409
        try:
            return jsonify(do_register(session, path, args, replace=bool(body.get("replace"))))
        except RegistryError as error:
            return jsonify(error=str(error), code=error.code), 409

    log(f"open http://127.0.0.1:{args.port}  (localhost only)")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


def start_analysis(session, path, args, provenance, audit_info) -> None:
    collector = session.collector
    collector.paused = True
    if collector.error:
        session.note(f"撮影中のエラーのため解析しません: {collector.error}")
        return
    if session.recheck is not None:
        try:
            serial = session.recheck()
        except RuntimeError as error:
            collector.fail(f"解析前のserial再確認に失敗: {error}")
            return
        if serial != audit_info["serial_no"]:
            collector.fail(f"解析前のserial再確認で不一致: {serial} != {audit_info['serial_no']}")
            return
    threading.Thread(
        target=run_analysis, args=(session, path, args, provenance), daemon=True
    ).start()


def prepare_session(args, audit_info: dict, board: CalibrationBoard) -> Path:
    serial = audit_info["serial_no"]
    if args.resume:
        path = Path(args.resume).expanduser().resolve()
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if (
            meta["serial_no"] != serial
            or meta["board"]["sha256"] != board.sha256
            or meta["image_topic"] != audit_info["image_topic"]
        ):
            raise RuntimeError(
                f"--resume refused: {path} is for serial {meta['serial_no']} or other board/topic"
            )
        return path
    root = Path(args.output).expanduser() if args.output else DEFAULT_SESSIONS.expanduser()
    path = (root / f"{serial}_{args.camera_name}_{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    meta = {
        "kind": "rgb_intrinsic_capture",
        "camera_name": args.camera_name,
        "serial_no": serial,
        "device_name": audit_info["device"].get("device_name", ""),
        "audit": audit_info,
        "color_profile": audit_info["color_profile"],
        "image_topic": audit_info["image_topic"],
        "camera_info_topic": audit_info["camera_info_topic"],
        "image_processing": "raw",
        "board": {"path": board.path, **board.summary()},
        "target": args.target,
        "maximum": args.maximum,
        "operator_note": args.note,
        "started_unix": time.time(),
        "code": code_version(),
        "split_rule": "pose_index % 5 == 0 -> validation, else calibration (fixed at acceptance)",
        "selection_rule": solver.SELECTION,
    }
    (path / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def run_live(args) -> int:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    cv2.setNumThreads(1)
    board = CalibrationBoard.load(args.board)
    detector = BoardDetector(board)
    session = Session()
    session.recheck = None
    rclpy.init(args=None)
    node = Node("charuco_calibrate_intrinsics_read_only")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        audit_info = audit(node, args, session)
    except RuntimeError as error:
        log(f"STOP (audit): {error}")
        rclpy.shutdown()
        return 2
    session.note(
        f"camera {audit_info['node_name']} serial {audit_info['serial_no']} "
        f"({audit_info['serial_source']}), "
        f"profile {audit_info['color_profile']}, topic {audit_info['image_topic']}"
    )
    for name, value in audit_info["parameters"].items():
        session.note(f"  {name} = {value}")
    if args.audit_only:
        try:
            existing = find_model(
                audit_info["serial_no"],
                *wait_camera_size(node, audit_info, args),
                dirs=args.registry,
            )["path"]
        except (RegistryError, RuntimeError) as error:
            existing = str(error)
        print(json.dumps(dict(audit_info, registry_lookup=existing), indent=1, ensure_ascii=False))
        rclpy.shutdown()
        return 0
    path = prepare_session(args, audit_info, board)
    session.note(f"session: {path}")
    session.collector = Collector(path, detector, args.target, args.maximum)
    session.phase = "capture"
    provenance = {"code": code_version(), "session_dir": str(path)}

    def recheck() -> str:
        info = audit(node, args, session)
        return info["serial_no"]

    session.recheck = recheck
    bridge = CvBridge()
    state = {"info": None}

    def on_info(message) -> None:
        state["info"] = {
            "width": int(message.width),
            "height": int(message.height),
            "k": [float(v) for v in message.k],
            "d": [float(v) for v in message.d],
            "distortion_model": message.distortion_model,
            "frame_id": message.header.frame_id,
            "r": [float(v) for v in message.r],
            "p": [float(v) for v in message.p],
            "camera_info_topic": audit_info["camera_info_topic"],
            "image_topic": audit_info["image_topic"],
            "serial_no": audit_info["serial_no"],
        }

    def on_image(message) -> None:
        info = state["info"]
        if info is None or session.phase != "capture":
            return
        if (int(message.width), int(message.height)) != (info["width"], info["height"]):
            session.collector.fail(
                f"画像 {message.width}x{message.height} が CameraInfo "
                f"{info['width']}x{info['height']} と不一致"
            )
            return
        stamp = int(message.header.stamp.sec) + int(message.header.stamp.nanosec) * 1e-9
        session.collector.push(bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"), info, stamp)

    node.create_subscription(CameraInfo, audit_info["camera_info_topic"], on_info, 10)
    node.create_subscription(Image, audit_info["image_topic"], on_image, 2)

    def watcher() -> None:
        """Start the analysis when the live gate passes at the target; check serial each 15 s."""
        last = time.time()
        while True:
            time.sleep(1.0)
            collector = session.collector
            if session.phase == "capture" and not collector.error and time.time() - last > 15:
                last = time.time()
                try:
                    if recheck() != audit_info["serial_no"]:
                        collector.fail("撮影中にカメラのserialが変化しました")
                except RuntimeError as error:
                    collector.fail(f"カメラの再確認に失敗: {error}")
            if (
                session.phase == "capture"
                and not args.no_auto_analyse
                and len(collector.records) >= args.target
                and collector.live_gate().get("pass")
            ):
                session.note("目標枚数に達し、目安ゲートも合格したので解析を始めます")
                start_analysis(session, path, args, provenance, audit_info)

    threading.Thread(target=watcher, daemon=True).start()
    try:
        serve(session, path, args, provenance, audit_info)
    except KeyboardInterrupt:
        pass
    finally:
        session.collector.stop()
        rclpy.shutdown()
    return 0


def run_offline(args) -> int:
    path = Path(args.offline).expanduser().resolve()
    session = Session()
    report = pipeline.analyse(
        path,
        board_file=args.board if args.board != str(DEFAULT_BOARD_FILE) else None,
        bootstrap_n=args.bootstrap,
        log=session.note,
        provenance={"code": code_version(), "session_dir": str(path), "mode": "offline"},
    )
    session.report = report
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ("status", "selected", "reason", "heldout", "candidate_model")
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    if args.register_offline:
        if (
            input(
                f"register {path / 'candidate_model.yaml'} into "
                f"{registry_dirs(args.registry)[0]}? type yes: "
            ).strip()
            != "yes"
        ):
            print("not registered")
            return 0
        print(json.dumps(do_register(session, path, args, replace=args.replace), indent=1))
    return 0 if report.get("status") in ("PASS", "NO_IMPROVEMENT") else 3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_intrinsics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--camera-name", default="hand_camera")
    p.add_argument("--camera-namespace", default="camera")
    p.add_argument(
        "--node-name", default="", help="camera node (default /<namespace>/<camera-name>)"
    )
    p.add_argument("--image-topic", default="")
    p.add_argument("--camera-info-topic", default="")
    p.add_argument(
        "--expected-serial",
        default="",
        help="optional extra check against the device-reported serial",
    )
    p.add_argument("--board", default=str(DEFAULT_BOARD_FILE))
    p.add_argument(
        "--registry",
        default=None,
        help=f"registry dirs (os.pathsep/comma; first is written); default ${ENV_VAR} "
        "or ~/.config/charuco_ros2/camera_intrinsics",
    )
    p.add_argument("--output", default="", help=f"session root (default {DEFAULT_SESSIONS})")
    p.add_argument(
        "--resume",
        default="",
        help="continue a session directory (same serial, board and topic only)",
    )
    p.add_argument("--target", type=int, default=80)
    p.add_argument("--maximum", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=solver.BOOTSTRAP)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--note", default="", help="operator note stored in session.json")
    p.add_argument("--no-auto-analyse", action="store_true")
    p.add_argument("--audit-only", action="store_true", help="read-only camera audit, then exit")
    p.add_argument("--offline", default="", help="analyse a saved session without ROS")
    p.add_argument(
        "--register-offline",
        action="store_true",
        help="with --offline: ask on the terminal and register",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="with --register-offline: replace an existing entry (backed up)",
    )
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.offline:
        return run_offline(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
