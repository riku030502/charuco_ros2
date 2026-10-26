"""Registry of validated RGB intrinsic models keyed by camera serial and image profile (no ROS).

A registry is a list of directories searched in order. Each model is one YAML
file; the key is ``(serial_no, width, height, stream, image_processing)``.
Lookup fails closed: no entry, two entries for one key in the same directory,
a corrupt file or a non-VALIDATED status raise :class:`RegistryError`.
Directory order: the CLI ``--registry`` / ROS parameter, else the
``CHARUCO_ROS2_INTRINSICS_REGISTRY`` environment variable (``os.pathsep``
separated), else ``~/.config/charuco_ros2/camera_intrinsics``. The first
directory is the one the calibration command writes to.

The files stay readable by ``multi_cube_charuco_detector.load_pose_camera_model``
(name, width, height, K, D, camera_info_k, serial_no, source).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

ENV_VAR = "CHARUCO_ROS2_INTRINSICS_REGISTRY"
DEFAULT_DIR = Path("~/.config/charuco_ros2/camera_intrinsics")
SCHEMA_VERSION = 1
REQUIRED = ("name", "serial_no", "width", "height", "K", "D", "camera_info_k")


class RegistryError(RuntimeError):
    """Model lookup failed. code: NOT_REGISTERED, AMBIGUOUS, CORRUPT, NOT_VALIDATED or EXISTS."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def normalise_serial(value) -> str:
    """Serial as reported by the device: quotes, spaces and the launch-time leading '_' removed."""
    return str(value or "").strip().strip("'\" ").lstrip("_")


def registry_dirs(value=None) -> list:
    """Resolve the search path: None/"" (default), a path, an os.pathsep/comma string or a list."""
    if value in (None, "", []):
        value = os.environ.get(ENV_VAR, "") or str(DEFAULT_DIR)
    if isinstance(value, (str, Path)):
        value = [part for chunk in str(value).split(os.pathsep) for part in chunk.split(",")]
    return [Path(str(v).strip()).expanduser() for v in value if str(v).strip()]


def model_filename(
    serial_no: str, width: int, height: int, stream: str = "color", processing: str = "raw"
) -> str:
    return f"{serial_no}_{stream}_{int(width)}x{int(height)}_{processing}.yaml"


def parse_model(path) -> dict:
    """Load and validate one model file (legacy files without stream/processing: color/raw)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(f"missing {missing}")
        model = {
            "name": str(data["name"]),
            "serial_no": normalise_serial(data["serial_no"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "stream": str(data.get("stream", "color")),
            "image_processing": str(data.get("image_processing", "raw")),
            "distortion_model": str(data.get("distortion_model", "plumb_bob")),
            "K": np.asarray(data["K"], np.float64).reshape(3, 3),
            "D": np.asarray(data["D"], np.float64).reshape(-1),
            "camera_info_k": np.asarray(data["camera_info_k"], np.float64).reshape(3, 3),
            "status": str(data.get("status", "VALIDATED" if "source" in data else "")),
            "source": str(data.get("source", "")),
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except Exception as error:  # noqa: BLE001 - any parse failure is a corrupt entry
        raise RegistryError("CORRUPT", f"{path}: {error}") from error
    if not model["serial_no"] or model["width"] <= 0 or model["height"] <= 0:
        raise RegistryError("CORRUPT", f"{path}: empty serial_no or invalid size")
    if model["distortion_model"] != "plumb_bob" or len(model["D"]) != 5:
        raise RegistryError("CORRUPT", f"{path}: only plumb_bob with 5 coefficients is supported")
    if not (np.all(np.isfinite(model["K"])) and np.all(np.isfinite(model["D"]))):
        raise RegistryError("CORRUPT", f"{path}: non-finite K/D")
    return model


def find_model(
    serial_no: str,
    width: int,
    height: int,
    stream: str = "color",
    processing: str = "raw",
    dirs=None,
) -> dict:
    """Return the validated model for the key; the first directory holding the key wins."""
    serial_no = normalise_serial(serial_no)
    if not serial_no:
        raise RegistryError("NOT_REGISTERED", "camera serial_no is unknown")
    searched = registry_dirs(dirs)
    for directory in searched:
        if not directory.is_dir():
            continue
        matches = []
        for path in sorted(directory.glob("*.yaml")):
            model = parse_model(path)
            if (
                model["serial_no"],
                model["width"],
                model["height"],
                model["stream"],
                model["image_processing"],
            ) == (serial_no, int(width), int(height), stream, processing):
                matches.append(model)
        if len(matches) > 1:
            raise RegistryError(
                "AMBIGUOUS",
                f"{len(matches)} models for {serial_no} {width}x{height} {stream}/{processing} "
                f"in {directory}: {[Path(m['path']).name for m in matches]}",
            )
        if matches:
            if not matches[0]["status"].startswith("VALIDATED"):
                raise RegistryError(
                    "NOT_VALIDATED", f"{matches[0]['path']} status '{matches[0]['status']}'"
                )
            return matches[0]
    raise RegistryError(
        "NOT_REGISTERED",
        f"no model for serial {serial_no} {width}x{height} {stream}/{processing} in "
        f"{[str(d) for d in searched]}",
    )


def models_for_serial(serial_no: str, dirs=None) -> list:
    """All parseable entries of one serial (diagnostics only, no selection)."""
    out = []
    for directory in registry_dirs(dirs):
        if directory.is_dir():
            for path in sorted(directory.glob("*.yaml")):
                try:
                    model = parse_model(path)
                except RegistryError:
                    continue
                if model["serial_no"] == normalise_serial(serial_no):
                    out.append(model)
    return out


def write_model_yaml(path, document: dict) -> None:
    """Write a model document (plain types). Never overwrites."""
    path = Path(path)
    if path.exists():
        raise RegistryError("EXISTS", f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# RGB intrinsic model {document['name']} for serial {document['serial_no']} "
        f"{document['width']}x{document['height']} {document.get('stream', 'color')}/"
        f"{document.get('image_processing', 'raw')}.\n"
        "# image_raw is NOT undistorted; pass these K/D with the observed 2D points to PnP.\n"
    )
    path.write_text(
        header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def register(candidate_path, dirs=None, replace: bool = False) -> dict:
    """Copy a validated candidate YAML into the first registry directory.

    An existing entry for the same key is never overwritten silently: it needs
    ``replace=True`` and is first moved to ``<dir>/history/`` with a timestamp.
    """
    model = parse_model(candidate_path)
    if not model["status"].startswith("VALIDATED"):
        raise RegistryError(
            "NOT_VALIDATED", f"{candidate_path} status '{model['status']}' cannot be registered"
        )
    directory = registry_dirs(dirs)[0]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / model_filename(
        model["serial_no"],
        model["width"],
        model["height"],
        model["stream"],
        model["image_processing"],
    )
    key = (
        model["serial_no"],
        model["width"],
        model["height"],
        model["stream"],
        model["image_processing"],
    )
    existing = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            other = parse_model(path)
        except RegistryError:
            continue
        if (
            other["serial_no"],
            other["width"],
            other["height"],
            other["stream"],
            other["image_processing"],
        ) == key:
            existing.append(path)
    backups = []
    if existing:
        if not replace:
            raise RegistryError(
                "EXISTS",
                f"{[str(p) for p in existing]} already registered for {key}; "
                "explicit replacement required",
            )
        history = directory / "history"
        history.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for path in existing:
            backup = history / f"{path.stem}.replaced_{stamp}.yaml"
            shutil.move(str(path), str(backup))
            backups.append(str(backup))
    shutil.copyfile(str(candidate_path), str(target))
    registered = parse_model(target)
    return {"path": str(target), "sha256": registered["sha256"], "replaced_backups": backups}
