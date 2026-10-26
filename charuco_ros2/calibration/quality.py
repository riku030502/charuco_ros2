"""Per-pose admission and the data-sufficiency gate for intrinsic calibration (no ROS).

Generalised from ARGUS Plan019 (edge sharpness) and Plan020 (gate). Image size and
the reference principal point come from the camera's own CameraInfo; nothing is
tied to one serial or resolution. Thresholds live in ``GATE`` and must not be
relaxed after data are seen.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import cv2
import numpy as np

GATE = {
    "corners_min": 12,
    "edge_width_max_px": 4.5,
    "edge_valid_fraction_min": 0.8,
    "admitted_poses_min": 40,
    "cell_grid": (8, 6),
    "cell_occupancy_min": 0.50,
    "outer_radius": 0.70,
    "outer_corners_min": 50,
    "outer_poses_min": 5,
    "outer_max_single_pose_share": 0.40,
    "outer_quadrant_corners_min": 5,
    "outer_quadrants_min": 3,
    "region_grid": (3, 3),
    "region_poses_min": 2,
    "frontal_deg": 15.0,
    "frontal_poses_min": 5,
    "tilted_deg": 30.0,
    "tilted_poses_min": 10,
    "extreme_corner_radius": 0.85,
}
# Plan019 edge profile
STEP = 0.25
HALF_FRACTION = 0.12
POSITIONS = (0.35, 0.50, 0.65)
MIN_CONTRAST = 20.0


def split_of(pose_index: int) -> str:
    """Fixed pose-level split, assigned at acceptance time: every 5th pose is held out (80:20)."""
    return "validation" if pose_index % 5 == 0 else "calibration"


def normalised_radius(points, principal_point) -> np.ndarray:
    """|p - c| / |c| with c the reference principal point (1 = image corner for a centred c)."""
    centre = np.asarray(principal_point, float).reshape(2)
    return np.linalg.norm(np.asarray(points, float).reshape(-1, 2) - centre, axis=1) / float(
        np.linalg.norm(centre)
    )


def quadrant(points, principal_point) -> np.ndarray:
    centre = np.asarray(principal_point, float).reshape(2)
    points = np.asarray(points, float).reshape(-1, 2)
    return (points[:, 0] >= centre[0]).astype(int) + 2 * (points[:, 1] >= centre[1]).astype(int)


def cells(points, width: int, height: int, grid=GATE["cell_grid"]) -> set:
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return set()
    return set(
        zip(
            np.clip((points[:, 0] / width * grid[0]).astype(int), 0, grid[0] - 1).tolist(),
            np.clip((points[:, 1] / height * grid[1]).astype(int), 0, grid[1] - 1).tolist(),
        )
    )


def board_tilt_deg(rvec) -> float:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return float(np.degrees(np.arccos(min(1.0, abs(float(rotation[2, 2]))))))


def rotation_deg(rvec_a, rvec_b) -> float:
    a, _ = cv2.Rodrigues(np.asarray(rvec_a, float).reshape(3, 1))
    b, _ = cv2.Rodrigues(np.asarray(rvec_b, float).reshape(3, 1))
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def novel(rvec, tvec, accepted: Iterable, min_translation_m=0.030, min_rotation_deg=8.0) -> bool:
    """True when (rvec, tvec) differs from every accepted (rvec, tvec) by >= 30 mm or >= 8 deg."""
    tvec = np.asarray(tvec, float).reshape(3)
    return all(
        np.linalg.norm(tvec - np.asarray(t, float).reshape(3)) >= min_translation_m
        or rotation_deg(rvec, r) >= min_rotation_deg
        for r, t in accepted
    )


def displacement(previous: Mapping | None, ids, points) -> float:
    """Median motion (px) of corners common to the previous analysed frame (inf if < 8)."""
    if not previous:
        return float("inf")
    old = {int(i): np.asarray(p, float) for i, p in zip(previous["ids"], previous["points"])}
    moved = [
        np.linalg.norm(old[int(i)] - np.asarray(p, float))
        for i, p in zip(ids, points)
        if int(i) in old
    ]
    return float(np.median(moved)) if len(moved) >= 8 else float("inf")


def grid_neighbours(ids, columns: int, rows: int) -> list:
    present = {int(v) for v in ids}
    pairs = []
    for item in sorted(present):
        row, column = divmod(item, columns)
        if column + 1 < columns and item + 1 in present:
            pairs.append((item, item + 1))
        if row + 1 < rows and item + columns in present:
            pairs.append((item, item + columns))
    return pairs


def _profile(gray, start, end):
    segment = end - start
    length = float(np.linalg.norm(segment))
    normal = np.array([-segment[1], segment[0]]) / length
    half = HALF_FRACTION * length
    offsets = np.arange(-half, half + 1e-9, STEP)
    profiles = []
    for t in POSITIONS:
        points = start + t * segment + offsets[:, None] * normal
        profiles.append(
            cv2.remap(
                gray,
                points[:, 0].astype(np.float32).reshape(1, -1),
                points[:, 1].astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1)
        )
    return offsets, np.mean(profiles, axis=0).astype(np.float64)


def edge_width(offsets, profile) -> dict:
    """10-90 % transition width (px) of a contrast-normalised edge profile (ARGUS Plan019)."""
    if profile[:4].mean() > profile[-4:].mean():
        profile = profile[::-1]
    dark, bright = float(profile[:4].mean()), float(profile[-4:].mean())
    contrast = bright - dark
    output = {
        "contrast": contrast,
        "bright": bright,
        "width_10_90_px": float("nan"),
        "valid": False,
    }
    if contrast < MIN_CONTRAST:
        return output
    q = (profile - dark) / contrast
    peak = int(np.argmax(np.diff(q)))
    low = peak
    while low > 0 and q[low] > 0.1:
        low -= 1
    high = peak + 1
    while high < len(q) - 1 and q[high] < 0.9:
        high += 1
    if q[low] > 0.1 or q[high] < 0.9:
        return output

    def crossing(i, j, level):
        if q[j] == q[i]:
            return float(offsets[i])
        return float(offsets[i] + (level - q[i]) * (offsets[j] - offsets[i]) / (q[j] - q[i]))

    s10, s90 = crossing(low, low + 1, 0.1), crossing(high - 1, high, 0.9)
    output["width_10_90_px"] = s90 - s10
    output["valid"] = bool(s90 > s10)
    return output


def sharpness(gray, ids, points, columns: int, rows: int) -> dict:
    """Median 10-90 edge width of grid-adjacent corners, valid and near-saturated fractions."""
    gray = gray.astype(np.float32)
    index = {int(v): k for k, v in enumerate(ids)}
    points = np.asarray(points, float).reshape(-1, 2)
    rows_out = [
        edge_width(*_profile(gray, points[index[a]], points[index[b]]))
        for a, b in grid_neighbours(ids, columns, rows)
    ]
    widths = [row["width_10_90_px"] for row in rows_out if row["valid"]]
    return {
        "edges": len(rows_out),
        "valid_fraction": len(widths) / len(rows_out) if rows_out else 0.0,
        "width_median_px": float(np.median(widths)) if widths else float("nan"),
        "bright_ge_250_fraction": (
            float(np.mean([row["bright"] >= 250 for row in rows_out]))
            if rows_out
            else float("nan")
        ),
    }


def admitted(record: Mapping) -> bool:
    width = record.get("width_median_px")
    return bool(
        record["charuco_n"] >= GATE["corners_min"]
        and width is not None
        and np.isfinite(width)
        and width <= GATE["edge_width_max_px"]
        and record["valid_fraction"] >= GATE["edge_valid_fraction_min"]
    )


def data_gate(records: list, width: int, height: int, principal_point) -> dict:
    """Pre-registered data-sufficiency gate on (re-detected) pose records.

    Each record needs pose_index, charuco_n, charuco_points, width_median_px,
    valid_fraction and tilt_deg (None when no pose).
    """
    g = GATE
    poses = [r for r in records if admitted(r)]
    calibration = [r for r in poses if split_of(r["pose_index"]) == "calibration"]
    occupied = set()
    for r in poses:
        occupied |= cells(r["charuco_points"], width, height)
    total_cells = g["cell_grid"][0] * g["cell_grid"][1]
    outer_counts, outer_quadrants, extreme = {}, np.zeros(4, int), np.zeros(4, int)
    for r in calibration:
        pts = np.asarray(r["charuco_points"], float).reshape(-1, 2)
        radius = normalised_radius(pts, principal_point)
        mask = radius >= g["outer_radius"]
        if mask.any():
            outer_counts[r["pose_index"]] = int(mask.sum())
            outer_quadrants += np.bincount(quadrant(pts[mask], principal_point), minlength=4)
        far = radius >= g["extreme_corner_radius"]
        if far.any():
            extreme += np.bincount(quadrant(pts[far], principal_point), minlength=4)
    outer_total = int(sum(outer_counts.values()))
    share = max(outer_counts.values()) / outer_total if outer_total else 1.0
    regions = {}
    for r in poses:
        for cell in cells(r["charuco_points"], width, height, grid=g["region_grid"]):
            regions.setdefault(cell, set()).add(r["pose_index"])
    gx, gy = g["region_grid"]
    tilts = [r["tilt_deg"] for r in poses if r.get("tilt_deg") is not None]
    checks = {
        "admitted_poses": {"value": len(poses), "pass": len(poses) >= g["admitted_poses_min"]},
        "cell_occupancy_8x6": {
            "value": len(occupied) / total_cells,
            "pass": len(occupied) / total_cells >= g["cell_occupancy_min"],
        },
        "outer_corners_calibration": {
            "value": outer_total,
            "pass": outer_total >= g["outer_corners_min"],
        },
        "outer_poses": {
            "value": len(outer_counts),
            "pass": len(outer_counts) >= g["outer_poses_min"],
        },
        "outer_max_single_pose_share": {
            "value": share,
            "pass": bool(outer_total) and share <= g["outer_max_single_pose_share"],
        },
        "outer_quadrants_ge5": {
            "value": outer_quadrants.tolist(),
            "pass": int(np.sum(outer_quadrants >= g["outer_quadrant_corners_min"]))
            >= g["outer_quadrants_min"],
        },
        "regions_3x3_ge2_poses": {
            "value": {
                f"{x},{y}": len(regions.get((x, y), ())) for y in range(gy) for x in range(gx)
            },
            "pass": all(
                len(regions.get((x, y), ())) >= g["region_poses_min"]
                for y in range(gy)
                for x in range(gx)
            ),
        },
        "frontal_poses_lt15deg": {
            "value": int(sum(t < g["frontal_deg"] for t in tilts)),
            "pass": sum(t < g["frontal_deg"] for t in tilts) >= g["frontal_poses_min"],
        },
        "tilted_poses_ge30deg": {
            "value": int(sum(t >= g["tilted_deg"] for t in tilts)),
            "pass": sum(t >= g["tilted_deg"] for t in tilts) >= g["tilted_poses_min"],
        },
    }
    report = {
        "extreme_corners_r085_by_quadrant": extreme.tolist(),
        "note": "quadrants 0=top-left 1=top-right 2=bottom-left 3=bottom-right; report only",
    }
    return {
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "admitted": len(poses),
        "calibration": len(calibration),
        "validation": len(poses) - len(calibration),
        "total_records": len(records),
        "report": report,
    }
