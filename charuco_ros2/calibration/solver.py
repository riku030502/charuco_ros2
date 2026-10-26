"""OpenCV intrinsic calibration (M2/M3), bootstrap, held-out evaluation and frozen model selection.

Generalised from ARGUS Plan020 ``calibrate.py``: the reference K0 and the image
size are the camera's own CameraInfo values, never another unit's. ``SELECTION``
is fixed before any data are seen (ARGUS plan/history/028/method_spec.json).
"""

from __future__ import annotations

import cv2
import numpy as np

from charuco_ros2.robust_charuco import ippe_lm

from .quality import normalised_radius

CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
FIX_K = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
FLAGS = {"M2": FIX_K, "M3": cv2.CALIB_USE_INTRINSIC_GUESS}
LABELS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
RADIUS_BINS = (0.0, 0.25, 0.5, 0.7, 0.85, 1.01)
BOOTSTRAP = 200
SELECTION = {
    "min_validation_views": 8,
    "max_relative_std_f": 0.01,  # calibrateCameraExtended std(fx), std(fy) / value (M3)
    "max_bootstrap_width_f": 0.02,  # 95 % bootstrap width of fx, fy / value (M3)
    "max_bootstrap_width_k1": 0.10,  # 95 % bootstrap width of k1 (M2, M3)
    "min_median_reduction_vs_m0": 0.20,  # held-out RMSE median must fall by >= 20 % vs M0
    "p95_not_worse_than_m0": True,
    "tie_relative": 0.02,  # M3 must beat M2 by > 2 % of M2's median, else the simpler M2
}


def calibrate(views: list, flags: int, size: tuple, k0: np.ndarray) -> dict:
    """views: [{"obj": (N,3), "img": (N,2)}]; size = (width, height)."""
    obj = [np.asarray(v["obj"], np.float32) for v in views]
    img = [np.asarray(v["img"], np.float32).reshape(-1, 1, 2) for v in views]
    rms, k, d, _, _, std, _, per_view = cv2.calibrateCameraExtended(
        obj,
        img,
        tuple(int(v) for v in size),
        np.asarray(k0, float).copy(),
        np.zeros(5),
        flags=flags,
        criteria=CRITERIA,
    )
    return {
        "rms_px": float(rms),
        "K": k,
        "D": d.reshape(-1)[:5],
        "std": std.reshape(-1)[:9],
        "per_view_px": per_view.reshape(-1),
    }


def parameters(fit: dict) -> np.ndarray:
    k = fit["K"]
    return np.concatenate(
        [[k[0, 0], k[1, 1], k[0, 2], k[1, 2]], np.asarray(fit["D"]).reshape(-1)[:5]]
    )


def bootstrap(
    views: list, flags: int, size, k0, n: int = BOOTSTRAP, seed: int = 0, progress=None
) -> np.ndarray:
    """Pose-level resampling with replacement; returns (n, 9) parameters."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(
            parameters(
                calibrate(
                    [views[j] for j in rng.integers(0, len(views), len(views))], flags, size, k0
                )
            )
        )
        if progress:
            progress(i + 1, n)
    return np.asarray(out).reshape(-1, len(LABELS))


def evaluate(views: list, k: np.ndarray, d: np.ndarray, principal_point) -> dict:
    """Per-view IPPE->LM re-fit with (K, D): RMSE median/p95 and mean radial residual per bin."""
    rmse, radius, radial, failed = [], [], [], 0
    k = np.asarray(k, float)
    d = np.asarray(d, float).reshape(-1)
    for v in views:
        obj, img = np.asarray(v["obj"], float), np.asarray(v["img"], float)
        pose = ippe_lm(obj, img, k, d)
        if pose is None:
            failed += 1
            continue
        projected = cv2.projectPoints(obj, pose[0], pose[1], k, d)[0].reshape(-1, 2)
        residual = img - projected
        rmse.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
        outward = img - k[:2, 2]
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
        radius.extend(normalised_radius(img, principal_point))
        radial.extend(np.sum(residual * outward, axis=1))
    radius, radial = np.asarray(radius), np.asarray(radial)
    bins = {}
    for low, high in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:]):
        mask = (radius >= low) & (radius < high)
        bins[f"{low:.2f}-{min(high, 1.0):.2f}"] = {
            "n": int(mask.sum()),
            "radial_mean_px": float(radial[mask].mean()) if mask.any() else None,
        }
    return {
        "views": len(rmse),
        "pnp_failed": failed,
        "rmse_median_px": float(np.median(rmse)) if rmse else None,
        "rmse_p95_px": float(np.percentile(rmse, 95)) if rmse else None,
        "per_view_rmse_px": rmse,
        "radial_bins": bins,
    }


def radial_curve(k, d, principal_point, radii) -> list:
    """Displacement (px, outward positive) of (K, D) along the principal point -> (0, 0) line."""
    centre = np.asarray(principal_point, float).reshape(2)
    k = np.asarray(k, float)
    pixels = centre[None, :] - np.asarray(radii, float)[:, None] * centre[None, :]
    rays = np.column_stack(
        [
            (pixels[:, 0] - k[0, 2]) / k[0, 0],
            (pixels[:, 1] - k[1, 2]) / k[1, 1],
            np.ones(len(pixels)),
        ]
    )
    distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, np.asarray(d, float))[
        0
    ].reshape(-1, 2)
    delta = distorted - pixels
    return (
        np.linalg.norm(delta, axis=1) * np.sign(np.sum(delta * (pixels - centre), axis=1) + 1e-12)
    ).tolist()


def stability(fit: dict, boots: np.ndarray) -> dict:
    p = parameters(fit)
    low, high = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    width = high - low
    return {
        "relative_std_fx": float(fit["std"][0] / p[0]),
        "relative_std_fy": float(fit["std"][1] / p[1]),
        "bootstrap_width_rel_fx": float(width[0] / p[0]),
        "bootstrap_width_rel_fy": float(width[1] / p[1]),
        "bootstrap_width_k1": float(width[4]),
        "bootstrap_p2_5": dict(zip(LABELS, low.tolist())),
        "bootstrap_p97_5": dict(zip(LABELS, high.tolist())),
    }


def select(evaluations: dict, stabilities: dict, finite: dict) -> dict:
    """Frozen rule. Returns {"selected": "M0"|"M2"|"M3", "status": ..., "candidates": {...}}."""
    s = SELECTION
    m0 = evaluations["M0"]
    out = {"candidates": {}, "rule": dict(s)}
    if m0["views"] < s["min_validation_views"]:
        out.update(
            selected="M0",
            status="INSUFFICIENT_DATA",
            reason=f"held-out views {m0['views']} < {s['min_validation_views']}",
        )
        return out
    acceptable = []
    for name in ("M2", "M3"):
        e, st, reasons = evaluations[name], stabilities[name], []
        if not finite[name]:
            reasons.append("non-finite parameters")
        if e["views"] < m0["views"]:
            reasons.append("PnP failed on held-out views")
        if name == "M3":
            if max(st["relative_std_fx"], st["relative_std_fy"]) >= s["max_relative_std_f"]:
                reasons.append("relative std of fx/fy >= 1 %")
            if (
                max(st["bootstrap_width_rel_fx"], st["bootstrap_width_rel_fy"])
                >= s["max_bootstrap_width_f"]
            ):
                reasons.append("bootstrap width of fx/fy >= 2 %")
        if st["bootstrap_width_k1"] >= s["max_bootstrap_width_k1"]:
            reasons.append("bootstrap width of k1 >= 0.10")
        if (
            e["rmse_median_px"] is None
            or e["rmse_median_px"] > (1 - s["min_median_reduction_vs_m0"]) * m0["rmse_median_px"]
        ):
            reasons.append("held-out RMSE median reduction vs M0 < 20 %")
        if s["p95_not_worse_than_m0"] and (
            e["rmse_p95_px"] is None or e["rmse_p95_px"] > m0["rmse_p95_px"]
        ):
            reasons.append("held-out RMSE p95 worse than M0")
        out["candidates"][name] = {"acceptable": not reasons, "reasons": reasons}
        if not reasons:
            acceptable.append(name)
    if not acceptable:
        out.update(
            selected="M0",
            status="NO_IMPROVEMENT",
            reason="no candidate met every rule; CameraInfo K/D (M0) is kept",
        )
    elif acceptable == ["M2", "M3"]:
        m2, m3 = evaluations["M2"]["rmse_median_px"], evaluations["M3"]["rmse_median_px"]
        chosen = "M3" if m3 < (1 - s["tie_relative"]) * m2 else "M2"
        out.update(
            selected=chosen,
            status="PASS",
            reason=f"both acceptable; M3 {m3:.4f} vs M2 {m2:.4f} px "
            f"(tie {s['tie_relative']:.0%} -> M2)",
        )
    else:
        out.update(
            selected=acceptable[0], status="PASS", reason=f"only {acceptable[0]} acceptable"
        )
    return out
