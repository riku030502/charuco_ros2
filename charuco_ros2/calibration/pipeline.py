"""Offline analysis of one capture session.

re-detect -> gate -> M2/M3 -> held-out evaluation -> selection -> outputs.

Images are read one at a time (never all in RAM). OpenCV uses one thread.
Session layout (written by the capture step)::

    <session>/session.json  camera_info.json  poses.jsonl  poses/pose_000.png ...
    <session>/analysis/     corners.jsonl gate.json fits.json evaluation.json selection.json
                            report.json tables/ figures/
    <session>/candidate_model.yaml   (only when the gate passed)
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from charuco_ros2.robust_charuco import ippe_lm

from . import solver
from .board import BoardDetector, CalibrationBoard
from .quality import (
    GATE,
    admitted,
    board_tilt_deg,
    data_gate,
    normalised_radius,
    sharpness,
    split_of,
)
from .registry import write_model_yaml


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )


def _csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_session(session: Path) -> dict:
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    camera = json.loads((session / "camera_info.json").read_text(encoding="utf-8"))
    return {"session": meta, "camera_info": camera}


def redetect(
    session: Path, detector: BoardDetector, k0: np.ndarray, size: tuple, log=print
) -> list:
    """Re-detect every saved pose PNG with the same detector and K0/D0 (pose only for tilt)."""
    board = detector.board
    files = sorted((session / "poses").glob("pose_*.png"))
    records = []
    for n, path in enumerate(files, 1):
        data = path.read_bytes()
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or (gray.shape[1], gray.shape[0]) != tuple(size):
            raise RuntimeError(f"{path}: unreadable or size differs from CameraInfo {size}")
        ids, points = detector.detect(gray)
        index = int(path.stem.split("_")[1])
        row = {
            "pose_index": index,
            "file": path.name,
            "split": split_of(index),
            "png_sha256": hashlib.sha256(data).hexdigest(),
            "charuco_n": int(len(ids)),
            "charuco_ids": ids.tolist(),
            "charuco_points": points.tolist(),
            "tilt_deg": None,
            "distance_mm": None,
            "rmse_m0_px": None,
        }
        if len(ids) >= 4:
            pose = ippe_lm(detector.objects(ids), points, k0, np.zeros(5))
            if pose is not None:
                rotation, _ = cv2.Rodrigues(pose[0])
                row["tilt_deg"] = board_tilt_deg(pose[0])
                row["distance_mm"] = float(
                    np.linalg.norm(rotation @ board.centre_m + pose[1].reshape(3)) * 1000
                )
                row["rmse_m0_px"] = pose[2]
        if len(ids) >= 2:
            row.update(sharpness(gray, ids, points, board.inner_columns, board.inner_rows))
        else:
            row.update(
                {
                    "edges": 0,
                    "valid_fraction": 0.0,
                    "width_median_px": float("nan"),
                    "bright_ge_250_fraction": float("nan"),
                }
            )
        row["radius_max"] = float(normalised_radius(points, k0[:2, 2]).max()) if len(ids) else None
        row["admitted"] = admitted(row)
        records.append(row)
        del gray, data
        if n % 10 == 0 or n == len(files):
            log(f"re-detected {n}/{len(files)}")
    return records


def views_of(records: list, detector: BoardDetector) -> dict:
    views = {"calibration": [], "validation": []}
    for row in records:
        if admitted(row):
            views[row["split"]].append(
                {
                    "pose_index": row["pose_index"],
                    "obj": detector.objects(row["charuco_ids"]),
                    "img": np.asarray(row["charuco_points"], float),
                }
            )
    return views


def model_document(
    name: str,
    k,
    d,
    meta: dict,
    camera: dict,
    board: CalibrationBoard,
    metrics: dict,
    provenance: dict,
    status: str,
) -> dict:
    session = meta
    return {
        "schema_version": 1,
        "name": name,
        "status": status,
        "serial_no": str(session["serial_no"]),
        "camera_name": session.get("camera_name", ""),
        "device_name": session.get("device_name", ""),
        "stream": "color",
        "image_processing": "raw",
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "color_profile": session.get("color_profile", ""),
        "distortion_model": "plumb_bob",
        "K": np.asarray(k, float).reshape(3, 3).tolist(),
        "D": np.asarray(d, float).reshape(-1)[:5].tolist(),
        "camera_info_k": np.asarray(camera["k"], float).reshape(3, 3).tolist(),
        "camera_info_d": [float(v) for v in camera.get("d", [])],
        "camera_info_distortion_model": camera.get("distortion_model", ""),
        "image_topic": session.get("image_topic", ""),
        "camera_info_topic": session.get("camera_info_topic", ""),
        "board": board.summary(),
        "validation": metrics,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance,
        "source": f"charuco_ros2 calibrate_intrinsics session {provenance.get('session_dir', '')}",
    }


def figures(out: Path, records: list, k0, size, models: dict, evaluations: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    out.mkdir(parents=True, exist_ok=True)
    centre = np.asarray(k0, float)[:2, 2]
    written = []
    points = [
        np.asarray(r["charuco_points"], float).reshape(-1, 2) for r in records if r["admitted"]
    ]
    points = np.vstack(points) if points else np.zeros((0, 2))
    outer = (
        normalised_radius(points, centre) >= GATE["outer_radius"]
        if len(points)
        else np.zeros(0, bool)
    )
    fig, ax = plt.subplots(figsize=(8, 8 * size[1] / size[0] + 0.6))
    ax.scatter(points[~outer, 0], points[~outer, 1], s=2, color="#9aa5b1", label="r<0.70")
    ax.scatter(points[outer, 0], points[outer, 1], s=4, color="#d9822b", label="r≥0.70")
    ax.add_patch(
        plt.Circle(
            centre, GATE["outer_radius"] * np.linalg.norm(centre), fill=False, color="#d9822b"
        )
    )
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_aspect("equal")
    ax.set_title(f"採用姿勢のコーナー分布（外周 r≥0.70: {int(outer.sum())} 点）", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "corner_coverage.png", dpi=110)
    plt.close(fig)
    written.append("corner_coverage.png")
    colours = {"M0": "#9aa5b1", "M2": "#2a6fdb", "M3": "#14532d"}
    radii = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (k, d) in models.items():
        ax.plot(
            radii,
            solver.radial_curve(k, d, centre, radii),
            color=colours.get(name),
            linewidth=2,
            label=name,
        )
    ax.axvspan(GATE["outer_radius"], 1.0, color="#fde68a", alpha=0.35, linewidth=0)
    ax.set_xlabel("正規化画像半径 r（主点→画像隅=1）")
    ax.set_ylabel("歪みによる像の移動 [px]（外向き正）")
    ax.set_title("モデル別の放射方向歪み（主点→左上隅の対角線）", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "radial_distortion.png", dpi=130)
    plt.close(fig)
    written.append("radial_distortion.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [n for n in ("M0", "M2", "M3") if n in evaluations]
    axes[0].boxplot([evaluations[n]["per_view_rmse_px"] for n in names], labels=names)
    axes[0].set_ylabel("保持群の再投影RMSE [px]（姿勢ごと）")
    axes[0].set_title("独立検証（保持群）", fontsize=10)
    axes[0].grid(axis="y", alpha=0.25)
    bins = list(evaluations["M0"]["radial_bins"])
    for i, n in enumerate(names):
        values = [evaluations[n]["radial_bins"][b]["radial_mean_px"] for b in bins]
        axes[1].bar(
            np.arange(len(bins)) + (i - 1) * 0.27,
            [np.nan if v is None else v for v in values],
            0.26,
            color=colours.get(n),
            label=n,
        )
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(bins)), bins, fontsize=8)
    axes[1].set_xlabel("正規化半径の区間")
    axes[1].set_ylabel("平均放射残差 [px]（外向き正）")
    axes[1].set_title("半径別の系統残差（保持群）", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "heldout_residuals.png", dpi=130)
    plt.close(fig)
    written.append("heldout_residuals.png")
    return written


def analyse(
    session: Path, board_file=None, bootstrap_n: int = solver.BOOTSTRAP, log=print, provenance=None
) -> dict:
    """Run the full offline analysis of a session directory and write every output under it."""
    cv2.setNumThreads(1)
    session = Path(session).resolve()
    started = time.time()
    loaded = load_session(session)
    meta, camera = loaded["session"], loaded["camera_info"]
    board = CalibrationBoard.load(board_file or meta["board"]["path"])
    if board.sha256 != meta["board"]["sha256"]:
        raise RuntimeError(
            f"board file {board.path} differs from the capture board ({meta['board']['sha256']})"
        )
    detector = BoardDetector(board)
    size = (int(camera["width"]), int(camera["height"]))
    k0 = np.asarray(camera["k"], float).reshape(3, 3)
    d0 = np.asarray(camera.get("d") or [0.0] * 5, float).reshape(-1)
    if d0.size != 5 or np.any(d0 != 0.0):
        raise RuntimeError(
            f"CameraInfo D {d0.tolist()} is not five zeros; M0 is not the factory pinhole"
        )
    out = session / "analysis"
    log("phase: re-detect")
    records = redetect(session, detector, k0, size, log)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corners.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    _csv(
        out / "tables/poses.csv",
        [
            {k: v for k, v in r.items() if k not in ("charuco_ids", "charuco_points")}
            for r in records
        ],
    )
    gate = data_gate(records, size[0], size[1], k0[:2, 2])
    _json(out / "gate.json", gate)
    provenance = dict(
        provenance or {},
        session_dir=str(session),
        poses_jsonl_sha256=hashlib.sha256((session / "poses.jsonl").read_bytes()).hexdigest(),
        png_sha256_manifest_sha256=hashlib.sha256(
            "".join(r["png_sha256"] for r in records).encode()
        ).hexdigest(),
    )
    report = {
        "session": str(session),
        "serial_no": meta["serial_no"],
        "camera_name": meta.get("camera_name"),
        "size": list(size),
        "board": board.summary(),
        "gate": {
            "pass": gate["pass"],
            "admitted": gate["admitted"],
            "calibration": gate["calibration"],
            "validation": gate["validation"],
            "total_records": gate["total_records"],
        },
        "provenance": provenance,
    }
    if not gate["pass"]:
        failed = [name for name, item in gate["checks"].items() if not item["pass"]]
        report.update(
            status="INSUFFICIENT_DATA",
            selected="M0",
            failed_gate_rules=failed,
            elapsed_s=time.time() - started,
        )
        _json(out / "report.json", report)
        log(f"INSUFFICIENT_DATA: failed gate rules {failed}; no calibration")
        return report
    views = views_of(records, detector)
    fits, boots, stab, finite = {}, {}, {}, {}
    for name, flags in solver.FLAGS.items():
        log(f"phase: calibrate {name} on {len(views['calibration'])} poses")
        fits[name] = solver.calibrate(views["calibration"], flags, size, k0)
        finite[name] = bool(np.all(np.isfinite(solver.parameters(fits[name]))))
        boots[name] = solver.bootstrap(
            views["calibration"],
            flags,
            size,
            k0,
            bootstrap_n,
            progress=lambda i, n, name=name: (i % 25 == 0 or i == n)
            and log(f"bootstrap {name} {i}/{n}"),
        )
        stab[name] = solver.stability(fits[name], boots[name])
    models = {"M0": (k0, np.zeros(5)), **{n: (f["K"], f["D"]) for n, f in fits.items()}}
    evaluations = {
        n: solver.evaluate(views["validation"], k, d, k0[:2, 2]) for n, (k, d) in models.items()
    }
    selection = solver.select(evaluations, stab, finite)
    fit_rows = []
    for name, fit in fits.items():
        fit_rows.append(
            {
                "model": name,
                "flags": int(solver.FLAGS[name]),
                "calibration_views": len(views["calibration"]),
                "rms_px": fit["rms_px"],
                **dict(zip(solver.LABELS, solver.parameters(fit).tolist())),
                **{f"std_{lab}": float(v) for lab, v in zip(solver.LABELS, fit["std"])},
                **{k: v for k, v in stab[name].items() if not isinstance(v, dict)},
            }
        )
    _csv(out / "tables/fits.csv", fit_rows)
    _csv(
        out / "tables/heldout_evaluation.csv",
        [
            {
                "model": n,
                "views": e["views"],
                "pnp_failed": e["pnp_failed"],
                "rmse_median_px": e["rmse_median_px"],
                "rmse_p95_px": e["rmse_p95_px"],
            }
            for n, e in evaluations.items()
        ],
    )
    _csv(
        out / "tables/heldout_radial_bins.csv",
        [
            {"model": n, "radius_bin": b, **v}
            for n, e in evaluations.items()
            for b, v in e["radial_bins"].items()
        ],
    )
    _json(
        out / "fits.json",
        {
            n: {
                "K": f["K"].tolist(),
                "D": f["D"].tolist(),
                "rms_px": f["rms_px"],
                "std": f["std"].tolist(),
                "stability": stab[n],
            }
            for n, f in fits.items()
        },
    )
    _json(out / "bootstrap.json", {n: b.tolist() for n, b in boots.items()})
    _json(out / "evaluation.json", evaluations)
    _json(out / "selection.json", selection)
    written = figures(out / "figures", records, k0, size, models, evaluations)
    chosen = selection["selected"]
    metrics = {
        "selection_status": selection["status"],
        "selected": chosen,
        "reason": selection["reason"],
        "gate_pass": True,
        "admitted_poses": gate["admitted"],
        "calibration_poses": gate["calibration"],
        "validation_poses": gate["validation"],
        "heldout_rmse_median_px": {n: e["rmse_median_px"] for n, e in evaluations.items()},
        "heldout_rmse_p95_px": {n: e["rmse_p95_px"] for n, e in evaluations.items()},
        "note": "reprojection RMSE and stability are not absolute pose accuracy",
    }
    if chosen != "M0":
        metrics["stability"] = {k: v for k, v in stab[chosen].items() if not isinstance(v, dict)}
    status = (
        "VALIDATED" if selection["status"] in ("PASS", "NO_IMPROVEMENT") else selection["status"]
    )
    k, d = models[chosen]
    document = model_document(chosen, k, d, meta, camera, board, metrics, provenance, status)
    candidate = session / "candidate_model.yaml"
    if candidate.exists():
        candidate.rename(session / f"candidate_model.previous_{int(time.time())}.yaml")
    write_model_yaml(candidate, document)
    report.update(
        status=selection["status"],
        selected=chosen,
        reason=selection["reason"],
        candidates=selection["candidates"],
        heldout=metrics["heldout_rmse_median_px"],
        heldout_p95=metrics["heldout_rmse_p95_px"],
        parameters={
            n: dict(zip(solver.LABELS, solver.parameters(f).tolist())) for n, f in fits.items()
        },
        stability={
            n: {k: v for k, v in s.items() if not isinstance(v, dict)} for n, s in stab.items()
        },
        candidate_model=str(candidate),
        candidate_status=status,
        figures=written,
        bootstrap_n=bootstrap_n,
        elapsed_s=time.time() - started,
    )
    _json(out / "report.json", report)
    log(f"{selection['status']}: selected {chosen} ({selection['reason']})")
    return report
