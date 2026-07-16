#!/usr/bin/env python3
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


CUBE_CHARUCO_DEFAULTS = {
    "squares_x": 4,
    "squares_y": 4,
    "square_length_m": 0.010,
    "marker_length_m": 0.007,
    "dictionary": "DICT_4X4_100",
}

CUBE_USED_MARKER_ID_RANGES = [
    (0, 7),
    (8, 15),
    (16, 23),
    (24, 31),
    (32, 39),
    (40, 47),
    (48, 55),
    (56, 63),
    (64, 71),
    (72, 79),
]

ARUCO_DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def normalize_dictionary_name(dictionary: str) -> str:
    if not dictionary:
        raise ValueError("dictionary must not be empty")

    name = str(dictionary).strip()
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"

    if name not in ARUCO_DICTIONARIES:
        known = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(
            f"Unknown ArUco dictionary '{dictionary}'. Known: {known}"
        )

    return name


def get_aruco_dictionary(dictionary: str):
    return cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARIES[normalize_dictionary_name(dictionary)]
    )


def dictionary_capacity(dictionary: str) -> int | None:
    name = normalize_dictionary_name(dictionary)
    try:
        return int(name.rsplit("_", 1)[1])
    except ValueError:
        return None


def mm_to_m(value_mm: float) -> float:
    return float(value_mm) / 1000.0


def m_to_mm(value_m: float) -> float:
    return float(value_m) * 1000.0


def marker_count_for_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary: str,
) -> int:
    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_m),
        float(marker_length_m),
        get_aruco_dictionary(dictionary),
    )
    return len(board.getIds())


def marker_ids_from_range(
    marker_id_start: int,
    marker_count: int,
) -> list[int]:
    return list(
        range(int(marker_id_start), int(marker_id_start) + int(marker_count))
    )


def create_charuco_board(config: dict):
    dictionary_name = normalize_dictionary_name(config["dictionary"])
    marker_ids = config.get("marker_ids")
    args = [
        (int(config["squares_x"]), int(config["squares_y"])),
        float(config["square_length_m"]),
        float(config["marker_length_m"]),
        get_aruco_dictionary(dictionary_name),
    ]
    if marker_ids is not None:
        args.append(np.array(marker_ids, dtype=np.int32))
    return cv2.aruco.CharucoBoard(*args)


def validate_board_config(config: dict) -> dict:
    normalized = dict(config)
    normalized["dictionary"] = normalize_dictionary_name(
        normalized["dictionary"]
    )
    normalized["squares_x"] = int(normalized["squares_x"])
    normalized["squares_y"] = int(normalized["squares_y"])
    normalized["square_length_m"] = float(normalized["square_length_m"])
    normalized["marker_length_m"] = float(normalized["marker_length_m"])

    if normalized["squares_x"] < 2 or normalized["squares_y"] < 2:
        raise ValueError("squares_x and squares_y must both be >= 2")
    if normalized["marker_length_m"] <= 0.0:
        raise ValueError("marker_length_m must be > 0")
    if normalized["square_length_m"] <= normalized["marker_length_m"]:
        raise ValueError("square_length_m must be larger than marker_length_m")

    marker_count = marker_count_for_board(
        normalized["squares_x"],
        normalized["squares_y"],
        normalized["square_length_m"],
        normalized["marker_length_m"],
        normalized["dictionary"],
    )
    normalized["marker_count"] = marker_count

    marker_ids = normalized.get("marker_ids")
    if marker_ids is None:
        marker_id_start = int(normalized.get("marker_id_start", 0))
        marker_ids = marker_ids_from_range(marker_id_start, marker_count)
    else:
        marker_ids = [int(marker_id) for marker_id in marker_ids]
        marker_id_start = min(marker_ids) if marker_ids else 0

    if len(marker_ids) != marker_count:
        raise ValueError(
            f"Board needs {marker_count} marker IDs, got {len(marker_ids)}"
        )
    if len(set(marker_ids)) != len(marker_ids):
        raise ValueError(
            f"Duplicated marker IDs in board config: {marker_ids}"
        )

    capacity = dictionary_capacity(normalized["dictionary"])
    if capacity is not None and (
        min(marker_ids) < 0 or max(marker_ids) >= capacity
    ):
        raise ValueError(
            f"Marker IDs {min(marker_ids)}-{max(marker_ids)} exceed "
            f"{normalized['dictionary']} capacity 0-{capacity - 1}"
        )

    normalized["marker_id_start"] = marker_id_start
    normalized["marker_ids"] = marker_ids
    return normalized


def board_config_from_cli(
    *,
    name: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary: str,
    marker_id_start: int,
) -> dict:
    config = {
        "name": name,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": mm_to_m(square_length_mm),
        "marker_length_m": mm_to_m(marker_length_mm),
        "dictionary": dictionary,
        "marker_id_start": marker_id_start,
    }
    return validate_board_config(config)


def load_board_config(path: str | os.PathLike) -> dict:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Board config must be a mapping: {path}")

    return validate_board_config(data)


def save_board_config(config: dict, path: str | os.PathLike) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = validate_board_config(config)

    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            yaml.safe_dump(data, f, sort_keys=False)
