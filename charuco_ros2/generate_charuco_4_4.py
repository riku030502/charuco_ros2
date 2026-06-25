#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


# =========================
# ChArUco board parameters
# =========================
squares_x = 4          # 横方向のマス数
squares_y = 4          # 縦方向のマス数

square_length_mm = 10  # 1マス 10 mm
marker_length_mm = 7   # マーカ部分 7 mm

# 印刷解像度
dpi = 300

# =========================
# Margin / colored border
# =========================
# ChArUco本体と色付き縁の間の白い余白
white_margin_mm = 0

# 色付き縁の太さ
# 今回は 2 mm
color_border_mm = 2

# まとめ画像内で、各面の間に入れる余白
face_gap_mm = 5

# まとめ画像全体の外側余白
sheet_margin_mm = 5

# face名を画像上に描くか
draw_face_label = True

# 出力先
output_dir = Path("boards/generated/cube_charuco_sheets")

# ArUco dictionary
# 今回は 10面 × 8ID = 80 ID 必要なので DICT_4X4_100 で足りる
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

# 面の並び
face_names = ["front", "left", "right", "back", "top"]


# =========================
# Cube configs
# =========================
# left_cube: 赤
# right_cube: 青
#
# 1面あたり8個のmarker IDを使う。
# 5面 × 2立方体 = 10面、すべてユニークIDにする。
cube_configs = [
    {
        "cube_name": "left_cube",
        "border_color_rgb": (255, 0, 0),  # 赤
        "faces": [
            {
                "face_name": "front",
                "name": "left_cube_front_charuco",
                "min_id": 0,
                "max_id": 7,
            },
            {
                "face_name": "left",
                "name": "left_cube_left_charuco",
                "min_id": 8,
                "max_id": 15,
            },
            {
                "face_name": "right",
                "name": "left_cube_right_charuco",
                "min_id": 16,
                "max_id": 23,
            },
            {
                "face_name": "back",
                "name": "left_cube_back_charuco",
                "min_id": 24,
                "max_id": 31,
            },
            {
                "face_name": "top",
                "name": "left_cube_top_charuco",
                "min_id": 32,
                "max_id": 39,
            },
        ],
    },
    {
        "cube_name": "right_cube",
        "border_color_rgb": (0, 0, 255),  # 青
        "faces": [
            {
                "face_name": "front",
                "name": "right_cube_front_charuco",
                "min_id": 40,
                "max_id": 47,
            },
            {
                "face_name": "left",
                "name": "right_cube_left_charuco",
                "min_id": 48,
                "max_id": 55,
            },
            {
                "face_name": "right",
                "name": "right_cube_right_charuco",
                "min_id": 56,
                "max_id": 63,
            },
            {
                "face_name": "back",
                "name": "right_cube_back_charuco",
                "min_id": 64,
                "max_id": 71,
            },
            {
                "face_name": "top",
                "name": "right_cube_top_charuco",
                "min_id": 72,
                "max_id": 79,
            },
        ],
    },
]


def mm_to_px(mm: float, minimum: int = 0) -> int:
    """mmをpixelに変換する。"""
    px = int(round(mm / 25.4 * dpi))
    return max(minimum, px)


def create_charuco_face(
    face_config: dict,
    border_color_rgb: tuple[int, int, int],
) -> Image.Image:
    """1面ぶんのChArUco画像を生成してPIL Imageで返す。"""

    marker_ids = np.arange(
        face_config["min_id"],
        face_config["max_id"] + 1,
        dtype=np.int32,
    )

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        aruco_dict,
        marker_ids,
    )

    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm

    board_width_px = mm_to_px(board_width_mm, minimum=1)
    board_height_px = mm_to_px(board_height_mm, minimum=1)

    white_margin_px = mm_to_px(white_margin_mm, minimum=0)
    color_border_px = mm_to_px(color_border_mm, minimum=1)

    charuco_width_px = board_width_px + 2 * white_margin_px
    charuco_height_px = board_height_px + 2 * white_margin_px

    charuco_img = board.generateImage(
        (charuco_width_px, charuco_height_px),
        marginSize=white_margin_px,
        borderBits=1,
    )

    pil_img = Image.fromarray(charuco_img).convert("RGB")

    # 色付き縁を追加
    pil_img = ImageOps.expand(
        pil_img,
        border=color_border_px,
        fill=border_color_rgb,
    )

    if draw_face_label:
        draw = ImageDraw.Draw(pil_img)

        label = (
            f"{face_config['face_name']}  "
            f"ID:{face_config['min_id']}-{face_config['max_id']}"
        )

        text_x = color_border_px + 5
        text_y = color_border_px + 5

        bbox = draw.textbbox((text_x, text_y), label)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(255, 255, 255),
        )
        draw.text(
            (text_x, text_y),
            label,
            fill=(0, 0, 0),
        )

    return pil_img


def create_cube_sheet(cube_config: dict) -> Image.Image:
    """
    1つの立方体ぶん、5面をまとめた画像を作る。

    配置:
              [ top ]

    [ left ] [ front ] [ right ] [ back ]
    """

    border_color_rgb = cube_config["border_color_rgb"]

    face_image_dict = {
        face_config["face_name"]: create_charuco_face(
            face_config,
            border_color_rgb,
        )
        for face_config in cube_config["faces"]
    }

    for face_name in face_names:
        if face_name not in face_image_dict:
            raise ValueError(
                f"{cube_config['cube_name']} does not have face: {face_name}"
            )

    face_width_px = face_image_dict["front"].width
    face_height_px = face_image_dict["front"].height

    gap_px = mm_to_px(face_gap_mm, minimum=1)
    sheet_margin_px = mm_to_px(sheet_margin_mm, minimum=1)

    # 横4枚、縦2段
    sheet_width_px = 4 * face_width_px + 3 * gap_px + 2 * sheet_margin_px
    sheet_height_px = 2 * face_height_px + gap_px + 2 * sheet_margin_px

    sheet_img = Image.new(
        "RGB",
        (sheet_width_px, sheet_height_px),
        (255, 255, 255),
    )

    positions = {
        "top": (
            sheet_margin_px + face_width_px + gap_px,
            sheet_margin_px,
        ),
        "left": (
            sheet_margin_px,
            sheet_margin_px + face_height_px + gap_px,
        ),
        "front": (
            sheet_margin_px + face_width_px + gap_px,
            sheet_margin_px + face_height_px + gap_px,
        ),
        "right": (
            sheet_margin_px + 2 * (face_width_px + gap_px),
            sheet_margin_px + face_height_px + gap_px,
        ),
        "back": (
            sheet_margin_px + 3 * (face_width_px + gap_px),
            sheet_margin_px + face_height_px + gap_px,
        ),
    }

    for face_name, position in positions.items():
        sheet_img.paste(face_image_dict[face_name], position)

    # 左上に立方体名を描画
    if draw_face_label:
        draw = ImageDraw.Draw(sheet_img)
        cube_label = cube_config["cube_name"]

        label_x = sheet_margin_px
        label_y = sheet_margin_px

        bbox = draw.textbbox((label_x, label_y), cube_label)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(255, 255, 255),
        )
        draw.text(
            (label_x, label_y),
            cube_label,
            fill=(0, 0, 0),
        )

    return sheet_img


def validate_configs() -> None:
    """ID数、重複、辞書範囲を確認する。"""

    reference_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        aruco_dict,
    )

    expected_marker_count = len(reference_board.getIds())

    used_ids = []

    for cube_config in cube_configs:
        if len(cube_config["faces"]) != len(face_names):
            raise ValueError(
                f"{cube_config['cube_name']} needs {len(face_names)} faces, "
                f"but has {len(cube_config['faces'])}"
            )

        for face_config in cube_config["faces"]:
            marker_count = face_config["max_id"] - face_config["min_id"] + 1

            if marker_count != expected_marker_count:
                raise ValueError(
                    f"{face_config['name']} needs {expected_marker_count} marker IDs, "
                    f"but range {face_config['min_id']}-{face_config['max_id']} has "
                    f"{marker_count}"
                )

            ids = list(range(face_config["min_id"], face_config["max_id"] + 1))
            used_ids.extend(ids)

    duplicated_ids = sorted(
        {marker_id for marker_id in used_ids if used_ids.count(marker_id) > 1}
    )

    if duplicated_ids:
        raise ValueError(
            f"Duplicated marker IDs found: {duplicated_ids}"
        )

    max_dictionary_id = 99  # DICT_4X4_100 は 0〜99

    if max(used_ids) > max_dictionary_id:
        raise ValueError(
            f"Marker ID {max(used_ids)} exceeds dictionary limit "
            f"0-{max_dictionary_id}."
        )


def print_detection_config_example() -> None:
    """detect_charuco.py側に貼るための設定例を表示する。"""

    print("# =========================")
    print("# Example board_id_configs")
    print("# =========================")
    print("board_id_configs = [")

    for cube_config in cube_configs:
        for face_config in cube_config["faces"]:
            print("    {")
            print(f'        "name": "{face_config["name"]}",')
            print(f'        "cube_name": "{cube_config["cube_name"]}",')
            print(f'        "face_name": "{face_config["face_name"]}",')
            print(f'        "min_id": {face_config["min_id"]},')
            print(f'        "max_id": {face_config["max_id"]},')
            print("    },")

    print("]")


def save_cube_sheet(cube_config: dict) -> None:
    """1つの立方体ぶんのまとめ画像を保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    sheet_img = create_cube_sheet(cube_config)

    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm

    face_width_mm = board_width_mm + 2 * white_margin_mm + 2 * color_border_mm
    face_height_mm = board_height_mm + 2 * white_margin_mm + 2 * color_border_mm

    # 横4枚、縦2段
    sheet_width_mm = 4 * face_width_mm + 3 * face_gap_mm + 2 * sheet_margin_mm
    sheet_height_mm = 2 * face_height_mm + face_gap_mm + 2 * sheet_margin_mm

    min_id = min(face["min_id"] for face in cube_config["faces"])
    max_id = max(face["max_id"] for face in cube_config["faces"])

    output_base = (
        f"{cube_config['cube_name']}_"
        f"5faces_sheet_net_"
        f"charuco_{squares_x}x{squares_y}_"
        f"square{square_length_mm}_marker{marker_length_mm}_"
        f"id{min_id}-{max_id}_"
        f"margin{white_margin_mm}_border{color_border_mm}_"
        f"gap{face_gap_mm}_sheetmargin{sheet_margin_mm}_{dpi}dpi"
    )

    output_png = output_dir / f"{output_base}.png"
    output_pdf = output_dir / f"{output_base}.pdf"

    sheet_img.save(output_png, dpi=(dpi, dpi))
    sheet_img.save(output_pdf, "PDF", resolution=dpi)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")
    print(f"Cube name: {cube_config['cube_name']}")
    print(f"Marker IDs: {min_id} - {max_id}")
    print(f"One face size: {face_width_mm:.2f} mm x {face_height_mm:.2f} mm")
    print(f"Sheet size: {sheet_width_mm:.2f} mm x {sheet_height_mm:.2f} mm")
    print(f"Pixel size: {sheet_img.width} px x {sheet_img.height} px")
    print(f"Border color RGB: {cube_config['border_color_rgb']}")
    print("")


def main() -> None:
    validate_configs()

    for cube_config in cube_configs:
        save_cube_sheet(cube_config)

    print_detection_config_example()


if __name__ == "__main__":
    main()