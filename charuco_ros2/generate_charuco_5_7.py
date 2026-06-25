#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# =========================
# ChArUco board parameters
# =========================
squares_x = 7          # 横方向のマス数
squares_y = 5          # 縦方向のマス数

square_length_mm = 10  # 1マス 10 mm
marker_length_mm = 7   # マーカ部分 7 mm

# 印刷解像度
dpi = 300

# =========================
# Margin / colored border
# =========================
# ChArUco本体と色付き縁の間の白い余白
# 今回は白余白なし
white_margin_mm = 0

# 色付き縁の太さ
color_border_mm = 3

# 出力先
output_dir = Path("boards/generated")

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# detect_charuco.py の board_id_configs と対応させる
# border_color_rgb は PIL 用なので RGB 指定
board_configs = [
    {
        "name": "left_camera_charuco",
        "min_id": 0,
        "max_id": 16,
        "border_color_rgb": (255, 0, 0),  # 赤
    },
    {
        "name": "right_camera_charuco",
        "min_id": 30,
        "max_id": 46,
        "border_color_rgb": (0, 0, 255),  # 青
    },
]


def mm_to_px(mm):
    return int(round(mm / 25.4 * dpi))


def generate_board(config):
    marker_ids = np.arange(
        config["min_id"],
        config["max_id"] + 1,
        dtype=np.int32
    )

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        aruco_dict,
        marker_ids
    )

    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm

    board_width_px = mm_to_px(board_width_mm)
    board_height_px = mm_to_px(board_height_mm)

    white_margin_px = mm_to_px(white_margin_mm)
    color_border_px = mm_to_px(color_border_mm)

    # ChArUco本体のみの画像サイズ
    charuco_width_px = board_width_px + 2 * white_margin_px
    charuco_height_px = board_height_px + 2 * white_margin_px

    # ChArUco本体を生成
    # white_margin_mm = 0 なので、marginSize=0 になる
    charuco_img = board.generateImage(
        (charuco_width_px, charuco_height_px),
        marginSize=white_margin_px,
        borderBits=1
    )

    # OpenCV出力はグレースケールなので、PILのRGB画像に変換
    pil_img = Image.fromarray(charuco_img).convert("RGB")

    # PILで色付き縁を追加
    pil_img = ImageOps.expand(
        pil_img,
        border=color_border_px,
        fill=config["border_color_rgb"]
    )

    output_base = (
        f"{config['name']}_7x5_square10_marker7_"
        f"id{config['min_id']}-{config['max_id']}_"
        f"margin{white_margin_mm}_border{color_border_mm}_300dpi"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{output_base}.png"
    output_pdf = output_dir / f"{output_base}.pdf"

    pil_img.save(output_png, dpi=(dpi, dpi))
    pil_img.save(output_pdf, "PDF", resolution=dpi)

    total_width_mm = board_width_mm + 2 * white_margin_mm + 2 * color_border_mm
    total_height_mm = board_height_mm + 2 * white_margin_mm + 2 * color_border_mm

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")
    print(f"Board name: {config['name']}")
    print(f"Marker IDs: {config['min_id']} - {config['max_id']}")
    print(f"Board size: {board_width_mm} mm x {board_height_mm} mm")
    print(
        "Image size with white margin: "
        f"{board_width_mm + 2 * white_margin_mm} mm x "
        f"{board_height_mm + 2 * white_margin_mm} mm"
    )
    print(
        "Image size with color border: "
        f"{total_width_mm} mm x {total_height_mm} mm"
    )
    print(f"Pixel size: {pil_img.width} px x {pil_img.height} px")
    print(f"Border color RGB: {config['border_color_rgb']}")
    print("")


def main():
    reference_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        aruco_dict
    )
    expected_marker_count = len(reference_board.getIds())

    for config in board_configs:
        marker_count = config["max_id"] - config["min_id"] + 1
        if marker_count != expected_marker_count:
            raise ValueError(
                f"{config['name']} needs {expected_marker_count} marker IDs, "
                f"but range {config['min_id']}-{config['max_id']} has "
                f"{marker_count}"
            )

        generate_board(config)


if __name__ == "__main__":
    main()