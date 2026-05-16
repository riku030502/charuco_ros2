#!/usr/bin/env python3
from pathlib import Path

import cv2
from PIL import Image

# =========================
# ChArUco board parameters
# =========================
squares_x = 9          # 横方向のマス数
squares_y = 7          # 縦方向のマス数

square_length_mm = 10  # 1マス 10 mm
marker_length_mm = 7   # マーカ部分 7 mm

# 印刷解像度
dpi = 300

# 余白
margin_mm = 5

# 出力先
output_dir = Path("boards/legacy")

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length_mm,
    marker_length_mm,
    aruco_dict
)

def mm_to_px(mm):
    return int(round(mm / 25.4 * dpi))

board_width_mm = squares_x * square_length_mm    # 70 mm
board_height_mm = squares_y * square_length_mm   # 90 mm

board_width_px = mm_to_px(board_width_mm)
board_height_px = mm_to_px(board_height_mm)
margin_px = mm_to_px(margin_mm)

image_width_px = board_width_px + 2 * margin_px
image_height_px = board_height_px + 2 * margin_px

img = board.generateImage(
    (image_width_px, image_height_px),
    marginSize=margin_px,
    borderBits=1
)

output_dir.mkdir(parents=True, exist_ok=True)
output_png = output_dir / "charuco_9x7_square10_marker7_300dpi.png"
output_pdf = output_dir / "charuco_9x7_square10_marker7_300dpi.pdf"

pil_img = Image.fromarray(img)
pil_img.save(output_png, dpi=(dpi, dpi))
pil_img.save(output_pdf, "PDF", resolution=dpi)

print(f"Saved: {output_png}")
print(f"Saved: {output_pdf}")
print(f"Board size: {board_width_mm} mm x {board_height_mm} mm")
print(f"Image size with margin: {board_width_mm + 2 * margin_mm} mm x {board_height_mm + 2 * margin_mm} mm")
print(f"Pixel size: {image_width_px} px x {image_height_px} px")
