#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from charuco_ros2.charuco_board_utils import (
    CUBE_CHARUCO_DEFAULTS,
    board_config_from_cli,
    create_charuco_board,
    m_to_mm,
    save_board_config,
    validate_board_config,
)


VALIDATION_SQUARE_LENGTH_MM = 20.0
VALIDATION_MARKER_LENGTH_MM = 14.0

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "boards"
    / "generated"
    / "validation_charuco"
)


def mm_to_px(mm: float, dpi: int, minimum: int = 0) -> int:
    px = int(round(float(mm) / 25.4 * int(dpi)))
    return max(minimum, px)


def create_board_image(
    config: dict,
    dpi: int,
    margin_mm: float,
    draw_label: bool,
):
    charuco_board = create_charuco_board(config)

    board_width_mm = m_to_mm(config["squares_x"] * config["square_length_m"])
    board_height_mm = m_to_mm(config["squares_y"] * config["square_length_m"])
    board_width_px = mm_to_px(board_width_mm, dpi, minimum=1)
    board_height_px = mm_to_px(board_height_mm, dpi, minimum=1)
    margin_px = mm_to_px(margin_mm, dpi, minimum=0)

    image = charuco_board.generateImage(
        (board_width_px + 2 * margin_px, board_height_px + 2 * margin_px),
        marginSize=margin_px,
        borderBits=1,
    )
    pil_image = Image.fromarray(image).convert("RGB")

    if draw_label:
        draw = ImageDraw.Draw(pil_image)
        ids = config["marker_ids"]
        lines = [
            str(config.get("name", "validation_charuco")),
            (
                f"{config['squares_x']}x{config['squares_y']} "
                f"sq={m_to_mm(config['square_length_m']):g}mm "
                f"mk={m_to_mm(config['marker_length_m']):g}mm"
            ),
            f"{config['dictionary']} ID:{min(ids)}-{max(ids)}",
        ]
        x = max(4, margin_px + 4)
        y = max(4, margin_px + 4)
        text = "\n".join(lines)
        bbox = draw.multiline_textbbox((x, y), text, spacing=2)
        draw.rectangle(
            [bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3],
            fill=(255, 255, 255),
        )
        draw.multiline_text((x, y), text, fill=(0, 0, 0), spacing=2)

    return pil_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a printable ChArUco board and detector config."
    )
    parser.add_argument("--name", default="validation_charuco")
    parser.add_argument(
        "--squares-x",
        type=int,
        default=CUBE_CHARUCO_DEFAULTS["squares_x"],
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=CUBE_CHARUCO_DEFAULTS["squares_y"],
    )
    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=VALIDATION_SQUARE_LENGTH_MM,
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=VALIDATION_MARKER_LENGTH_MM,
    )
    parser.add_argument(
        "--dictionary",
        default=CUBE_CHARUCO_DEFAULTS["dictionary"],
    )
    parser.add_argument(
        "--marker-id-start",
        type=int,
        default=80,
        help="Default 80 avoids the existing 0-79 cube ChArUco IDs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            f"{DEFAULT_OUTPUT_DIR} instead of the current working directory."
        ),
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-label", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = board_config_from_cli(
        name=args.name,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        dictionary=args.dictionary,
        marker_id_start=args.marker_id_start,
    )
    config = validate_board_config(config)

    output_dir = (
        DEFAULT_OUTPUT_DIR
        if args.output_dir is None
        else Path(args.output_dir).expanduser()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = config["marker_ids"]
    prefix = args.output_prefix or (
        f"{config['name']}_{config['squares_x']}x{config['squares_y']}_"
        f"square{args.square_length_mm:g}mm_marker{args.marker_length_mm:g}mm_"
        f"id{min(ids)}-{max(ids)}_{args.dpi}dpi"
    )

    image = create_board_image(
        config,
        dpi=args.dpi,
        margin_mm=args.margin_mm,
        draw_label=not args.no_label,
    )

    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"
    yaml_path = output_dir / f"{prefix}.yaml"
    json_path = output_dir / f"{prefix}.json"

    image.save(png_path, dpi=(args.dpi, args.dpi))
    image.save(pdf_path, "PDF", resolution=args.dpi)
    save_board_config(config, yaml_path)
    save_board_config(config, json_path)

    print(f"Saved PNG:  {png_path}")
    print(f"Saved PDF:  {pdf_path}")
    print(f"Saved YAML: {yaml_path}")
    print(f"Saved JSON: {json_path}")
    print(
        "Board: "
        f"{config['squares_x']}x{config['squares_y']}, "
        f"square={args.square_length_mm:g} mm, "
        f"marker={args.marker_length_mm:g} mm, "
        f"{config['dictionary']}, IDs {min(ids)}-{max(ids)}"
    )


if __name__ == "__main__":
    main()
