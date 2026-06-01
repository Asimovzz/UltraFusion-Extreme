import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def fit_thumbnail(path, size):
    if path is None or not path.exists():
        return Image.new("RGB", size, (225, 225, 225))
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def make_contact_sheet(input_dir, output, prefix, count, thumb_width, thumb_height):
    root = Path(input_dir)
    thumb_size = (thumb_width, thumb_height)
    rows = []
    for index in range(1, count + 1):
        scene = f"{prefix}_{index:04d}"
        output_paths = sorted(root.glob(f"{scene}_out_auto_*.png"))
        rows.append(
            [
                fit_thumbnail(root / f"{scene}_ue.png", thumb_size),
                fit_thumbnail(root / f"{scene}_oe.png", thumb_size),
                fit_thumbnail(output_paths[0] if output_paths else None, thumb_size),
            ]
        )

    gutter = 28
    width = thumb_width * 3
    height = gutter + (thumb_height + gutter) * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col, label in enumerate(["Under-exposed", "Over-exposed", "Auto output"]):
        draw.text((col * thumb_width + 8, 8), label, fill=(20, 20, 20))

    for row_index, row in enumerate(rows):
        y = gutter + row_index * (thumb_height + gutter)
        draw.text((8, y + thumb_height + 6), f"{row_index + 1:04d}", fill=(80, 80, 80))
        for col, img in enumerate(row):
            sheet.paste(img, (col * thumb_width, y))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create a compact preview contact sheet.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default="UltraFusionExtreme")
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--thumb_width", type=int, default=220)
    parser.add_argument("--thumb_height", type=int, default=170)
    args = parser.parse_args()
    make_contact_sheet(args.input, args.output, args.prefix, args.count, args.thumb_width, args.thumb_height)


if __name__ == "__main__":
    main()

