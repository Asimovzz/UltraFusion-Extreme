from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def image_brightness(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(luminance.mean())


def saturation_ratio(path, threshold=0.98):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return float((arr >= threshold).mean())


def collect_images(input_dir):
    root = Path(input_dir)
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rank_by_brightness(paths):
    ranked = [
        {
            "path": str(path),
            "brightness": image_brightness(path),
            "saturation_ratio": saturation_ratio(path),
        }
        for path in paths
    ]
    return sorted(ranked, key=lambda item: item["brightness"])


def build_pair_plan(ranked, order):
    if order == "dark-to-bright":
        sequence = ranked
    elif order == "bright-to-dark":
        sequence = list(reversed(ranked))
    elif order == "anchor-extremes":
        if len(ranked) < 2:
            sequence = ranked
        else:
            sequence = [ranked[0], ranked[-1]]
    else:
        raise ValueError(f"Unknown order: {order}")

    pairs = []
    for index in range(len(sequence) - 1):
        first = sequence[index]
        second = sequence[index + 1]
        if first["brightness"] <= second["brightness"]:
            ue, oe = first, second
        else:
            ue, oe = second, first
        pairs.append(
            {
                "step": index + 1,
                "ue": ue["path"],
                "oe": oe["path"],
                "ue_brightness": ue["brightness"],
                "oe_brightness": oe["brightness"],
            }
        )
    return {
        "order": order,
        "sequence": sequence,
        "pairs": pairs,
    }

