import argparse
import html
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def group_images(input_dir):
    groups = defaultdict(list)
    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.parent == Path(input_dir):
            key = path.stem.split("_out_")[0]
            key = key.rsplit("_ue2oe", 1)[0].rsplit("_occmask", 1)[0]
        else:
            key = path.parent.name
        groups[key].append(path)
    return groups


def rel(path, root):
    return path.relative_to(root).as_posix()


def write_report(input_dir, output_file, title):
    root = Path(input_dir).resolve()
    groups = group_images(root)
    rows = []
    for scene, paths in groups.items():
        cells = []
        for path in paths:
            label = html.escape(path.stem)
            src = html.escape(rel(path.resolve(), root))
            cells.append(f"<figure><img src=\"{src}\" alt=\"{label}\"><figcaption>{label}</figcaption></figure>")
        rows.append(f"<section><h2>{html.escape(scene)}</h2><div class=\"grid\">{''.join(cells)}</div></section>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 24px 32px; background: #111827; color: #fff; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ padding: 24px 32px; }}
    section {{ margin-bottom: 28px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #dde2e8; border-radius: 6px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; background: #e5e7eb; }}
    figcaption {{ padding: 8px 10px; font-size: 12px; color: #52606d; word-break: break-word; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1></header>
  <main>{''.join(rows)}</main>
</body>
</html>
"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a static HTML image report.")
    parser.add_argument("--input", required=True, help="Directory containing result or diagnosis images.")
    parser.add_argument("--output", default="reports/report.html")
    parser.add_argument("--title", default="UltraFusion-Extreme Report")
    args = parser.parse_args()
    write_report(args.input, args.output, args.title)
    print(args.output)


if __name__ == "__main__":
    main()
