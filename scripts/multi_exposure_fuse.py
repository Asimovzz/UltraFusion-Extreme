import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultrafusion_extreme.exposure import build_pair_plan, collect_images, rank_by_brightness


def copy_pair_dataset(pair, destination):
    scene_dir = destination / f"step_{pair['step']:02d}" / "0001"
    scene_dir.mkdir(parents=True, exist_ok=True)
    ue_src = Path(pair["ue"])
    oe_src = Path(pair["oe"])
    ue_dst = scene_dir / f"ue{ue_src.suffix.lower()}"
    oe_dst = scene_dir / f"oe{oe_src.suffix.lower()}"
    shutil.copy2(ue_src, ue_dst)
    shutil.copy2(oe_src, oe_dst)
    return scene_dir.parent


def main():
    parser = argparse.ArgumentParser(
        description="Build an exposure-order plan for sequential UltraFusion experiments."
    )
    parser.add_argument("--input_dir", default=None, help="Directory containing a multi-exposure burst.")
    parser.add_argument("--inputs", nargs="*", default=None, help="Explicit image list. Used when --input_dir is absent.")
    parser.add_argument("--order", default="bright-to-dark", choices=["bright-to-dark", "dark-to-bright", "anchor-extremes"])
    parser.add_argument("--output", default="reports/multi_exposure_plan")
    parser.add_argument("--write_pair_datasets", action="store_true")
    parser.add_argument("--inference_output", default="results_multi_exposure")
    args = parser.parse_args()

    if args.input_dir:
        paths = collect_images(args.input_dir)
    else:
        paths = [Path(path) for path in (args.inputs or [])]

    if len(paths) < 2:
        raise ValueError("At least two exposure images are required.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = rank_by_brightness(paths)
    plan = build_pair_plan(ranked, args.order)

    commands = []
    if args.write_pair_datasets:
        pair_root = output_dir / "pair_datasets"
        for pair in plan["pairs"]:
            dataset_dir = copy_pair_dataset(pair, pair_root)
            commands.append(
                [
                    "python",
                    "inference.py",
                    "--dataset",
                    dataset_dir.name,
                    "--input_dir",
                    str(dataset_dir),
                    "--output",
                    args.inference_output,
                    "--tiled",
                    "--tile_size",
                    "512",
                    "--tile_stride",
                    "256",
                    "--strategy",
                    "auto",
                    "--save_all",
                ]
            )
    plan["commands"] = commands

    with open(output_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
