# UltraFusion-Extreme Benchmark

UltraFusion-Extreme-Cases is a small real-shot stress-test set for UltraFusion inference. It is not meant to replace the official UltraFusion benchmark. Its role is to expose practical failure modes that are easy to miss on curated image pairs:

- extreme under/over exposure gaps
- saturation-heavy over-exposed inputs
- handheld motion and small misalignment
- pre-alignment mask instability
- orientation and resolution issues in casual real-world captures

## Data Layout

Each scene folder contains:

```text
0001/
  ue.png  # under-exposed input
  oe.png  # over-exposed input
```

The dataset metadata lives in `data/UltraFusion-Extreme-Cases/metadata.json`.

## Recommended Baselines

Run three policies and compare them side-by-side:

```shell
python inference.py --dataset UltraFusionExtreme --output results --tiled --tile_size 512 --tile_stride 256 --strategy noalign --save_all
python inference.py --dataset UltraFusionExtreme --output results --tiled --tile_size 512 --tile_stride 256 --strategy align --save_all
python inference.py --dataset UltraFusionExtreme --output results --tiled --tile_size 512 --tile_stride 256 --strategy auto --save_all
```

## Preview Baseline Results

For the project README, we generated a memory-friendly preview baseline on all 7 UltraFusion-Extreme cases:

```shell
conda run -n Ultrafusion python inference.py --dataset UltraFusionExtreme --output results_baseline_preview --tiled --tile_size 512 --tile_stride 256 --strategy auto --steps 10 --max_long_edge 512 --save_all
conda run -n Ultrafusion python scripts/build_report.py --input results_baseline_preview/UltraFusionExtreme --output reports/baseline_preview.html --title "UltraFusion-Extreme Preview Baseline"
conda run -n Ultrafusion python scripts/make_contact_sheet.py --input results_baseline_preview/UltraFusionExtreme --output assets/extreme_baseline/preview_contact_sheet.jpg --prefix UltraFusionExtreme --count 7
```

Settings:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM
- Diffusion steps: 10
- Long-edge resize: 512 px
- Alignment policy: `auto`
- Outputs: `results_baseline_preview/UltraFusionExtreme`
- HTML gallery: `reports/baseline_preview.html`
- README preview image: `assets/extreme_baseline/preview_contact_sheet.jpg`

Auto-alignment decisions at this preview resolution:

| Scene | Decision | Occlusion Ratio | Saturation Ratio |
| --- | --- | ---: | ---: |
| 0001 | auto_align | 0.0044 | 0.8389 |
| 0002 | auto_align | 0.1997 | 0.3908 |
| 0003 | auto_align | 0.0014 | 0.0269 |
| 0004 | auto_align | 0.0000 | 0.1515 |
| 0005 | auto_align | 0.0269 | 0.0131 |
| 0006 | auto_align | 0.0320 | 0.4627 |
| 0007 | auto_align | 0.0000 | 0.0001 |

This preview baseline is intended for quick visual inspection and README presentation. It should not be treated as a full-quality benchmark result because resizing changes RAFT mask statistics and 10 diffusion steps are much lower than the default 50-step inference.

Use `scripts/diagnose_alignment.py` before full diffusion inference when you only need to inspect the RAFT alignment behavior:

```shell
python scripts/diagnose_alignment.py --dataset UltraFusionExtreme --output reports/alignment_diagnosis
```

The diagnosis report writes per-scene inputs, IMF-adjusted under-exposed images, warped images, occlusion masks, flow magnitude visualizations, and `summary.json`.

Generate a static HTML gallery from the saved images:

```shell
python scripts/build_report.py --input reports/alignment_diagnosis --output reports/alignment_diagnosis.html
```

## What To Look For

The most useful comparisons are not only perceptual quality scores. For this dataset, inspect:

- whether the occlusion mask covers the actual moving/misaligned region
- whether the warped under-exposed image introduces incorrect structure
- whether `align` creates ghosting that `noalign` avoids
- whether `auto` makes the same decision a human would make after inspecting the mask
- whether repeated multi-exposure fusion blurs generated highlight regions
