# Method Notes

This fork packages a reproduction-oriented extension around the original UltraFusion codebase. The main contribution is inference-time analysis for real-shot extreme exposure cases, not retraining a new diffusion model.

## Alignment Failure Mode

UltraFusion pre-aligns the under-exposed image to the over-exposed image with RAFT and then uses forward-backward consistency to build an occlusion mask. In our real-shot tests, this can fail when:

- the over-exposed image is saturated over most of the scene
- the misaligned region is small or lacks reliable texture
- the under/over-exposed inputs describe visibly different structures
- the consistency mask is either too small to cover the artifact or too broad to preserve useful guidance

The `--strategy auto` policy is intentionally conservative. It measures the mask area and saturation ratio. If the mask looks unreliable in a saturation-heavy scene, it falls back to `noalign`; otherwise it uses the aligned guidance.

## Multi-Exposure Ordering

UltraFusion is a two-image method, but real HDR capture often provides more than two exposures. A practical way to study this is sequential pair fusion.

The reproduction report observed:

- dark-to-bright fusion can repeatedly regenerate highlight areas and blur detail
- bright-to-dark fusion often preserves more realistic high-light structure, but may produce lower overall brightness
- reducing the number of fusion steps can be better than fusing every adjacent exposure

`scripts/multi_exposure_fuse.py` therefore starts as an experiment planner. It ranks images by brightness, builds pair plans, and can materialize pair datasets that are directly consumable by `inference.py`.

## Future Work

Useful next steps:

- tune the auto-alignment thresholds against the real-shot cases
- add a soft-mask mode instead of hard `align` / `noalign`
- build an HTML report generator for visual side-by-side comparisons
- add no-reference IQA metrics and ghosting proxy metrics
- add a small gallery of failure cases and successful auto decisions

