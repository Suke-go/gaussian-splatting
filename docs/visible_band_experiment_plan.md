# Visible-Band Splat Budgeting Experiment Plan

This document defines the first reproducible experiment track for the
Visible-Band Splat Budgeting control layer. The goal is not to modify 3DGS
training, densification, COLMAP conversion, or CUDA rasterization internals.
The goal is to compare Gaussian selection policies on the same pretrained 3DGS
models and camera paths.

## Repository Safety

- Local branch: `visible-band`
- Working fork remote: `origin = git@github.com:Suke-go/gaussian-splatting.git`
- Official reference remote: `upstream = https://github.com/graphdeco-inria/gaussian-splatting.git`
- Push to upstream is disabled locally with `upstream DISABLED (push)`.
- Official files such as `train.py`, COLMAP conversion, densification logic,
  and `gaussian_renderer/__init__.py` are kept unchanged.

## Implementation Scope

Added modules:

- `visible_band/`: perceptual budgeting utilities
- `gaussian_renderer/render_visible_band.py`: selected-mask rasterizer wrapper
- `scripts/render_sequence_visible_band.py`: sequence rendering entry point
- `scripts/eval_visible_band.py`: full-vs-method evaluation entry point
- `configs/visible_band_*.yaml`: method, budget, gaze, and debug settings
- `paths/*.json`: placeholder camera path specifications

## Phase 1-5 Goal

The first milestone is a monocular sequence benchmark with:

- Methods: `full`, `ecc_only`, `visible_band`
- Optional baselines: `random`, `opacity_size`
- Scenes: `garden`, `room`, `bicycle` when paths are available
- Camera paths: `yaw`, `lateral`
- Fixed gaze: `gaze_uv = [0.5, 0.5]`
- Budget control: `match_global_budget: true`

Success condition:

- `visible_band` improves visible-band preservation over `ecc_only` at the
  same matched budget.
- `visible_band` reduces sub-threshold budget waste.
- Runtime overhead is recorded and explainable.

## Required Environment Gate

Before running renders:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; import simple_knn._C; print('3DGS extensions ok')"
python render.py --help
```

The current plain Python environment is only sufficient for syntax and CLI
checks. Rendering requires the official 3DGS CUDA/PyTorch environment.

## Render Commands

Set paths per scene:

```powershell
$SCENE = "data/mip360/garden"
$MODEL = "models/garden"
$OUT = "outputs/garden"
```

Full reference:

```powershell
python scripts/render_sequence_visible_band.py `
  -s $SCENE `
  -m $MODEL `
  --method full `
  --config configs/visible_band_default.yaml `
  --camera_path paths/yaw.json `
  --output "$OUT/full_yaw"
```

Eccentricity-only baseline:

```powershell
python scripts/render_sequence_visible_band.py `
  -s $SCENE `
  -m $MODEL `
  --method ecc_only `
  --config configs/visible_band_default.yaml `
  --camera_path paths/yaw.json `
  --output "$OUT/ecc_only_yaw"
```

Visible-band method:

```powershell
python scripts/render_sequence_visible_band.py `
  -s $SCENE `
  -m $MODEL `
  --method visible_band `
  --config configs/visible_band_default.yaml `
  --camera_path paths/yaw.json `
  --output "$OUT/visible_band_yaw"
```

Optional baselines:

```powershell
python scripts/render_sequence_visible_band.py -s $SCENE -m $MODEL --method random --config configs/visible_band_default.yaml --camera_path paths/yaw.json --output "$OUT/random_yaw"
python scripts/render_sequence_visible_band.py -s $SCENE -m $MODEL --method opacity_size --config configs/visible_band_default.yaml --camera_path paths/yaw.json --output "$OUT/opacity_size_yaw"
```

## Output Contract

Each run writes:

```text
outputs/<scene>/<method>_<path>/
  frames/
    00000.png
    ...
  debug/
    visible_band_maps/
    budget_maps/
    selected_masks/
    eccentricity_maps/
    band_contrast_maps/
  runtime.csv
  metadata.json
  metrics.json
```

`frames/` contains only RGB evaluation frames. Debug assets must stay under
`debug/` so they never mix with image metrics.

## Evaluation Commands

Evaluate each method against full:

```powershell
python scripts/eval_visible_band.py `
  --full "$OUT/full_yaw" `
  --method "$OUT/ecc_only_yaw" `
  --config configs/visible_band_default.yaml `
  --output "$OUT/ecc_only_yaw/metrics.json"

python scripts/eval_visible_band.py `
  --full "$OUT/full_yaw" `
  --method "$OUT/visible_band_yaw" `
  --config configs/visible_band_default.yaml `
  --output "$OUT/visible_band_yaw/metrics.json"
```

Primary metrics:

- visible-band preservation
- visible-band error
- sub-threshold budget waste
- selected Gaussian count
- frame-time breakdown

Secondary metrics:

- PSNR
- SSIM

LPIPS and FovVideoVDP are deferred until the dependency environment is stable.

## Runtime CSV Fields

The render script records per-frame rows with:

- `frame`
- `image`
- `method`
- `total_frame_time_sec`
- `prepass_time_sec`
- `band_time_sec`
- `budget_time_sec`
- `selection_time_sec`
- `rasterization_time_sec`
- `selected_count`
- `visible_selected_count`
- `total_count`
- `budget_sum`
- `selection_ratio`
- `subthreshold_budget_waste`

## Debug Figures

Representative frames are controlled by:

```yaml
debug:
  representative_frames: [0]
```

Paper figures from Phase 1-5:

- Full / Ecc-only / Visible-Band render comparison
- Low-res luminance, band contrast, visible-band map, tile budget map
- Selected Gaussian mask summary
- Visible-band preservation by eccentricity bin
- Frame-time breakdown

## Next Gates

1. Install or activate a CUDA-capable 3DGS Python environment.
2. Run `full` on one scene and confirm frame output.
3. Run `ecc_only` and `visible_band` on the same camera path.
4. Confirm `runtime.csv`, debug PNGs, selected mask `.pt` files.
5. Run `eval_visible_band.py` and compare `metrics.json`.
6. Repeat for `garden`, `room`, `bicycle` and `yaw`, `lateral`.
7. Only after Phase 1-5 is stable, add depth and temporal branch ablations.
