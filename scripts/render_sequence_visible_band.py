#!/usr/bin/env python3
"""Render full/eccentricity/visible-band 3DGS sequences.

Heavy 3DGS imports are delayed so `--help` works before CUDA dependencies are
installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence


METHODS = ("full", "ecc_only", "visible_band", "random", "opacity_size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Visible-Band Splat Budgeting sequences.")
    parser.add_argument("-s", "--source_path", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--depths", default="", type=str)
    parser.add_argument("-r", "--resolution", default=-1, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--train_test_exp", action="store_true")
    parser.add_argument("--data_device", default="cuda", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--method", default="visible_band", choices=METHODS)
    parser.add_argument("--config", default="configs/visible_band_default.yaml", type=str)
    parser.add_argument("--camera_path", default="", type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _runtime_imports() -> SimpleNamespace:
    try:
        import numpy as np
        import torch
        from gaussian_renderer import GaussianModel, render as render_full
        from gaussian_renderer.render_visible_band import rasterize_selected_gaussians
        from scene import Scene
        from utils.graphics_utils import getProjectionMatrix, getWorld2View2
        from visible_band.band_decompose import compute_band_contrast, decompose_bands
        from visible_band.budget import allocate_tile_budget
        from visible_band.config import load_config, to_dict
        from visible_band.debug_vis import save_tensor_png
        from visible_band.metrics_visible_band import compute_subthreshold_budget_waste
        from visible_band.prepass import render_lowres_prepass
        from visible_band.project import project_gaussians_to_screen
        from visible_band.selector import score_gaussians, select_gaussians_by_tile_budget
        from visible_band.visibility import (
            aggregate_map_to_tiles,
            aggregate_visible_energy_to_tiles,
            compute_eccentricity_map,
            estimate_visible_band_energy,
        )
    except Exception as exc:
        raise RuntimeError(
            "Rendering requires the official 3DGS Python/CUDA dependencies. "
            "Install torch, CUDA extensions, and the scene dependencies before running."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Rendering requires CUDA-capable PyTorch.")
    return SimpleNamespace(**locals())


def _make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=args.sh_degree,
        source_path=os.path.abspath(args.source_path),
        model_path=args.model_path,
        images=args.images,
        depths=args.depths,
        resolution=args.resolution,
        white_background=args.white_background,
        train_test_exp=args.train_test_exp,
        data_device=args.data_device,
        eval=args.eval,
    )


def _make_pipe_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
        antialiasing=args.antialiasing,
    )


def _slice(items: Sequence[Any], start: int, stride: int, max_frames: Optional[int]) -> List[Any]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    out = list(items)[max(0, start) :: stride]
    return out[:max_frames] if max_frames is not None else out


def _camera_from_json(entry: Mapping[str, Any], idx: int, cfg: Any, rt: SimpleNamespace) -> Any:
    width = int(entry.get("width", cfg.render.target_width))
    height = int(entry.get("height", cfg.render.target_height))
    fovx = float(entry.get("fovx", entry.get("fov_x", 0.0)))
    fovy = float(entry.get("fovy", entry.get("fov_y", 0.0)))
    if "fovx_deg" in entry:
        fovx = float(entry["fovx_deg"]) * 3.141592653589793 / 180.0
    if "fovy_deg" in entry:
        fovy = float(entry["fovy_deg"]) * 3.141592653589793 / 180.0
    if fovx == 0.0:
        fovx = cfg.gaze.fov_x_deg * 3.141592653589793 / 180.0
    if fovy == 0.0:
        fovy = cfg.gaze.fov_y_deg * 3.141592653589793 / 180.0

    if "w2c" in entry:
        w2c = rt.np.asarray(entry["w2c"], dtype=rt.np.float32)
    elif "world_to_camera" in entry:
        w2c = rt.np.asarray(entry["world_to_camera"], dtype=rt.np.float32)
    elif "c2w" in entry:
        w2c = rt.np.linalg.inv(rt.np.asarray(entry["c2w"], dtype=rt.np.float32)).astype(rt.np.float32)
    elif "transform_matrix" in entry:
        w2c = rt.np.linalg.inv(rt.np.asarray(entry["transform_matrix"], dtype=rt.np.float32)).astype(rt.np.float32)
    elif "R" in entry and "T" in entry:
        w2c = rt.getWorld2View2(rt.np.asarray(entry["R"], dtype=rt.np.float32), rt.np.asarray(entry["T"], dtype=rt.np.float32))
    else:
        raise ValueError("camera entry needs w2c/c2w/transform_matrix or R/T")
    world_view_transform = rt.torch.tensor(w2c, dtype=rt.torch.float32, device="cuda").transpose(0, 1)
    projection_matrix = rt.getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    return SimpleNamespace(
        uid=idx,
        image_name=str(entry.get("image_name", entry.get("img_name", f"path_{idx:05d}"))),
        image_width=width,
        image_height=height,
        FoVx=fovx,
        FoVy=fovy,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=world_view_transform.inverse()[3, :3],
        znear=0.01,
        zfar=100.0,
    )


def _load_cameras(scene: Any, args: argparse.Namespace, cfg: Any, rt: SimpleNamespace) -> List[Any]:
    if not args.camera_path:
        cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        return _slice(cameras, args.start, args.stride, args.max_frames)
    spec = json.loads(Path(args.camera_path).read_text(encoding="utf-8"))
    if isinstance(spec, Mapping) and spec.get("source") in ("train", "test"):
        cameras = scene.getTrainCameras() if spec["source"] == "train" else scene.getTestCameras()
        start = int(spec.get("start", args.start))
        stride = int(spec.get("stride", args.stride))
        max_frames = spec.get("max_frames", args.max_frames)
        max_frames = None if max_frames is None else int(max_frames)
        return _slice(cameras, start, stride, max_frames)
    entries = spec.get("cameras", spec.get("frames")) if isinstance(spec, Mapping) else spec
    if not isinstance(entries, list):
        raise ValueError("camera_path must be a list or contain cameras/frames")
    return _slice([_camera_from_json(entry, idx, cfg, rt) for idx, entry in enumerate(entries)], args.start, args.stride, args.max_frames)


def _save_rgb(tensor: Any, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to save rendered PNGs") from exc
    image = tensor.detach().clamp(0.0, 1.0)
    array = (image * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _tile_counts(tile_id: Any, visible: Any, num_tiles: int, torch: Any) -> Any:
    valid = visible & (tile_id >= 0) & (tile_id < num_tiles)
    if int(valid.sum().item()) == 0:
        return torch.zeros(num_tiles, dtype=torch.long, device=tile_id.device)
    return torch.bincount(tile_id[valid], minlength=num_tiles).to(torch.long)


def _pipeline_selection(camera: Any, gaussians: Any, pipe: Any, bg: Any, cfg: Any, method: str, frame_idx: int, rt: SimpleNamespace, debug_dirs: Dict[str, Path]) -> tuple[Any, Dict[str, Any]]:
    torch = rt.torch
    tile_size = int(cfg.render.tile_size)
    width = int(camera.image_width)
    height = int(camera.image_height)
    projected = rt.project_gaussians_to_screen(camera, gaussians, width, height, tile_size)
    tile_h, tile_w = projected["tile_grid"]
    num_tiles = tile_h * tile_w
    tile_count = _tile_counts(projected["tile_id"], projected["visible"], num_tiles, torch)
    ecc = rt.compute_eccentricity_map(height, width, tuple(cfg.gaze.gaze_uv), cfg.gaze.fov_x_deg, cfg.gaze.fov_y_deg, gaussians.get_xyz.device)
    tile_ecc = rt.aggregate_map_to_tiles(ecc, tile_size=tile_size, mode="mean")

    timings: Dict[str, float] = {"prepass_time_sec": 0.0, "band_time_sec": 0.0, "budget_time_sec": 0.0, "selection_time_sec": 0.0}
    debug_maps: Dict[str, Any] = {"eccentricity": ecc}

    if method == "visible_band":
        start = time.perf_counter()
        prepass = rt.render_lowres_prepass(camera, gaussians, pipe, bg, (cfg.render.lowres_width, cfg.render.lowres_height))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["prepass_time_sec"] = time.perf_counter() - start

        start = time.perf_counter()
        bands = rt.decompose_bands(prepass["luminance"], cfg)
        contrast = rt.compute_band_contrast(bands["bands"], bands["local_mean"])
        low_ecc = rt.compute_eccentricity_map(contrast.shape[-2], contrast.shape[-1], tuple(cfg.gaze.gaze_uv), cfg.gaze.fov_x_deg, cfg.gaze.fov_y_deg, contrast.device)
        visible = rt.estimate_visible_band_energy(contrast, low_ecc, cfg)
        visible_full = torch.nn.functional.interpolate(
            visible["visible_energy"].unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        tile_energy = rt.aggregate_visible_energy_to_tiles(visible_full, tile_size)["tile_visible_energy"]
        debug_maps.update(
            {
                "luminance": prepass["luminance"],
                "visible_band": visible_full.sum(dim=0, keepdim=True),
                "band_contrast": contrast.sum(dim=0, keepdim=True),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["band_time_sec"] = time.perf_counter() - start
    else:
        tile_energy = torch.zeros(num_tiles, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)

    start = time.perf_counter()
    budget_cfg = cfg
    if method == "ecc_only":
        budget_cfg = rt.load_config(rt.to_dict(cfg), overrides={"budget": {"lambda_visible": 0.0}})
    B_tile = rt.allocate_tile_budget(tile_energy, tile_ecc, tile_count, budget_cfg)
    if method in ("random", "opacity_size"):
        ecc_budget_cfg = rt.load_config(rt.to_dict(cfg), overrides={"budget": {"lambda_visible": 0.0}})
        B_tile = rt.allocate_tile_budget(torch.zeros_like(tile_energy), tile_ecc, tile_count, ecc_budget_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["budget_time_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    if method == "random":
        selected = torch.zeros_like(projected["visible"], dtype=torch.bool)
        generator = torch.Generator(device=gaussians.get_xyz.device)
        generator.manual_seed(13 + int(frame_idx))
        for tile in torch.unique(projected["tile_id"][projected["tile_id"] >= 0]).tolist():
            tile = int(tile)
            k = int(B_tile[tile].item()) if tile < B_tile.numel() else 0
            pool = torch.nonzero((projected["tile_id"] == tile) & projected["visible"], as_tuple=False).flatten()
            if k > 0 and int(pool.numel()) > 0:
                perm = torch.randperm(int(pool.numel()), generator=generator, device=pool.device)
                selected[pool[perm[: min(k, int(pool.numel()))]]] = True
    else:
        if method == "opacity_size" or method == "ecc_only":
            score = gaussians.get_opacity.reshape(-1) * projected["projected_area"].reshape(-1).clamp_min(0.0)
            score = torch.where(projected["visible"], score, torch.full_like(score, -float("inf")))
        else:
            score = rt.score_gaussians(projected, gaussians, tile_energy, cfg)
        selected = rt.select_gaussians_by_tile_budget(score, projected["tile_id"], projected["visible"], B_tile)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["selection_time_sec"] = time.perf_counter() - start

    selected_count = int(selected.sum().item())
    selected_visible = int((selected & projected["visible"]).sum().item())
    debug_maps["budget_map"] = B_tile.reshape(tile_h, tile_w).unsqueeze(0).float()
    waste = rt.compute_subthreshold_budget_waste(selected, projected, tile_energy, cfg)
    metadata = {
        "selected_count": selected_count,
        "visible_selected_count": selected_visible,
        "total_count": int(gaussians.get_xyz.shape[0]),
        "budget_sum": int(B_tile.sum().item()),
        "selection_ratio": float(selected_count) / float(gaussians.get_xyz.shape[0]),
        "subthreshold_budget_waste": waste,
        **timings,
    }

    rep = set(int(v) for v in cfg.debug.representative_frames)
    if frame_idx in rep:
        if cfg.debug.save_eccentricity_maps:
            rt.save_tensor_png(debug_maps["eccentricity"], str(debug_dirs["eccentricity"] / f"{frame_idx:05d}.png"))
        if cfg.debug.save_budget_maps:
            rt.save_tensor_png(debug_maps["budget_map"], str(debug_dirs["budget"] / f"{frame_idx:05d}.png"))
        if method == "visible_band":
            if cfg.debug.save_visible_band_maps:
                rt.save_tensor_png(debug_maps["visible_band"], str(debug_dirs["visible"] / f"{frame_idx:05d}.png"))
            if cfg.debug.save_band_contrast_maps:
                rt.save_tensor_png(debug_maps["band_contrast"], str(debug_dirs["contrast"] / f"{frame_idx:05d}.png"))
    return selected, metadata


def render_sequence(args: argparse.Namespace) -> None:
    rt = _runtime_imports()
    cfg = rt.load_config(args.config)
    random.seed(13)
    rt.torch.manual_seed(13)
    model_args = _make_model_args(args)
    pipe = _make_pipe_args(args)
    output = Path(args.output)
    frames_dir = output / "frames"
    selected_dir = output / "debug" / "selected_masks"
    debug_dirs = {
        "visible": output / "debug" / "visible_band_maps",
        "budget": output / "debug" / "budget_maps",
        "contrast": output / "debug" / "band_contrast_maps",
        "eccentricity": output / "debug" / "eccentricity_maps",
    }
    frames_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    with rt.torch.no_grad():
        gaussians = rt.GaussianModel(model_args.sh_degree)
        scene = rt.Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)
        cameras = _load_cameras(scene, args, cfg, rt)
        bg_color = [1.0, 1.0, 1.0] if model_args.white_background else [0.0, 0.0, 0.0]
        if cfg.render.background == "white":
            bg_color = [1.0, 1.0, 1.0]
        elif cfg.render.background == "black":
            bg_color = [0.0, 0.0, 0.0]
        bg = rt.torch.tensor(bg_color, dtype=rt.torch.float32, device="cuda")
        rows = []
        frame_meta = []
        for idx, camera in enumerate(cameras):
            frame_start = time.perf_counter()
            full = rt.render_full(camera, gaussians, pipe, bg, scaling_modifier=cfg.render.scaling_modifier)
            if args.method == "full":
                render_out = full
                meta = {
                    "selected_count": int(gaussians.get_xyz.shape[0]),
                    "visible_selected_count": int((full["radii"] > 0).sum().item()),
                    "total_count": int(gaussians.get_xyz.shape[0]),
                    "budget_sum": int(gaussians.get_xyz.shape[0]),
                    "selection_ratio": 1.0,
                    "subthreshold_budget_waste": 0.0,
                    "prepass_time_sec": 0.0,
                    "band_time_sec": 0.0,
                    "budget_time_sec": 0.0,
                    "selection_time_sec": 0.0,
                }
            else:
                selected, meta = _pipeline_selection(camera, gaussians, pipe, bg, cfg, args.method, idx, rt, debug_dirs)
                rt.torch.save(selected.detach().cpu(), selected_dir / f"{idx:05d}.pt")
                raster_start = time.perf_counter()
                render_out = rt.rasterize_selected_gaussians(camera, gaussians, pipe, bg, selected, scaling_modifier=cfg.render.scaling_modifier)
                if rt.torch.cuda.is_available():
                    rt.torch.cuda.synchronize()
                meta["rasterization_time_sec"] = time.perf_counter() - raster_start
            if rt.torch.cuda.is_available():
                rt.torch.cuda.synchronize()
            total_time = time.perf_counter() - frame_start
            path = frames_dir / f"{idx:05d}.png"
            _save_rgb(render_out["render"], path)
            meta.setdefault("rasterization_time_sec", total_time)
            row = {
                "frame": idx,
                "image": path.name,
                "method": args.method,
                "total_frame_time_sec": total_time,
                **meta,
            }
            rows.append(row)
            frame_meta.append(row)
            if not args.quiet:
                print(f"{idx:05d} {args.method} selected={row['selected_count']} total={row['total_count']} time={total_time:.4f}s")
    _write_runtime(output / "runtime.csv", rows)
    metadata = {
        "method": args.method,
        "source_path": args.source_path,
        "model_path": args.model_path,
        "config_path": args.config,
        "camera_path": args.camera_path,
        "frames": frame_meta,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"status": "rendered", "frame_count": len(frame_meta)}, indent=2), encoding="utf-8")


def _write_runtime(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    render_sequence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
