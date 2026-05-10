"""Projection helpers for Gaussian selection."""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch


def project_gaussians_to_screen(
    viewpoint_camera: Any,
    gaussians: Any,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> dict:
    """Project Gaussian centers to screen and tile IDs."""

    xyz = gaussians.get_xyz
    device = xyz.device
    dtype = xyz.dtype
    ones = torch.ones((xyz.shape[0], 1), dtype=dtype, device=device)
    points_h = torch.cat([xyz, ones], dim=1)
    clip = points_h.matmul(viewpoint_camera.full_proj_transform)
    w = clip[:, 3]
    ndc = clip[:, :3] / (w[:, None] + 1.0e-7)
    u = (ndc[:, 0] + 1.0) * 0.5 * float(image_width)
    v = (1.0 - ndc[:, 1]) * 0.5 * float(image_height)
    uv = torch.stack([u, v], dim=1)

    view = points_h.matmul(viewpoint_camera.world_view_transform)
    depth = view[:, 2]
    visible = (
        (depth > 1.0e-6)
        & (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )

    scales = gaussians.get_scaling
    max_scale = scales.abs().amax(dim=1) if scales.dim() > 1 else scales.abs()
    focal_x = float(image_width) / (2.0 * math.tan(float(viewpoint_camera.FoVx) * 0.5))
    focal_y = float(image_height) / (2.0 * math.tan(float(viewpoint_camera.FoVy) * 0.5))
    focal_px = 0.5 * (focal_x + focal_y)
    radius_px = focal_px * max_scale / depth.clamp_min(1.0e-6)
    radius_px = torch.where(visible, radius_px.clamp_min(0.0), torch.zeros_like(radius_px))
    projected_area = math.pi * radius_px * radius_px

    tile_w = (int(image_width) + int(tile_size) - 1) // int(tile_size)
    tile_h = (int(image_height) + int(tile_size) - 1) // int(tile_size)
    tx = torch.floor(u / float(tile_size)).long()
    ty = torch.floor(v / float(tile_size)).long()
    tile_id = ty * tile_w + tx
    valid_tile = visible & (tx >= 0) & (tx < tile_w) & (ty >= 0) & (ty < tile_h)
    tile_id = torch.where(valid_tile, tile_id, torch.full_like(tile_id, -1))

    return {
        "uv": uv,
        "xy": uv,
        "depth": depth,
        "visible": visible,
        "valid": visible,
        "radius_px": radius_px,
        "radius": radius_px,
        "tile_id": tile_id,
        "tile_ids": tile_id,
        "projected_area": projected_area,
        "tile_grid": (tile_h, tile_w),
    }


def tile_linear_index(
    xy: torch.Tensor,
    image_size: Tuple[int, int],
    tile_size: int,
    invalid_value: int = -1,
) -> torch.Tensor:
    height, width = image_size
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    tx = torch.floor(xy[..., 0] / float(tile_size)).long()
    ty = torch.floor(xy[..., 1] / float(tile_size)).long()
    valid = (tx >= 0) & (tx < tile_w) & (ty >= 0) & (ty < tile_h)
    idx = ty * tile_w + tx
    return torch.where(valid, idx, torch.full_like(idx, int(invalid_value)))


def sample_tile_values(
    tile_map: torch.Tensor,
    xy: torch.Tensor,
    image_size: Tuple[int, int],
    tile_size: int,
    default: float = 0.0,
) -> torch.Tensor:
    idx = tile_linear_index(xy, image_size, tile_size)
    flat = tile_map.reshape(-1)
    out = torch.full(idx.reshape(-1).shape, float(default), dtype=flat.dtype, device=flat.device)
    valid = (idx.reshape(-1) >= 0) & (idx.reshape(-1) < flat.numel())
    out[valid] = flat[idx.reshape(-1)[valid]]
    return out.reshape(idx.shape)


def gaussian_tile_bounds(
    xy: torch.Tensor,
    radius: torch.Tensor,
    image_size: Tuple[int, int],
    tile_size: int,
) -> torch.Tensor:
    height, width = image_size
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    x0 = torch.floor((xy[..., 0] - radius) / float(tile_size)).long().clamp(0, tile_w - 1)
    y0 = torch.floor((xy[..., 1] - radius) / float(tile_size)).long().clamp(0, tile_h - 1)
    x1 = torch.floor((xy[..., 0] + radius) / float(tile_size)).long().clamp(0, tile_w - 1)
    y1 = torch.floor((xy[..., 1] + radius) / float(tile_size)).long().clamp(0, tile_h - 1)
    return torch.stack([x0, y0, x1, y1], dim=-1)


def points_in_view(xy: torch.Tensor, depth: torch.Tensor, image_size: Tuple[int, int], near: float = 1.0e-6, far: float = 1.0e12) -> torch.Tensor:
    height, width = image_size
    return (depth > near) & (depth < far) & (xy[..., 0] >= 0.0) & (xy[..., 0] < width) & (xy[..., 1] >= 0.0) & (xy[..., 1] < height)


def project_points(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Use project_gaussians_to_screen for the visible-band control layer.")
