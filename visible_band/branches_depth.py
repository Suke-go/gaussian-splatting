"""Depth-aware helpers for visible-band experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_depth(
    depth: torch.Tensor,
    valid: torch.Tensor = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    d = depth.float()
    if valid is None:
        valid = torch.isfinite(d)
    values = d[valid]
    if values.numel() == 0:
        return torch.zeros_like(d)
    lo = values.min()
    hi = values.max()
    return ((d - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)


def depth_discontinuity(
    depth: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Sobel magnitude of normalized depth."""

    d = normalize_depth(depth, eps=eps)
    if d.dim() == 2:
        d = d.unsqueeze(0).unsqueeze(0)
    elif d.dim() == 3:
        d = d.unsqueeze(0)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=d.dtype,
        device=d.device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(F.pad(d[:, :1], (1, 1, 1, 1), mode="replicate"), kernel_x)
    gy = F.conv2d(F.pad(d[:, :1], (1, 1, 1, 1), mode="replicate"), kernel_y)
    return torch.sqrt(gx * gx + gy * gy + eps)


def depth_weight_map(
    depth: torch.Tensor,
    near_weight: float = 1.0,
    far_weight: float = 0.25,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Map normalized depth to near/far weights."""

    d = normalize_depth(depth, eps=eps)
    return float(near_weight) * (1.0 - d) + float(far_weight) * d


def apply_depth_visibility(
    visible_energy: torch.Tensor,
    depth: torch.Tensor,
    edge_boost: float = 0.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    weight = depth_weight_map(depth, eps=eps).to(
        dtype=visible_energy.dtype,
        device=visible_energy.device,
    )
    if weight.dim() == 2:
        weight = weight.unsqueeze(0).unsqueeze(0)
    elif weight.dim() == 3:
        weight = weight.unsqueeze(0)
    out = visible_energy * weight
    if edge_boost > 0.0:
        edge = depth_discontinuity(depth, eps=eps).to(dtype=out.dtype, device=out.device)
        out = out * (1.0 + float(edge_boost) * edge)
    return out


def compute_depth_discontinuity_bonus(
    lowres_depth: torch.Tensor,
    tile_size: int,
    config,
) -> torch.Tensor:
    """Return normalized tile depth-discontinuity bonus Tensor[T]."""

    edge = depth_discontinuity(lowres_depth)
    if edge.dim() == 4:
        edge = edge[0]
    if edge.dim() == 2:
        edge = edge.unsqueeze(0)
    height, width = edge.shape[-2:]
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    pooled = F.avg_pool2d(
        F.pad(edge.unsqueeze(0), (0, pad_w, 0, pad_h)),
        kernel_size=tile_size,
        stride=tile_size,
    )[0, 0]
    bonus = pooled.reshape(-1)
    if bonus.numel() and float(bonus.max().item()) > 0.0:
        bonus = bonus / bonus.max().clamp_min(1.0e-6)
    threshold = float(getattr(config.depth_branch, "depth_gradient_threshold", 0.0))
    return torch.where(bonus >= threshold, bonus, torch.zeros_like(bonus))
