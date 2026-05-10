"""Eccentricity maps, visibility thresholds, and tile aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .config import VisibleBandConfig


def compute_eccentricity_map(
    height: int,
    width: int,
    gaze_uv: tuple[float, float] = (0.5, 0.5),
    fov_x_deg: float = 100.0,
    fov_y_deg: float = 100.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return per-pixel eccentricity in visual degrees."""

    dev = device if device is not None else torch.device("cpu")
    ys = (torch.arange(height, dtype=torch.float32, device=dev) + 0.5) / float(height)
    xs = (torch.arange(width, dtype=torch.float32, device=dev) + 0.5) / float(width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    angle_x = (xx - float(gaze_uv[0])) * float(fov_x_deg)
    angle_y = (yy - float(gaze_uv[1])) * float(fov_y_deg)
    return torch.sqrt(angle_x * angle_x + angle_y * angle_y)


def _threshold_params(config: Any, band_idx: int) -> tuple[float, float]:
    if isinstance(config, VisibleBandConfig):
        names = list(config.bands.names)
        name = names[band_idx] if band_idx < len(names) else names[-1]
        params = getattr(config.visibility_threshold, name, None)
        if params is None:
            params = config.visibility_threshold.high
        return float(params.base), float(params.ecc_slope)
    if isinstance(config, Mapping):
        names = config.get("bands", {}).get("names", ("very_low", "low", "mid", "high"))
        name = names[band_idx] if band_idx < len(names) else names[-1]
        params = config.get("visibility_threshold", {}).get(name, {})
        return float(params.get("base", 0.02)), float(params.get("ecc_slope", 0.01))
    defaults = [(0.02, 0.005), (0.03, 0.010), (0.05, 0.030), (0.08, 0.070)]
    return defaults[min(band_idx, len(defaults) - 1)]


def estimate_visible_band_energy(
    band_contrast: torch.Tensor,
    ecc_deg: torch.Tensor,
    config: Any,
) -> dict:
    """Estimate supra-threshold visible band energy.

    V_b(x,y) = max(C_b(x,y) - T_b(ecc), 0)
    """

    contrast = band_contrast
    if isinstance(contrast, Mapping):
        contrast = contrast["contrast"]
    if contrast.dim() == 4:
        contrast = contrast[0]
    if ecc_deg.dim() == 3:
        ecc_deg = ecc_deg[0]
    if ecc_deg.shape[-2:] != contrast.shape[-2:]:
        ecc = F.interpolate(
            ecc_deg.reshape(1, 1, *ecc_deg.shape[-2:]),
            size=contrast.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    else:
        ecc = ecc_deg
    thresholds = []
    for band_idx in range(contrast.shape[0]):
        base, slope = _threshold_params(config, band_idx)
        thresholds.append(base + slope * ecc)
    threshold = torch.stack(thresholds, dim=0).to(dtype=contrast.dtype, device=contrast.device)
    visible_energy = torch.clamp(contrast - threshold, min=0.0)
    visible_mask = visible_energy > 0
    return {
        "visible_energy": visible_energy,
        "visible_mask": visible_mask,
        "threshold": threshold,
    }


def aggregate_visible_energy_to_tiles(visible_energy: torch.Tensor, tile_size: int) -> dict:
    """Aggregate visible energy to T tiles.

    Return:
      {"tile_visible_energy": Tensor[T], "tile_band_energy": Tensor[T,B]}
    """

    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    energy = visible_energy
    if isinstance(energy, Mapping):
        energy = energy["visible_energy"]
    if energy.dim() == 4:
        energy = energy[0]
    if energy.dim() != 3:
        raise ValueError("visible_energy must have shape B,H,W")
    bands, height, width = energy.shape
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    padded = F.pad(energy.unsqueeze(0), (0, pad_w, 0, pad_h))
    pooled = F.avg_pool2d(padded, kernel_size=tile_size, stride=tile_size)[0]
    tile_band_energy = pooled.reshape(bands, -1).transpose(0, 1).contiguous()
    tile_visible_energy = tile_band_energy.sum(dim=1)
    return {
        "tile_visible_energy": tile_visible_energy,
        "tile_band_energy": tile_band_energy,
    }


def aggregate_map_to_tiles(map_tensor: torch.Tensor, tile_size: int, mode: str = "mean") -> torch.Tensor:
    x = map_tensor
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    height, width = x.shape[-2:]
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    if mode == "max":
        pooled = F.max_pool2d(x, kernel_size=tile_size, stride=tile_size)
    elif mode == "sum":
        pooled = F.avg_pool2d(x, kernel_size=tile_size, stride=tile_size) * float(tile_size * tile_size)
    else:
        pooled = F.avg_pool2d(x, kernel_size=tile_size, stride=tile_size)
    return pooled.reshape(-1)


def eccentricity_bins(ecc_deg: torch.Tensor) -> torch.Tensor:
    bins = torch.zeros_like(ecc_deg, dtype=torch.long)
    bins = torch.where(ecc_deg >= 5.0, torch.ones_like(bins), bins)
    bins = torch.where(ecc_deg >= 15.0, torch.full_like(bins, 2), bins)
    bins = torch.where(ecc_deg >= 30.0, torch.full_like(bins, 3), bins)
    return bins


def visible_energy(
    bands_or_contrast: torch.Tensor,
    eccentricity: torch.Tensor | None = None,
    band_weights: list[float] | tuple[float, ...] | None = None,
    normalize: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    energy = bands_or_contrast.abs()
    if energy.dim() == 4:
        energy = energy[0]
    if band_weights is not None:
        weights = torch.tensor(band_weights, dtype=energy.dtype, device=energy.device)
        if weights.numel() < energy.shape[0]:
            weights = torch.cat([weights, weights[-1:].repeat(energy.shape[0] - weights.numel())])
        energy = energy * weights[: energy.shape[0]].view(-1, 1, 1)
    out = energy.sum(dim=0, keepdim=True)
    if eccentricity is not None:
        ecc = eccentricity
        if ecc.dim() == 2:
            ecc = ecc.unsqueeze(0)
        out = out / (1.0 + ecc.to(out.device, out.dtype) / 30.0)
    if normalize:
        out = out / out.amax().clamp_min(eps)
    return out


def tile_aggregate(energy: torch.Tensor, tile_size: int = 16, mode: str = "mean", pad_value: float = 0.0) -> torch.Tensor:
    del pad_value
    return aggregate_map_to_tiles(energy, tile_size=tile_size, mode=mode)


def tile_coordinates(height: int, width: int, tile_size: int, device: torch.device | None = None) -> torch.Tensor:
    tile_h = math.ceil(height / tile_size)
    tile_w = math.ceil(width / tile_size)
    yy, xx = torch.meshgrid(torch.arange(tile_h, device=device), torch.arange(tile_w, device=device), indexing="ij")
    return torch.stack([xx, yy], dim=-1)
