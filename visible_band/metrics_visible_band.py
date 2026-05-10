"""Visible-band objective metrics."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .band_decompose import compute_band_contrast, decompose_bands, to_luminance
from .visibility import compute_eccentricity_map, eccentricity_bins, estimate_visible_band_energy


ECC_BIN_NAMES = ("0-5", "5-15", "15-30", "30+")


def _visible_repr(image: torch.Tensor, gaze_uv: tuple[float, float], config: Any) -> dict:
    lum = to_luminance(image, keepdim=True)
    bands = decompose_bands(lum, config)
    contrast = compute_band_contrast(bands["bands"], bands["local_mean"])
    ecc = compute_eccentricity_map(
        contrast.shape[-2],
        contrast.shape[-1],
        gaze_uv,
        config.gaze.fov_x_deg,
        config.gaze.fov_y_deg,
        contrast.device,
    )
    visible = estimate_visible_band_energy(contrast, ecc, config)
    return {"luminance": lum, "bands": bands, "contrast": contrast, "ecc": ecc, **visible}


def compute_visible_band_preservation(
    full_image: torch.Tensor,
    method_image: torch.Tensor,
    gaze_uv: tuple[float, float],
    config: Any,
) -> dict:
    """Ratio of full supra-threshold band energy preserved by method render."""

    full = _visible_repr(full_image, gaze_uv, config)
    method = _visible_repr(method_image, gaze_uv, config)
    mask = full["visible_mask"]
    full_energy = full["visible_energy"]
    method_energy = method["visible_energy"]
    preserved = torch.minimum(method_energy, full_energy)
    denom = full_energy[mask].sum().clamp_min(1.0e-8)
    total = (preserved[mask].sum() / denom).clamp(0.0, 1.0)
    by_band: Dict[str, float] = {}
    names = list(config.bands.names)
    for idx in range(full_energy.shape[0]):
        band_mask = mask[idx]
        band_denom = full_energy[idx][band_mask].sum().clamp_min(1.0e-8)
        key = names[idx] if idx < len(names) else f"band_{idx}"
        by_band[key] = float((preserved[idx][band_mask].sum() / band_denom).clamp(0.0, 1.0).item())
    bins = eccentricity_bins(full["ecc"])
    by_ecc: Dict[str, float] = {}
    for idx, name in enumerate(ECC_BIN_NAMES):
        pix = bins == idx
        band_pix = pix.unsqueeze(0).expand_as(mask) & mask
        if int(band_pix.sum().item()) == 0:
            by_ecc[name] = 0.0
        else:
            by_ecc[name] = float((preserved[band_pix].sum() / full_energy[band_pix].sum().clamp_min(1.0e-8)).clamp(0.0, 1.0).item())
    return {
        "preservation_by_band": by_band,
        "preservation_by_ecc_bin": by_ecc,
        "total_preservation": float(total.item()),
    }


def compute_visible_band_error(
    full_image: torch.Tensor,
    method_image: torch.Tensor,
    gaze_uv: tuple[float, float],
    config: Any,
) -> dict:
    """Compute |C_full - C_method| by band and eccentricity bin."""

    full = _visible_repr(full_image, gaze_uv, config)
    method = _visible_repr(method_image, gaze_uv, config)
    error = (full["contrast"] - method["contrast"]).abs()
    names = list(config.bands.names)
    by_band: Dict[str, float] = {}
    for idx in range(error.shape[0]):
        key = names[idx] if idx < len(names) else f"band_{idx}"
        by_band[key] = float(error[idx].mean().item())
    bins = eccentricity_bins(full["ecc"])
    by_ecc_band: Dict[str, Dict[str, float]] = {}
    for ecc_idx, ecc_name in enumerate(ECC_BIN_NAMES):
        pix = bins == ecc_idx
        by_ecc_band[ecc_name] = {}
        for band_idx in range(error.shape[0]):
            key = names[band_idx] if band_idx < len(names) else f"band_{band_idx}"
            by_ecc_band[ecc_name][key] = float(error[band_idx][pix].mean().item()) if int(pix.sum().item()) else 0.0
    return {
        "error_by_band": by_band,
        "error_by_ecc_bin_band": by_ecc_band,
        "total_error": float(error.mean().item()),
    }


def compute_subthreshold_budget_waste(
    selected_mask: torch.Tensor,
    projected: dict,
    visible_band_tiles: torch.Tensor,
    config: Any,
) -> float:
    """Fraction of selected Gaussians assigned to low visible-energy peripheral tiles."""

    del config
    selected = selected_mask.reshape(-1).to(torch.bool)
    if int(selected.sum().item()) == 0:
        return 0.0
    tile_id = projected["tile_id"].reshape(-1).to(selected.device)
    tile_energy = visible_band_tiles.reshape(-1).to(selected.device)
    valid = selected & (tile_id >= 0) & (tile_id < tile_energy.numel())
    if int(valid.sum().item()) == 0:
        return 0.0
    selected_energy = tile_energy[tile_id[valid]]
    threshold = tile_energy.mean()
    waste = selected_energy <= threshold
    return float(waste.float().mean().item())


def compute_selection_churn(selected_t: torch.Tensor, selected_prev: torch.Tensor) -> float:
    a = selected_t.reshape(-1).to(torch.bool)
    b = selected_prev.reshape(-1).to(torch.bool)
    union = (a | b).sum()
    if int(union.item()) == 0:
        return 0.0
    inter = (a & b).sum()
    return float((1.0 - inter.float() / union.float()).item())


def visible_band_metrics(pred: torch.Tensor, target: torch.Tensor, gaze_uv: tuple[float, float], config: Any) -> Dict[str, float]:
    preservation = compute_visible_band_preservation(target, pred, gaze_uv, config)
    error = compute_visible_band_error(target, pred, gaze_uv, config)
    return {
        "total_preservation": preservation["total_preservation"],
        "total_visible_band_error": error["total_error"],
    }


compute_visible_band_metrics = visible_band_metrics
compute_visible_band_mae = visible_band_metrics
compute_visible_band_rmse = visible_band_metrics
compute_visible_band_energy_error = compute_visible_band_error
compute_visible_band_contrast_error = compute_visible_band_error
compute_visible_contrast_error = compute_visible_band_error
visible_band_mae = visible_band_metrics
visible_band_rmse = visible_band_metrics
visible_band_energy_error = compute_visible_band_error
visible_band_contrast_error = compute_visible_band_error
metric_visible_band = visible_band_metrics
