"""Gaussian scoring and top-k selection."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .config import VisibleBandConfig


def compute_gaussian_contrast_proxy(
    gaussians: Any,
    projected: dict,
    tile_luminance_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use SH DC color magnitude as an inexpensive local contrast proxy."""

    features_dc = gaussians.get_features_dc
    dc = features_dc.reshape(features_dc.shape[0], -1)
    proxy = dc.abs().mean(dim=1)
    if tile_luminance_mean is not None:
        tile_id = projected["tile_id"]
        means = tile_luminance_mean.reshape(-1).to(device=proxy.device, dtype=proxy.dtype)
        valid = (tile_id >= 0) & (tile_id < means.numel())
        denom = torch.ones_like(proxy)
        denom[valid] = means[tile_id[valid]].abs().clamp_min(1.0e-4)
        proxy = proxy / denom
    return proxy


def score_gaussians(
    projected: dict,
    gaussians: Any,
    tile_visible_energy: torch.Tensor,
    config: VisibleBandConfig,
) -> torch.Tensor:
    """Score Gaussians from their tile visible energy and attributes."""

    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    tile_id = projected["tile_id"].to(device=device)
    visible = projected["visible"].to(device=device)
    tile_energy = tile_visible_energy.reshape(-1).to(device=device, dtype=gaussians.get_xyz.dtype)
    energy = torch.zeros(n, dtype=gaussians.get_xyz.dtype, device=device)
    valid_tile = (tile_id >= 0) & (tile_id < tile_energy.numel())
    energy[valid_tile] = tile_energy[tile_id[valid_tile]]
    opacity = gaussians.get_opacity.reshape(-1).to(dtype=energy.dtype)
    area = projected["projected_area"].reshape(-1).to(device=device, dtype=energy.dtype)
    if area.numel() > 0:
        area = area / area.max().clamp_min(1.0e-6)
    contrast_proxy = compute_gaussian_contrast_proxy(gaussians, projected).to(dtype=energy.dtype)
    if contrast_proxy.numel() > 0:
        contrast_proxy = contrast_proxy / contrast_proxy.max().clamp_min(1.0e-6)
    score_cfg = config.score
    score = (
        float(score_cfg.weight_visible_energy) * energy
        * (float(score_cfg.weight_opacity) * opacity)
        * (1.0 + float(score_cfg.weight_projected_area) * area)
        * (1.0 + float(score_cfg.weight_contrast_proxy) * contrast_proxy)
    )
    return torch.where(visible & valid_tile, score, torch.full_like(score, -float("inf")))


def select_gaussians_by_tile_budget(
    score: torch.Tensor,
    tile_id: torch.Tensor,
    visible: torch.Tensor,
    B_tile: torch.Tensor,
) -> torch.Tensor:
    """Select top-scoring Gaussians per tile using Python loop + torch.topk."""

    flat_score = score.reshape(-1)
    flat_tile = tile_id.reshape(-1).to(device=flat_score.device)
    flat_visible = visible.reshape(-1).to(device=flat_score.device)
    budgets = B_tile.reshape(-1).to(device=flat_score.device, dtype=torch.long)
    selected = torch.zeros_like(flat_visible, dtype=torch.bool)
    for tile in torch.unique(flat_tile[flat_tile >= 0]).tolist():
        tile = int(tile)
        if tile >= budgets.numel():
            continue
        budget = int(budgets[tile].item())
        if budget <= 0:
            continue
        in_tile = (flat_tile == tile) & flat_visible & torch.isfinite(flat_score)
        count = int(in_tile.sum().item())
        if count == 0:
            continue
        k = min(budget, count)
        indices = torch.nonzero(in_tile, as_tuple=False).flatten()
        top = torch.topk(flat_score[indices], k=k, largest=True, sorted=False).indices
        selected[indices[top]] = True
    return selected.reshape_as(visible)


def compute_gaussian_scores(*args: Any, **kwargs: Any) -> torch.Tensor:
    return score_gaussians(*args, **kwargs)


def select_gaussians(
    scores: torch.Tensor,
    tile_budget: torch.Tensor | None = None,
    xy: torch.Tensor | None = None,
    image_size: tuple[int, int] | None = None,
    tile_size: int | None = None,
    global_budget: int | None = None,
    min_score: float | None = None,
) -> torch.Tensor:
    del xy, image_size, tile_size
    flat_scores = scores.reshape(-1)
    eligible = torch.isfinite(flat_scores)
    if min_score is not None:
        eligible = eligible & (flat_scores >= float(min_score))
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    if tile_budget is None:
        if global_budget is None:
            selected[eligible] = True
        else:
            k = min(int(global_budget), int(eligible.sum().item()))
            if k > 0:
                indices = torch.nonzero(eligible, as_tuple=False).flatten()
                top = torch.topk(flat_scores[indices], k=k, largest=True, sorted=False).indices
                selected[indices[top]] = True
        return selected.reshape_as(scores)
    raise ValueError("Use select_gaussians_by_tile_budget for tile-wise selection.")


def selected_mask_summary(
    selected: torch.Tensor,
    scores: torch.Tensor | None = None,
    tile_ids: torch.Tensor | None = None,
) -> Dict[str, object]:
    mask = selected.to(torch.bool).reshape(-1)
    total = int(mask.numel())
    selected_count = int(mask.sum().item())
    summary: Dict[str, object] = {
        "total_count": total,
        "selected_count": selected_count,
        "rejected_count": total - selected_count,
        "selected_ratio": float(selected_count) / float(total) if total else 0.0,
    }
    if scores is not None and selected_count > 0:
        selected_scores = scores.reshape(-1)[mask]
        summary["score_mean"] = float(selected_scores.mean().item())
        summary["score_min"] = float(selected_scores.min().item())
        summary["score_max"] = float(selected_scores.max().item())
    if tile_ids is not None and selected_count > 0:
        valid = tile_ids.reshape(-1)[mask]
        valid = valid[valid >= 0]
        if valid.numel() > 0:
            unique, counts = torch.unique(valid, return_counts=True)
            summary["tiles_touched"] = int(unique.numel())
            summary["max_selected_per_tile"] = int(counts.max().item())
    return summary
