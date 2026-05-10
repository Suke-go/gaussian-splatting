"""Tile-wise Gaussian budget allocation."""

from __future__ import annotations

from typing import Any

import torch

from .config import VisibleBandConfig


def _match_integer_budget(
    budgets: torch.Tensor,
    target: int,
    priorities: torch.Tensor,
    max_values: torch.Tensor,
) -> torch.Tensor:
    out = budgets.to(torch.long).clone()
    target = max(0, int(target))
    current = int(out.sum().item())
    flat = out.reshape(-1)
    prio = priorities.reshape(-1).to(flat.device)
    caps = max_values.reshape(-1).to(flat.device).to(torch.long)
    if current < target:
        remaining = target - current
        order = torch.argsort(prio, descending=True).tolist()
        while remaining > 0:
            changed = False
            for idx in order:
                if remaining <= 0:
                    break
                if int(flat[idx].item()) >= int(caps[idx].item()):
                    continue
                flat[idx] += 1
                remaining -= 1
                changed = True
            if not changed:
                break
    elif current > target:
        surplus = current - target
        order = torch.argsort(prio, descending=False).tolist()
        for idx in order:
            if surplus <= 0:
                break
            remove = min(int(flat[idx].item()), surplus)
            if remove > 0:
                flat[idx] -= remove
                surplus -= remove
    return out


def base_keep_ratio(ecc_deg: torch.Tensor, config: VisibleBandConfig) -> torch.Tensor:
    cfg = config.budget
    fovea = torch.full_like(ecc_deg, float(cfg.keep_ratio_fovea))
    para = torch.full_like(ecc_deg, float(cfg.keep_ratio_parafovea))
    peri = torch.full_like(ecc_deg, float(cfg.keep_ratio_periphery))
    return torch.where(ecc_deg <= cfg.fovea_deg, fovea, torch.where(ecc_deg <= cfg.parafovea_deg, para, peri))


def allocate_tile_budget(
    tile_visible_energy: torch.Tensor,
    tile_eccentricity: torch.Tensor,
    tile_gaussian_count: torch.Tensor,
    config: VisibleBandConfig,
) -> torch.Tensor:
    """Allocate the number of Gaussians to keep per tile."""

    count = tile_gaussian_count.to(dtype=torch.float32)
    ecc = tile_eccentricity.to(device=count.device, dtype=count.dtype).reshape_as(count)
    energy = tile_visible_energy.to(device=count.device, dtype=count.dtype).reshape_as(count).clamp_min(0.0)
    cfg = config.budget
    base = base_keep_ratio(ecc, config)
    if energy.numel() > 0 and float(energy.max().item()) > 0.0:
        norm_energy = energy / energy.max().clamp_min(1.0e-6)
    else:
        norm_energy = torch.zeros_like(energy)
    ratio = base + float(cfg.lambda_visible) * norm_energy
    ratio = ratio.clamp(float(cfg.min_keep_ratio), float(cfg.max_keep_ratio))
    raw = count * ratio
    budgets = torch.ceil(raw).to(torch.long)
    budgets = torch.minimum(budgets, tile_gaussian_count.to(torch.long))
    if cfg.match_global_budget:
        if cfg.target_global_budget is not None:
            target = int(cfg.target_global_budget)
        else:
            target = int(torch.ceil(count * base.clamp(float(cfg.min_keep_ratio), float(cfg.max_keep_ratio))).sum().item())
        budgets = _match_integer_budget(
            budgets,
            target=target,
            priorities=norm_energy + base * 1.0e-3,
            max_values=tile_gaussian_count.to(torch.long),
        )
    return budgets


def match_global_budget(
    budgets: torch.Tensor,
    target: int,
    priorities: torch.Tensor | None = None,
    min_per_tile: int = 0,
    max_per_tile: int = 0,
) -> torch.Tensor:
    del min_per_tile
    if priorities is None:
        priorities = budgets.to(torch.float32)
    max_values = torch.full_like(budgets, int(max_per_tile) if max_per_tile > 0 else int(max(target, int(budgets.max().item()) if budgets.numel() else 0)))
    return _match_integer_budget(budgets, target, priorities, max_values)


def flatten_tile_budget(tile_budget: torch.Tensor) -> torch.Tensor:
    return tile_budget.to(torch.long).reshape(-1)
