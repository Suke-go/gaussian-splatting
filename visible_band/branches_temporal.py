"""Temporal helpers for visible-band energy and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


@dataclass
class TemporalState:
    prev_selected_mask: torch.Tensor
    prev_visible_energy: Optional[torch.Tensor] = None
    prev_camera_pose: Optional[torch.Tensor] = None


def temporal_ema(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    momentum: float = 0.9,
) -> torch.Tensor:
    """Exponential moving average for visible-band maps."""

    if previous is None:
        return current.clone()
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    return float(momentum) * previous.to(current) + (1.0 - float(momentum)) * current


def temporal_consistency_error(
    current: torch.Tensor,
    previous: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    error = (current - previous.to(current)).abs()
    if weight is None:
        return error.mean()
    w = weight.to(dtype=error.dtype, device=error.device)
    return (error * w).sum() / w.sum().clamp_min(eps)


def aggregate_temporal_energy(
    energies: Iterable[torch.Tensor],
    decay: float = 0.8,
) -> torch.Tensor:
    """Aggregate recent energy maps with exponential recency weighting."""

    energy_list: List[torch.Tensor] = list(energies)
    if not energy_list:
        raise ValueError("energies must not be empty")
    if decay < 0.0:
        raise ValueError("decay must be non-negative")
    out = torch.zeros_like(energy_list[-1])
    norm = 0.0
    for age, energy in enumerate(reversed(energy_list)):
        weight = float(decay) ** age
        out = out + weight * energy.to(out)
        norm += weight
    return out / max(norm, 1.0e-6)


def selection_stability(
    current_mask: torch.Tensor,
    previous_mask: torch.Tensor,
) -> torch.Tensor:
    """Return fraction of unchanged selection decisions."""

    current = current_mask.to(dtype=torch.bool)
    previous = previous_mask.to(dtype=torch.bool, device=current.device)
    return (current == previous).to(dtype=torch.float32).mean()


def apply_temporal_hysteresis(
    score: torch.Tensor,
    selected_candidate: torch.Tensor,
    temporal_state: TemporalState,
    config,
) -> torch.Tensor:
    """Prefer previously selected Gaussians when scores are near threshold."""

    selected = selected_candidate.to(dtype=torch.bool).clone()
    prev = temporal_state.prev_selected_mask.to(dtype=torch.bool, device=selected.device)
    if prev.shape != selected.shape:
        return selected
    finite_scores = score[torch.isfinite(score)]
    if finite_scores.numel() == 0:
        return selected
    max_score = finite_scores.max().clamp_min(1.0e-6)
    normalized = score / max_score
    keep_threshold = float(config.temporal_branch.keep_threshold)
    enter_threshold = float(config.temporal_branch.enter_threshold)
    keep = prev & (normalized >= keep_threshold)
    enter = selected & (normalized >= enter_threshold)
    return keep | enter
