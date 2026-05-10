"""Orientation-aware visible-band helpers."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if image.dim() == 2:
        x = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        x = image.unsqueeze(0)
    elif image.dim() == 4:
        x = image
    else:
        raise ValueError("image must have shape HW, CHW, or NCHW")
    x = x.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


def orientation_map(image: torch.Tensor) -> torch.Tensor:
    gx, gy = image_gradients(image)
    return torch.atan2(gy, gx)


def orientation_energy(image: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    gx, gy = image_gradients(image)
    return torch.sqrt(gx * gx + gy * gy + eps)


def orientation_alignment(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Cosine alignment of image-gradient orientation."""

    pgx, pgy = image_gradients(pred)
    tgx, tgy = image_gradients(target)
    dot = pgx * tgx + pgy * tgy
    pn = torch.sqrt(pgx * pgx + pgy * pgy + eps)
    tn = torch.sqrt(tgx * tgx + tgy * tgy + eps)
    alignment = dot / (pn * tn).clamp_min(eps)
    if weight is None:
        return alignment.mean()
    w = weight.to(dtype=alignment.dtype, device=alignment.device)
    return (alignment * w).sum() / w.sum().clamp_min(eps)


def orientation_weighted_energy(
    visible_energy: torch.Tensor,
    image: torch.Tensor,
    orientation_boost: float = 0.25,
) -> torch.Tensor:
    orient = orientation_energy(image).to(
        dtype=visible_energy.dtype,
        device=visible_energy.device,
    )
    orient = orient / orient.detach().amax().clamp_min(1.0e-6)
    return visible_energy * (1.0 + float(orientation_boost) * orient)


def compute_orientation_visible_energy(
    luminance: torch.Tensor,
    ecc_map: torch.Tensor,
    config,
) -> dict:
    """Simple orientation-sensitive variant based on Sobel energy."""

    orient = orientation_energy(luminance)
    if orient.dim() == 4:
        orient = orient[0]
    weighted = orient / orient.amax().clamp_min(1.0e-6)
    if ecc_map.dim() == 2:
        ecc = ecc_map.unsqueeze(0)
    else:
        ecc = ecc_map
    falloff = 1.0 / (1.0 + ecc.to(weighted.device, weighted.dtype) / 30.0)
    energy = weighted * falloff
    return {"orientation_energy": energy.unsqueeze(0), "orientation_count": 1}
