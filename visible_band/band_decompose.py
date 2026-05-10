"""Luminance conversion, band decomposition, and local contrast."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .config import VisibleBandConfig


def _as_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError("expected tensor shaped HW, CHW, or NCHW")


def _restore_chw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[0] == 1:
        return x[0]
    return x


def to_luminance(
    image: torch.Tensor,
    weights: Sequence[float] = (0.2126, 0.7152, 0.0722),
    keepdim: bool = True,
) -> torch.Tensor:
    """Convert RGB tensors to luminance.

    Official 3DGS tensors are CHW in [0, 1].  NCHW inputs are also accepted.
    """

    if image.dim() == 2:
        lum = image.unsqueeze(0)
        return lum if keepdim else image
    if image.shape[-3] == 1:
        return image if keepdim else image.squeeze(-3)
    if image.shape[-3] < 3:
        raise ValueError("RGB luminance conversion expects at least 3 channels")
    coeff = torch.tensor(weights, dtype=image.dtype, device=image.device)
    view_shape = [1] * image.dim()
    view_shape[-3] = 3
    lum = (image[..., :3, :, :] * coeff.view(*view_shape)).sum(dim=-3, keepdim=True)
    return lum if keepdim else lum.squeeze(-3)


def gaussian_kernel1d(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
    kernel_size: int = 0,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if kernel_size <= 0:
        radius = max(1, int(round(3.0 * sigma)))
        kernel_size = radius * 2 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=dtype, device=device) - center
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def gaussian_blur(image: torch.Tensor, sigma: float, kernel_size: int = 0) -> torch.Tensor:
    x = _as_nchw(image)
    kernel = gaussian_kernel1d(sigma, x.dtype, x.device, kernel_size)
    pad = kernel.numel() // 2
    channels = x.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    mode = "reflect"
    if x.shape[-1] <= pad or x.shape[-2] <= pad:
        mode = "replicate"
    out = F.pad(x, (pad, pad, 0, 0), mode=mode)
    out = F.conv2d(out, kernel_x, groups=channels)
    out = F.pad(out, (0, 0, pad, pad), mode=mode)
    out = F.conv2d(out, kernel_y, groups=channels)
    return _restore_chw(out)


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, VisibleBandConfig):
        return getattr(config.bands, name, default)
    if isinstance(config, Mapping):
        section = config.get("bands", config)
        if isinstance(section, Mapping):
            return section.get(name, default)
    return default


def decompose_bands(luminance: torch.Tensor, config: Any = None, **kwargs: Any) -> dict:
    """Decompose low-res luminance into same-resolution spatial bands.

    Return:
      {"bands": Tensor[B,H,W], "local_mean": Tensor[1,H,W]}
    """

    method = str(kwargs.get("method", _cfg_value(config, "type", "laplacian"))).lower()
    num_bands = int(kwargs.get("num_bands", _cfg_value(config, "num_bands", 4)))
    sigma_base = float(kwargs.get("sigma_base", _cfg_value(config, "sigma_base", 1.0)))
    sigma_scale = float(kwargs.get("sigma_scale", _cfg_value(config, "sigma_scale", 2.0)))
    local_mean_sigma = float(kwargs.get("local_mean_sigma", _cfg_value(config, "local_mean_sigma", 4.0)))
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    lum = luminance
    if lum.dim() == 2:
        lum = lum.unsqueeze(0)
    if lum.dim() == 4:
        if lum.shape[0] != 1:
            raise ValueError("decompose_bands expects a single luminance image")
        lum = lum[0]
    if lum.shape[0] != 1:
        raise ValueError("luminance must have one channel")

    bands = []
    if method in ("dog", "difference_of_gaussians", "difference-of-gaussians"):
        blurs = [
            gaussian_blur(lum, sigma_base * (sigma_scale ** idx))
            for idx in range(num_bands + 1)
        ]
        bands = [blurs[idx] - blurs[idx + 1] for idx in range(num_bands)]
        local_mean = blurs[-1].abs()
    elif method in ("laplacian", "laplace", "lap"):
        current = lum
        for idx in range(num_bands):
            low = gaussian_blur(current, sigma_base * (sigma_scale ** idx))
            bands.append(current - low)
            current = low
        local_mean = gaussian_blur(lum, local_mean_sigma).abs()
    else:
        raise ValueError(f"unknown band decomposition method: {method}")
    return {"bands": torch.cat(bands, dim=0), "local_mean": local_mean}


def compute_band_contrast(
    bands: torch.Tensor,
    local_mean: torch.Tensor | None = None,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """Compute C_b(x,y) = |Band_b(x,y)| / (local_mean(x,y) + eps)."""

    if isinstance(bands, Mapping):
        local_mean = bands.get("local_mean", local_mean)
        bands = bands["bands"]
    if bands.dim() == 4:
        if bands.shape[0] != 1:
            raise ValueError("compute_band_contrast expects a single band stack")
        bands = bands[0]
    if local_mean is None:
        local_mean = bands.abs().mean(dim=0, keepdim=True)
    if local_mean.dim() == 2:
        local_mean = local_mean.unsqueeze(0)
    if local_mean.dim() == 4:
        local_mean = local_mean[0]
    return bands.abs() / (local_mean.to(device=bands.device, dtype=bands.dtype) + float(eps))


def band_contrast(bands: torch.Tensor, local_mean: torch.Tensor | None = None, eps: float = 1.0e-4) -> torch.Tensor:
    return compute_band_contrast(bands, local_mean=local_mean, eps=eps)


def decompose_laplacian(image: torch.Tensor, num_bands: int = 4, **kwargs: Any) -> list[torch.Tensor]:
    result = decompose_bands(image, {"bands": {"type": "laplacian", "num_bands": num_bands}}, **kwargs)
    return [result["bands"][idx : idx + 1] for idx in range(result["bands"].shape[0])]


def decompose_dog(image: torch.Tensor, num_bands: int = 4, **kwargs: Any) -> list[torch.Tensor]:
    result = decompose_bands(image, {"bands": {"type": "dog", "num_bands": num_bands}}, **kwargs)
    return [result["bands"][idx : idx + 1] for idx in range(result["bands"].shape[0])]
