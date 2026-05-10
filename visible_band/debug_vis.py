"""Debug PNG writers for visible-band tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch


def tensor_to_uint8_image(
    tensor: torch.Tensor,
    normalize: bool = True,
    clamp: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Convert HW/CHW/NCHW tensor to uint8 HWC/gray image tensor."""

    x = tensor.detach().cpu().float()
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if normalize:
        min_value = x.min()
        max_value = x.max()
        x = (x - min_value) / (max_value - min_value).clamp_min(eps)
    elif clamp:
        x = x.clamp(0.0, 1.0)
    x = (x * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    if x.dim() == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def save_tensor_png(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = True,
    clamp: bool = True,
) -> str:
    """Save a tensor as PNG and return the written path."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to save debug PNGs") from exc
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_uint8_image(tensor, normalize=normalize, clamp=clamp).numpy()
    Image.fromarray(image).save(str(output_path))
    return str(output_path)


def save_debug_maps(
    maps: Mapping[str, torch.Tensor],
    output_dir: str,
    prefix: str = "",
    normalize: bool = True,
) -> Mapping[str, str]:
    """Save named tensors as PNGs in `output_dir`."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, tensor in maps.items():
        safe_name = name.replace("/", "_").replace("\\", "_")
        filename = "{}{}.png".format(prefix, safe_name)
        written[name] = save_tensor_png(tensor, str(out_dir / filename), normalize=normalize)
    return written


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Return an RGB tensor with a mask overlay for debug images."""

    img = image.detach().clone()
    if img.dim() == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    if img.dim() == 4:
        img = img[0]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if color is None:
        color_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=img.dtype, device=img.device)
    else:
        color_tensor = color.to(dtype=img.dtype, device=img.device)
    m = mask.detach().to(dtype=img.dtype, device=img.device)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 4:
        m = m[0]
    m = m[:1].clamp(0.0, 1.0)
    return img * (1.0 - alpha * m) + color_tensor.view(3, 1, 1) * (alpha * m)
