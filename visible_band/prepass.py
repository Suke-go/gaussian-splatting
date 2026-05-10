"""Low-resolution render prepass for visible-band estimation."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from .band_decompose import to_luminance


def render_lowres_prepass(
    viewpoint_camera: Any,
    gaussians: Any,
    pipe: Any,
    bg_color: torch.Tensor,
    resolution: tuple[int, int],
    mode: str = "luminance",
) -> dict:
    """Render a low-resolution prepass with the official rasterizer.

    Return:
      {"luminance": Tensor[1,H,W], "depth": Tensor[1,H,W] | None, "alpha": None}
    """

    from gaussian_renderer import render

    width, height = int(resolution[0]), int(resolution[1])
    lowres_camera = copy.copy(viewpoint_camera)
    lowres_camera.image_width = width
    lowres_camera.image_height = height
    out = render(lowres_camera, gaussians, pipe, bg_color)
    image = out["render"]
    if image.shape[-2:] != (height, width):
        image = F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    luminance = to_luminance(image, keepdim=True)
    depth = out.get("depth")
    if depth is not None:
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        if depth.shape[-2:] != (height, width):
            depth = F.interpolate(depth.unsqueeze(0), size=(height, width), mode="nearest")[0]
    return {
        "luminance": luminance if mode == "luminance" else image,
        "depth": depth,
        "alpha": None,
    }
