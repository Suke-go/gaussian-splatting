#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import inspect
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _normalize_selected_mask(selected_mask, total_count, device):
    if torch.is_tensor(selected_mask):
        mask_or_indices = selected_mask.to(device=device)
    else:
        mask_or_indices = torch.as_tensor(selected_mask, device=device)

    mask_or_indices = mask_or_indices.reshape(-1)
    if mask_or_indices.dtype == torch.bool:
        if mask_or_indices.numel() != total_count:
            raise ValueError(
                f"selected_mask has {mask_or_indices.numel()} entries, expected {total_count}."
            )
        return mask_or_indices

    if mask_or_indices.numel() == total_count and (
        torch.is_floating_point(mask_or_indices)
        or bool(torch.logical_or(mask_or_indices == 0, mask_or_indices == 1).all().item())
    ):
        return mask_or_indices != 0

    selected_indices = mask_or_indices.to(dtype=torch.long)
    if selected_indices.numel() > 0:
        out_of_range = torch.logical_or(
            selected_indices < 0, selected_indices >= total_count
        ).any()
        if bool(out_of_range.item()):
            raise ValueError("selected_mask index entries are out of range.")

    selected_bool = torch.zeros(total_count, dtype=torch.bool, device=device)
    selected_bool[selected_indices] = True
    return selected_bool


def _select_override_color(
    override_color, selected_mask, selected_count, total_count, device, dtype
):
    if override_color is None:
        return None

    if not torch.is_tensor(override_color):
        override_color = torch.as_tensor(override_color, device=device, dtype=dtype)
    else:
        override_color = override_color.to(device=device, dtype=dtype)

    if override_color.ndim == 1:
        if override_color.numel() != 3:
            raise ValueError("1D override_color must contain exactly 3 RGB values.")
        return override_color.reshape(1, -1).expand(selected_count, -1)

    if override_color.ndim != 2 or override_color.shape[1] != 3:
        raise ValueError("override_color must have shape (3,), (N, 3), or (selected_N, 3).")

    if override_color.shape[0] == total_count:
        return override_color[selected_mask]

    if override_color.shape[0] == selected_count:
        return override_color

    raise ValueError(
        "override_color must be a single RGB color, full-size color tensor, "
        "or selected-size color tensor."
    )


def rasterize_selected_gaussians(
    viewpoint_camera,
    gaussians: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    selected_mask,
    scaling_modifier=1.0,
    override_color=None,
    *,
    separate_sh=False,
):
    """
    Render only the Gaussians selected by selected_mask.

    The rasterizer receives compact selected tensors, while returned radii,
    visibility_filter, and viewspace_points remain indexed like the official
    renderer so existing training code can consume them.
    """

    full_xyz = gaussians.get_xyz
    total_count = full_xyz.shape[0]
    device = full_xyz.device
    selected_mask = _normalize_selected_mask(selected_mask, total_count, device)
    selected_indices = selected_mask.nonzero().squeeze(1)
    selected_count = selected_indices.numel()

    screenspace_points = (
        torch.zeros_like(
            full_xyz, dtype=full_xyz.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    if selected_count == 0:
        zero_grad_link = screenspace_points.sum() * 0.0
        rendered_image = (
            bg_color.to(device=device, dtype=full_xyz.dtype)
            .reshape(3, 1, 1)
            .expand(3, image_height, image_width)
            .clone()
            + zero_grad_link
        )
        radii = torch.zeros(total_count, dtype=torch.int32, device=device)
        depth_image = (
            torch.zeros((1, image_height, image_width), dtype=full_xyz.dtype, device=device)
            + zero_grad_link
        )

        return {
            "render": rendered_image.clamp(0, 1),
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth_image,
            "selected_count": 0,
            "total_count": total_count,
            "selected_indices": selected_indices,
            "selected_mask": selected_mask,
            "selected_radii": radii[selected_mask],
            "selected_visibility_filter": (radii[selected_mask] > 0).nonzero(),
            "visible_selected_count": 0,
        }

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        antialiasing=getattr(pipe, "antialiasing", False),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = full_xyz[selected_mask]
    means2D = screenspace_points[selected_mask]
    opacity = gaussians.get_opacity[selected_mask]

    scales = None
    rotations = None
    cov3D_precomp = None

    if getattr(pipe, "compute_cov3D_python", False):
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)[selected_mask]
    else:
        scales = gaussians.get_scaling[selected_mask]
        rotations = gaussians.get_rotation[selected_mask]

    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pipe, "convert_SHs_python", False):
            selected_features = gaussians.get_features[selected_mask]
            shs_view = selected_features.transpose(1, 2).view(
                -1, 3, (gaussians.max_sh_degree + 1) ** 2
            )
            dir_pp = (
                means3D
                - viewpoint_camera.camera_center.repeat(selected_features.shape[0], 1)
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
                gaussians.active_sh_degree, shs_view, dir_pp_normalized
            )
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc = gaussians.get_features_dc[selected_mask]
                shs = gaussians.get_features_rest[selected_mask]
            else:
                shs = gaussians.get_features[selected_mask]
    else:
        colors_precomp = _select_override_color(
            override_color,
            selected_mask,
            selected_count,
            total_count,
            device,
            full_xyz.dtype,
        )

    rasterize_args = {
        "means3D": means3D,
        "means2D": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacities": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }
    rasterizer_params = inspect.signature(rasterizer.forward).parameters
    accepts_dc = "dc" in rasterizer_params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in rasterizer_params.values()
    )
    if separate_sh and dc is not None and accepts_dc:
        rasterize_args["dc"] = dc
    elif separate_sh and dc is not None:
        rasterize_args["shs"] = gaussians.get_features[selected_mask]

    rendered_image, selected_radii, depth_image = rasterizer(**rasterize_args)

    radii = torch.zeros(total_count, dtype=selected_radii.dtype, device=device)
    radii[selected_mask] = selected_radii
    selected_visibility_filter = (selected_radii > 0).nonzero()

    return {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "selected_count": int(selected_count),
        "total_count": total_count,
        "selected_indices": selected_indices,
        "selected_mask": selected_mask,
        "selected_radii": selected_radii,
        "selected_visibility_filter": selected_visibility_filter,
        "visible_selected_count": int((selected_radii > 0).sum().item()),
    }


__all__ = ["rasterize_selected_gaussians"]
