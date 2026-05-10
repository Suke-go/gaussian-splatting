"""Visible-band control utilities for Gaussian Splatting experiments.

This package is intentionally independent from the official 3DGS training,
COLMAP loading, renderer, and densification code.  Torch-dependent helpers are
loaded lazily so config loading remains lightweight.
"""

from importlib import import_module

from .config import (
    BandDecompositionConfig,
    BudgetConfig,
    SelectorConfig,
    VisibilityConfig,
    VisibleBandConfig,
    load_config,
    load_yaml_config,
)


_LAZY_ATTRS = {
    "allocate_tile_budget": ".budget",
    "band_contrast": ".band_decompose",
    "compute_band_contrast": ".band_decompose",
    "compute_gaussian_scores": ".selector",
    "compute_gaussian_contrast_proxy": ".selector",
    "compute_depth_discontinuity_bonus": ".branches_depth",
    "compute_eccentricity_map": ".visibility",
    "compute_selection_churn": ".metrics_visible_band",
    "compute_visible_band_contrast_error": ".metrics_visible_band",
    "compute_visible_band_energy_error": ".metrics_visible_band",
    "compute_visible_band_error": ".metrics_visible_band",
    "compute_visible_band_mae": ".metrics_visible_band",
    "compute_visible_band_metrics": ".metrics_visible_band",
    "compute_visible_band_preservation": ".metrics_visible_band",
    "compute_visible_band_rmse": ".metrics_visible_band",
    "compute_visible_contrast_error": ".metrics_visible_band",
    "decompose_bands": ".band_decompose",
    "decompose_dog": ".band_decompose",
    "decompose_laplacian": ".band_decompose",
    "eccentricity_map": ".visibility",
    "estimate_visible_band_energy": ".visibility",
    "gaussian_blur": ".band_decompose",
    "gaussian_tile_bounds": ".project",
    "match_global_budget": ".budget",
    "metric_visible_band": ".metrics_visible_band",
    "points_in_view": ".project",
    "project_gaussians_to_screen": ".project",
    "project_points": ".project",
    "render_lowres_prepass": ".prepass",
    "score_gaussians": ".selector",
    "select_gaussians": ".selector",
    "select_gaussians_by_tile_budget": ".selector",
    "selected_mask_summary": ".selector",
    "aggregate_visible_energy_to_tiles": ".visibility",
    "tile_aggregate": ".visibility",
    "to_luminance": ".band_decompose",
    "visible_band_contrast_error": ".metrics_visible_band",
    "visible_band_energy_error": ".metrics_visible_band",
    "visible_band_mae": ".metrics_visible_band",
    "visible_band_metrics": ".metrics_visible_band",
    "visible_band_rmse": ".metrics_visible_band",
    "visible_energy": ".visibility",
    "TemporalState": ".branches_temporal",
    "apply_temporal_hysteresis": ".branches_temporal",
    "compute_orientation_visible_energy": ".branches_orientation",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError("module 'visible_band' has no attribute '{}'".format(name))
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "BandDecompositionConfig",
    "BudgetConfig",
    "SelectorConfig",
    "VisibilityConfig",
    "VisibleBandConfig",
    "allocate_tile_budget",
    "aggregate_visible_energy_to_tiles",
    "band_contrast",
    "compute_band_contrast",
    "compute_eccentricity_map",
    "compute_depth_discontinuity_bonus",
    "compute_gaussian_contrast_proxy",
    "compute_gaussian_scores",
    "compute_selection_churn",
    "compute_visible_band_contrast_error",
    "compute_visible_band_energy_error",
    "compute_visible_band_error",
    "compute_visible_band_mae",
    "compute_visible_band_metrics",
    "compute_visible_band_preservation",
    "compute_visible_band_rmse",
    "compute_visible_contrast_error",
    "decompose_bands",
    "decompose_dog",
    "decompose_laplacian",
    "eccentricity_map",
    "estimate_visible_band_energy",
    "gaussian_blur",
    "gaussian_tile_bounds",
    "load_config",
    "load_yaml_config",
    "match_global_budget",
    "metric_visible_band",
    "points_in_view",
    "project_gaussians_to_screen",
    "project_points",
    "render_lowres_prepass",
    "score_gaussians",
    "select_gaussians",
    "select_gaussians_by_tile_budget",
    "selected_mask_summary",
    "tile_aggregate",
    "to_luminance",
    "visible_band_contrast_error",
    "visible_band_energy_error",
    "visible_band_mae",
    "visible_band_metrics",
    "visible_band_rmse",
    "visible_energy",
    "TemporalState",
    "apply_temporal_hysteresis",
    "compute_orientation_visible_energy",
]
