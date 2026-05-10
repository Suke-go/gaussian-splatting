"""Configuration for Visible-Band Splat Budgeting.

The public config follows the v0.1 research spec.  PyYAML is used when
available, with a small fallback parser so `--help` and config smoke tests do
not require the full 3DGS environment.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


@dataclass
class MethodConfig:
    name: str = "visible_band"
    use_depth_branch: bool = False
    use_temporal_branch: bool = False
    use_orientation_branch: bool = False


@dataclass
class RenderConfig:
    target_width: int = 1920
    target_height: int = 1080
    lowres_width: int = 256
    lowres_height: int = 256
    tile_size: int = 32
    background: str = "model"
    scaling_modifier: float = 1.0


@dataclass
class GazeConfig:
    mode: str = "fixed"
    gaze_uv: Tuple[float, float] = (0.5, 0.5)
    fov_x_deg: float = 100.0
    fov_y_deg: float = 100.0


@dataclass
class BandsConfig:
    type: str = "laplacian"
    num_bands: int = 4
    names: Tuple[str, ...] = ("very_low", "low", "mid", "high")
    sigma_base: float = 1.0
    sigma_scale: float = 2.0
    local_mean_sigma: float = 4.0


@dataclass
class ThresholdBandConfig:
    base: float
    ecc_slope: float


@dataclass
class VisibilityThresholdConfig:
    very_low: ThresholdBandConfig = field(default_factory=lambda: ThresholdBandConfig(0.02, 0.005))
    low: ThresholdBandConfig = field(default_factory=lambda: ThresholdBandConfig(0.03, 0.010))
    mid: ThresholdBandConfig = field(default_factory=lambda: ThresholdBandConfig(0.05, 0.030))
    high: ThresholdBandConfig = field(default_factory=lambda: ThresholdBandConfig(0.08, 0.070))


@dataclass
class BudgetConfig:
    fovea_deg: float = 5.0
    parafovea_deg: float = 15.0
    keep_ratio_fovea: float = 1.0
    keep_ratio_parafovea: float = 0.50
    keep_ratio_periphery: float = 0.25
    lambda_visible: float = 0.25
    min_keep_ratio: float = 0.05
    max_keep_ratio: float = 1.0
    match_global_budget: bool = True
    target_global_budget: Optional[int] = None


@dataclass
class ScoreConfig:
    weight_visible_energy: float = 1.0
    weight_opacity: float = 1.0
    weight_projected_area: float = 0.5
    weight_contrast_proxy: float = 0.5


@dataclass
class DepthBranchConfig:
    lambda_depth: float = 0.20
    depth_gradient_threshold: float = 0.10


@dataclass
class TemporalBranchConfig:
    enter_threshold: float = 0.50
    keep_threshold: float = 0.35
    lambda_temporal: float = 0.15


@dataclass
class DebugConfig:
    save_visible_band_maps: bool = True
    save_budget_maps: bool = True
    save_selected_masks: bool = True
    save_band_contrast_maps: bool = True
    save_eccentricity_maps: bool = True
    representative_frames: Tuple[int, ...] = (0,)


@dataclass
class VisibleBandConfig:
    method: MethodConfig = field(default_factory=MethodConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    bands: BandsConfig = field(default_factory=BandsConfig)
    visibility_threshold: VisibilityThresholdConfig = field(default_factory=VisibilityThresholdConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    depth_branch: DepthBranchConfig = field(default_factory=DepthBranchConfig)
    temporal_branch: TemporalBranchConfig = field(default_factory=TemporalBranchConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


# Backward-compatible names from the first implementation slice.
BandDecompositionConfig = BandsConfig
VisibilityConfig = GazeConfig
SelectorConfig = ScoreConfig


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        loaded = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
        except ImportError:
            loaded = _parse_minimal_yaml(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("visible-band config root must be a mapping")
    return dict(loaded)


def load_config(
    path_or_mapping: Optional[Any] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> VisibleBandConfig:
    data: Dict[str, Any] = {}
    if path_or_mapping is not None:
        if isinstance(path_or_mapping, (str, Path)):
            data = load_yaml_config(str(path_or_mapping))
        elif isinstance(path_or_mapping, Mapping):
            data = dict(path_or_mapping)
        else:
            raise TypeError("path_or_mapping must be a path, mapping, or None")
    if overrides:
        data = _deep_update(data, dict(overrides))
    cfg = _from_mapping(data)
    validate_config(cfg)
    return cfg


def validate_config(config: VisibleBandConfig) -> None:
    if config.render.tile_size < 1:
        raise ValueError("render.tile_size must be >= 1")
    if config.render.lowres_width < 1 or config.render.lowres_height < 1:
        raise ValueError("lowres dimensions must be positive")
    if config.bands.num_bands < 1:
        raise ValueError("bands.num_bands must be >= 1")
    if len(config.bands.names) < config.bands.num_bands:
        raise ValueError("bands.names must contain at least bands.num_bands names")
    if config.budget.min_keep_ratio < 0.0 or config.budget.max_keep_ratio <= 0.0:
        raise ValueError("budget keep ratios must be positive")
    if config.budget.min_keep_ratio > config.budget.max_keep_ratio:
        raise ValueError("min_keep_ratio must be <= max_keep_ratio")


def to_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("to_dict expects a dataclass or mapping")


def _from_mapping(data: Mapping[str, Any]) -> VisibleBandConfig:
    return VisibleBandConfig(
        method=_merge_dataclass(MethodConfig, data.get("method", {})),
        render=_merge_dataclass(RenderConfig, data.get("render", {})),
        gaze=_merge_dataclass(GazeConfig, data.get("gaze", {})),
        bands=_merge_dataclass(BandsConfig, data.get("bands", {})),
        visibility_threshold=VisibilityThresholdConfig(
            very_low=_threshold(data, "very_low", 0.02, 0.005),
            low=_threshold(data, "low", 0.03, 0.010),
            mid=_threshold(data, "mid", 0.05, 0.030),
            high=_threshold(data, "high", 0.08, 0.070),
        ),
        budget=_merge_dataclass(BudgetConfig, data.get("budget", {})),
        score=_merge_dataclass(ScoreConfig, data.get("score", {})),
        depth_branch=_merge_dataclass(DepthBranchConfig, data.get("depth_branch", {})),
        temporal_branch=_merge_dataclass(TemporalBranchConfig, data.get("temporal_branch", {})),
        debug=_merge_dataclass(DebugConfig, data.get("debug", {})),
    )


def _threshold(data: Mapping[str, Any], name: str, base: float, slope: float) -> ThresholdBandConfig:
    section = data.get("visibility_threshold", {})
    values = section.get(name, {}) if isinstance(section, Mapping) else {}
    if not isinstance(values, Mapping):
        values = {}
    return ThresholdBandConfig(
        base=float(values.get("base", base)),
        ecc_slope=float(values.get("ecc_slope", slope)),
    )


def _merge_dataclass(cls: Any, values: Any) -> Any:
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    defaults = cls()
    kwargs = {}
    for name in defaults.__dataclass_fields__:
        if name in values:
            value = values[name]
            default_value = getattr(defaults, name)
            if isinstance(default_value, tuple) and isinstance(value, list):
                value = tuple(value)
            kwargs[name] = value
    return cls(**kwargs)


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(raw_value.strip())
    return root


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return parsed
    except Exception:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")
