#!/usr/bin/env python3
"""Evaluate visible-band sequence outputs against full renders."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full vs method visible-band renders.")
    parser.add_argument("--full", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--config", default="configs/visible_band_default.yaml", type=str)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--fail_on_missing", action="store_true")
    return parser


def _resolve_dirs(path_text: str) -> Tuple[Path, Path]:
    path = Path(path_text)
    for child in ("frames", "renders"):
        if (path / child).is_dir():
            return path, path / child
    if path.name in ("frames", "renders"):
        return path.parent, path
    return path, path


def _collect_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def _metadata(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _runtime_imports() -> Any:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from visible_band.config import load_config
        from visible_band.metrics_visible_band import (
            compute_subthreshold_budget_waste,
            compute_visible_band_error,
            compute_visible_band_preservation,
        )
    except Exception as exc:
        raise RuntimeError("Evaluation requires numpy, PIL, torch, and visible_band modules.") from exc
    return type("Runtime", (), locals())


def _load_image(path: Path, rt: Any) -> Tuple[Any, Any]:
    image = rt.Image.open(path).convert("RGB")
    np_image = rt.np.asarray(image, dtype=rt.np.float32) / 255.0
    torch_image = rt.torch.from_numpy(np_image).permute(2, 0, 1).contiguous()
    return np_image, torch_image


def _psnr(a: Any, b: Any) -> float:
    import numpy as np

    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _uniform_filter2d(channel: Any, window: int) -> Any:
    import numpy as np

    window = max(1, min(int(window), channel.shape[0], channel.shape[1]))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return channel
    pad = window // 2
    padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]
    return total / float(window * window)


def _ssim(a: Any, b: Any, window: int = 11) -> float:
    import numpy as np

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for c in range(a.shape[2]):
        x = a[:, :, c]
        y = b[:, :, c]
        mux = _uniform_filter2d(x, window)
        muy = _uniform_filter2d(y, window)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = _uniform_filter2d(x * x, window) - mux2
        sigy2 = _uniform_filter2d(y * y, window) - muy2
        sigxy = _uniform_filter2d(x * y, window) - muxy
        scores.append(float(np.mean(((2.0 * muxy + c1) * (2.0 * sigxy + c2)) / np.maximum((mux2 + muy2 + c1) * (sigx2 + sigy2 + c2), 1.0e-12))))
    return float(sum(scores) / len(scores))


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else None


def _frame_meta(metadata: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    frames = metadata.get("frames", [])
    out: Dict[str, Mapping[str, Any]] = {}
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, Mapping):
                name = str(frame.get("image", f"{int(frame.get('frame', 0)):05d}.png"))
                out[name] = frame
    return out


def evaluate(full_path: str, method_path: str, config_path: str, fail_on_missing: bool = False) -> Dict[str, Any]:
    rt = _runtime_imports()
    cfg = rt.load_config(config_path)
    full_run, full_dir = _resolve_dirs(full_path)
    method_run, method_dir = _resolve_dirs(method_path)
    full_images = _collect_images(full_dir)
    method_images = _collect_images(method_dir)
    common = sorted(set(full_images) & set(method_images))
    missing_method = sorted(set(full_images) - set(method_images))
    missing_full = sorted(set(method_images) - set(full_images))
    if fail_on_missing and (missing_method or missing_full):
        raise RuntimeError(f"unmatched images: missing_method={missing_method}, missing_full={missing_full}")
    if not common:
        raise RuntimeError("No matching frames found.")
    method_meta = _frame_meta(_metadata(method_run))
    psnrs: List[float] = []
    ssims: List[float] = []
    preservations: List[float] = []
    errors: List[float] = []
    wastes: List[float] = []
    per_view = []
    for name in common:
        full_np, full_t = _load_image(full_images[name], rt)
        method_np, method_t = _load_image(method_images[name], rt)
        if full_np.shape != method_np.shape:
            raise RuntimeError(f"shape mismatch for {name}: {full_np.shape} vs {method_np.shape}")
        psnr = _psnr(full_np, method_np)
        ssim = _ssim(full_np, method_np)
        preservation = rt.compute_visible_band_preservation(full_t, method_t, tuple(cfg.gaze.gaze_uv), cfg)
        error = rt.compute_visible_band_error(full_t, method_t, tuple(cfg.gaze.gaze_uv), cfg)
        meta = dict(method_meta.get(name, {}))
        waste = float(meta.get("subthreshold_budget_waste", 0.0))
        psnrs.append(psnr)
        ssims.append(ssim)
        preservations.append(float(preservation["total_preservation"]))
        errors.append(float(error["total_error"]))
        wastes.append(waste)
        per_view.append(
            {
                "name": name,
                "psnr": psnr,
                "ssim": ssim,
                "visible_band_preservation": preservation,
                "visible_band_error": error,
                "subthreshold_budget_waste": waste,
                "method_metadata": meta,
            }
        )
    return {
        "full": str(full_run),
        "method": str(method_run),
        "image_pairs": len(common),
        "missing_from_method": missing_method,
        "missing_from_full": missing_full,
        "metrics": {
            "psnr_mean": _mean(psnrs),
            "ssim_mean": _mean(ssims),
            "visible_band_preservation_mean": _mean(preservations),
            "visible_band_error_mean": _mean(errors),
            "subthreshold_budget_waste_mean": _mean(wastes),
        },
        "per_view": per_view,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    result = evaluate(args.full, args.method, args.config, args.fail_on_missing)
    output = Path(args.output) if args.output else _resolve_dirs(args.method)[0] / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Image pairs: {result['image_pairs']}")
    print(f"PSNR mean: {result['metrics']['psnr_mean']}")
    print(f"SSIM mean: {result['metrics']['ssim_mean']}")
    print(f"Visible-band preservation mean: {result['metrics']['visible_band_preservation_mean']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
