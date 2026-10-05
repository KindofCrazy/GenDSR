"""The single supported method and training recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_FUSION = {
    "format_version": 3,
    "feature": {"dim": 1536},
    "alignment": {"type": "temporal_interpolate_spatial_pool", "spatial_merge_size": 2},
    "branches": [
        {"name": "raw", "operator": "identity", "projection": "raw"},
        {
            "name": "delta", "operator": "finite_difference", "projection": "delta",
            "order": 1, "stride": 1, "compute_order": "delta_then_resize",
            "temporal_upsample": "interpolate",
        },
    ],
    "branch_gate": {
        "type": "conditioned", "activation": "direct",
        "condition_on_semantic": True, "condition_on_task": True,
        "raw_weight": "learned",
    },
    "branch_reducer": {"type": "additive"},
    "outer_gate": {"type": "fixed", "init": 0.0, "fixed_value": 1.0},
    "outer_reducer": {"type": "additive"},
    "initialization": {"delta_scale": 0.1},
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("fusion") != _FUSION:
        raise ValueError("The configuration must contain the supported GenDSR fusion method")
    if config.get("video", {}).get("num_frames") != 32:
        raise ValueError("The video policy requires 32 uniformly sampled frames")
    if config.get("video", {}).get("max_pixels") != 230400:
        raise ValueError("The video policy requires max_pixels=230400")
    if config.get("video", {}).get("min_pixels") != 784:
        raise ValueError("The video policy requires min_pixels=784")
    wan = config.get("wan", {})
    if wan != {"model": "Wan2.1-T2V-1.3B", "layer": 20, "timestep": 300, "dtype": "bfloat16"}:
        raise ValueError("Wan settings must use the fixed model, layer, timestep, and dtype")
    return config


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "gendsr.yaml"
