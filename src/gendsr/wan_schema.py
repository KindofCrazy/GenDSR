"""Cache schema and shape utilities for clip-level Wan feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_WAN_CONFIG: Dict[str, Any] = {
    "generator_model": "Wan2.1-T2V-1.3B",
    "generator_task": "t2v-1.3B",
    "generator_feature_scope": "clip_level_wan_feature",
    "generator_layer": 20,
    "diffusion_timestep": 300,
    "generator_feat_dim": 1536,
    "generator_vae_stride": (4, 8, 8),
    "generator_patch_size": (1, 2, 2),
    "generator_input_resize": "wan_native_resize_crop",
    "canonical_num_frames": 32,
}


@dataclass(frozen=True)
class WanFeatureConfig:
    generator_model: str = "Wan2.1-T2V-1.3B"
    generator_task: str = "t2v-1.3B"
    generator_feature_scope: str = "clip_level_wan_feature"
    generator_layer: int = 20
    diffusion_timestep: int = 300
    generator_feat_dim: int = 1536
    generator_vae_stride: Tuple[int, int, int] = (4, 8, 8)
    generator_patch_size: Tuple[int, int, int] = (1, 2, 2)
    generator_input_resize: str = "wan_native_resize_crop"
    canonical_num_frames: int = 32
    generator_size_key: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["generator_vae_stride"] = list(self.generator_vae_stride)
        data["generator_patch_size"] = list(self.generator_patch_size)
        return data


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _wan_temporal_latent_length(num_frames: int, stride_t: int) -> int:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return (num_frames - 1) // stride_t + 1


def compute_wan_grid_shapes(
    *,
    source_num_frames: int,
    generator_input_height_width: Sequence[int],
    vae_stride: Sequence[int] = DEFAULT_WAN_CONFIG["generator_vae_stride"],
    patch_size: Sequence[int] = DEFAULT_WAN_CONFIG["generator_patch_size"],
    latent_channels: int = 16,
) -> Dict[str, Any]:
    """Compute Wan latent and denoiser token grids for a clip-level input.

    Wan2.1 T2V expects latent tensors shaped [C, F_latent, H/8, W/8]. Its DiT
    patch embedding uses patch_size=(1,2,2), so token grids are lower spatial
    resolution than the VAE latent grid.
    """
    if len(generator_input_height_width) != 2:
        raise ValueError("generator_input_height_width must be [height, width]")
    if len(vae_stride) != 3 or len(patch_size) != 3:
        raise ValueError("vae_stride and patch_size must be length-3 sequences")

    height, width = [int(x) for x in generator_input_height_width]
    stride_t, stride_h, stride_w = [int(x) for x in vae_stride]
    patch_t, patch_h, patch_w = [int(x) for x in patch_size]

    latent_t = _wan_temporal_latent_length(int(source_num_frames), stride_t)
    latent_h = height // stride_h
    latent_w = width // stride_w
    token_t = _ceil_div(latent_t, patch_t)
    token_h = latent_h // patch_h
    token_w = latent_w // patch_w

    return {
        "generator_latent_grid_shape": [latent_t, latent_h, latent_w],
        "generator_latent_tensor_shape": [latent_channels, latent_t, latent_h, latent_w],
        "generator_token_grid_shape": [token_t, token_h, token_w],
        "generator_num_tokens": int(token_t * token_h * token_w),
        "generator_effective_downsample": [
            stride_t * patch_t,
            stride_h * patch_h,
            stride_w * patch_w,
        ],
    }


def build_cache_metadata(
    *,
    sample_id: str,
    video_path: str,
    source_num_frames: int,
    source_frame_indices: Sequence[int],
    source_timestamps_sec: Sequence[float],
    source_height_width: Sequence[int],
    generator_input_height_width: Sequence[int],
    source_padding_policy: str = "none",
    config: WanFeatureConfig = WanFeatureConfig(),
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable metadata record for one cached feature tensor."""
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "sample_id": str(sample_id),
        "video_path": str(video_path),
        "source_num_frames": int(source_num_frames),
        "source_frame_indices": [int(x) for x in source_frame_indices],
        "source_timestamps_sec": [float(x) for x in source_timestamps_sec],
        "source_height_width": [int(x) for x in source_height_width],
        "generator_input_height_width": [int(x) for x in generator_input_height_width],
        "source_padding_policy": source_padding_policy,
        "feature_layout": "T,H,W,C",
    }
    metadata.update(config.to_dict())
    metadata.update(
        compute_wan_grid_shapes(
            source_num_frames=source_num_frames,
            generator_input_height_width=generator_input_height_width,
            vae_stride=config.generator_vae_stride,
            patch_size=config.generator_patch_size,
        )
    )
    if extra:
        metadata.update(dict(extra))
    return metadata


REQUIRED_METADATA_KEYS = {
    "schema_version",
    "sample_id",
    "video_path",
    "source_num_frames",
    "source_frame_indices",
    "source_timestamps_sec",
    "source_height_width",
    "generator_input_height_width",
    "source_padding_policy",
    "generator_model",
    "generator_task",
    "generator_feature_scope",
    "generator_layer",
    "diffusion_timestep",
    "generator_vae_stride",
    "generator_patch_size",
    "generator_latent_grid_shape",
    "generator_token_grid_shape",
    "generator_effective_downsample",
    "generator_feat_dim",
    "feature_layout",
}


def validate_cache_record(record: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_METADATA_KEYS.difference(record.keys()))
    if missing:
        raise ValueError(f"cache metadata missing required keys: {missing}")

    frame_count = int(record["source_num_frames"])
    if len(record["source_frame_indices"]) != frame_count:
        raise ValueError("source_frame_indices length does not match source_num_frames")
    if len(record["source_timestamps_sec"]) != frame_count:
        raise ValueError("source_timestamps_sec length does not match source_num_frames")
    if record["generator_feature_scope"] != "clip_level_wan_feature":
        raise ValueError("infra mainline expects clip_level_wan_feature")
    if record["feature_layout"] != "T,H,W,C":
        raise ValueError("infra mainline expects feature_layout=T,H,W,C")


def validate_index_records(records: Iterable[Mapping[str, Any]]) -> None:
    for record in records:
        validate_cache_record(record)

