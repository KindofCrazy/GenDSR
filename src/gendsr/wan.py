"""Extract fixed Wan features for every unique local video in prepared data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cache import FeatureCacheWriter, sha256_file
from .config import load_config
from .data import load_prepared
from .video import FRAME_POLICY_SHA256, decode_video
from .wan_schema import WanFeatureConfig, build_cache_metadata

_SIZES = {"landscape": (480, 832), "portrait": (832, 480)}


def _resize_center_crop(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = target
    output = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        source_width, source_height = image.size
        scale = max(width / source_width, height / source_height)
        new_width = max(width, round(source_width * scale))
        new_height = max(height, round(source_height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        output.append(np.asarray(image.crop((left, top, left + width, top + height))))
    return np.stack(output, axis=0)


def unique_videos(rows: list[dict]) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for row in rows:
        sample_id = str(row["vgm_feature_key"])
        path = Path(row["video"]).resolve()
        if row.get("frame_policy", "uniform32") != "uniform32":
            raise ValueError(f"Unsupported frame policy for {sample_id}")
        existing = videos.get(sample_id)
        if existing is not None and existing != path:
            raise ValueError(f"videoID={sample_id} has conflicting video paths")
        videos[sample_id] = path
    return videos


def extract(
    *, rows: list[dict], cache_root: Path, checkpoint_dir: Path, wan_repo: Path,
    config_path: Path, device: str = "cuda",
) -> dict[str, int]:
    import torch

    load_config(config_path)
    videos = unique_videos(rows)
    if not videos:
        raise ValueError("Prepared data has no videos to extract")
    if not checkpoint_dir.is_dir() or not wan_repo.is_dir():
        raise FileNotFoundError("Wan checkpoint and official Wan source directories are required")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Wan extraction requires an available CUDA device")
    writer = FeatureCacheWriter(cache_root)
    from .wan_extractor import WanClipFeatureExtractor

    extractor = WanClipFeatureExtractor(
        checkpoint_dir=str(checkpoint_dir), wan_repo=str(wan_repo),
        config=WanFeatureConfig(), device=device,
    )
    for sample_id, path in videos.items():
        frames, indices, fps, source_hw = decode_video(str(path))
        size = _SIZES["landscape" if source_hw[1] >= source_hw[0] else "portrait"]
        resized = _resize_center_crop(frames, size)
        metadata = build_cache_metadata(
            sample_id=sample_id,
            video_path=str(path),
            source_num_frames=len(indices),
            source_frame_indices=indices,
            source_timestamps_sec=[float(index / fps) for index in indices],
            source_height_width=source_hw,
            generator_input_height_width=size,
            config=WanFeatureConfig(),
            extra={
                "frame_policy": "uniform32",
                "frame_policy_sha256": FRAME_POLICY_SHA256,
                "canonical_num_frames": len(indices),
                "video_sha256": sha256_file(path),
                "video_mode": "normal",
                "generator_condition": "wan_t5_empty_prompt",
                "video_fps": fps,
            },
        )
        feature = extractor.extract(resized, metadata=metadata)
        writer.write(feature, metadata)
        print(f"cached {sample_id}", flush=True)
    summary = {"annotation_rows": len(rows), "cached_videos": len(videos)}
    (cache_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--wan-checkpoint", required=True, type=Path)
    parser.add_argument("--wan-repo", required=True, type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args(argv)
    rows = load_prepared(args.data, expected_rows=args.expected_samples)
    print(json.dumps(extract(
        rows=rows, cache_root=args.cache_root, checkpoint_dir=args.wan_checkpoint,
        wan_repo=args.wan_repo, config_path=args.config, device=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
