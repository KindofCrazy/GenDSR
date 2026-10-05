"""One uniform frame policy shared by feature extraction and Qwen."""

from __future__ import annotations

import hashlib
import json
from types import MethodType

import numpy as np

NUM_FRAMES = 32
MIN_PIXELS = 784
MAX_PIXELS = 230400
FRAME_POLICY = {"sampling": "endpoint_linspace_round", "num_frames": NUM_FRAMES}
FRAME_POLICY_SHA256 = hashlib.sha256(
    json.dumps(FRAME_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def apply_video_mode(indices: list[int], mode: str = "normal") -> list[int]:
    if not indices:
        raise ValueError("Frame index list is empty")
    if mode == "normal":
        return list(indices)
    if mode == "repeat-first":
        return [indices[0]] * len(indices)
    if mode == "reversed-video":
        return list(reversed(indices))
    raise ValueError(f"Unsupported video mode: {mode}")


def uniform_frame_indices(total_frames: int, mode: str = "normal") -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES).round().astype(int).tolist()
    return apply_video_mode(indices, mode)


def dyn_policy_sha256(indices: list[int]) -> str:
    if not indices or any(type(value) is not int or value < 0 for value in indices):
        raise ValueError("Official frame indices must be non-empty non-negative integers")
    return hashlib.sha256(
        json.dumps({"sampling": "dyn_official", "indices": indices}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def decode_video(
    path: str, mode: str = "normal", *, frame_indices: list[int] | None = None,
    expected_total_frames: int | None = None,
):
    """Return RGB frames, indices, fps, and source shape using decord."""
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("Video decoding requires the SFT or Wan environment") from exc
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    if expected_total_frames is not None and len(reader) != expected_total_frames:
        raise ValueError(
            f"Official source frame count mismatch: expected {expected_total_frames}, "
            f"decoded {len(reader)}"
        )
    if frame_indices is None:
        indices = uniform_frame_indices(len(reader), mode)
    else:
        if not frame_indices or min(frame_indices) < 0 or max(frame_indices) >= len(reader):
            raise ValueError("Official frame indices are outside the source video")
        indices = apply_video_mode(frame_indices, mode)
    frames = reader.get_batch(indices).asnumpy()
    fps = float(reader.get_avg_fps())
    return frames, indices, fps, list(frames.shape[1:3])


def configure_qwen_video_processor(processor, mode: str = "normal") -> None:
    """Force the same frame indices and pixel budget in Transformers."""
    video = getattr(processor, "video_processor", None)
    if video is None:
        raise ValueError("Qwen processor has no video_processor")

    def sample_frames(self, metadata, num_frames=None, fps=None, **kwargs):
        del self, num_frames, fps, kwargs
        explicit = getattr(video, "_gendsr_explicit_indices", None)
        expected_total = getattr(video, "_gendsr_expected_total_frames", None)
        if expected_total is not None and int(metadata.total_num_frames) != expected_total:
            raise ValueError("Official source frame count mismatch")
        if explicit is None:
            indices = uniform_frame_indices(int(metadata.total_num_frames), mode)
        else:
            if max(explicit) >= int(metadata.total_num_frames):
                raise ValueError("Official frame indices are outside the source video")
            indices = apply_video_mode(explicit, mode)
        return np.asarray(indices)

    video.sample_frames = MethodType(sample_frames, video)
    video.fps = None
    video.min_frames = NUM_FRAMES
    video.max_frames = NUM_FRAMES
    if hasattr(video, "max_pixels"):
        video.max_pixels = MAX_PIXELS
    if hasattr(video, "min_pixels"):
        video.min_pixels = MIN_PIXELS
    if hasattr(video, "size"):
        size = video.size
        if isinstance(size, dict):
            size["shortest_edge"] = MIN_PIXELS * NUM_FRAMES
            size["longest_edge"] = MAX_PIXELS * NUM_FRAMES
        else:
            size.shortest_edge = MIN_PIXELS * NUM_FRAMES
            size.longest_edge = MAX_PIXELS * NUM_FRAMES

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        from transformers.video_utils import load_video

        paths = video_url_or_urls if isinstance(video_url_or_urls, list) else [video_url_or_urls]
        loaded = [
            load_video(path, backend="decord", sample_indices_fn=sample_indices_fn)
            for path in paths
        ]
        self._gendsr_video_metadata = [metadata for _, metadata in loaded]
        if isinstance(video_url_or_urls, list):
            return list(zip(*loaded))
        return loaded[0]

    video.fetch_videos = MethodType(fetch_videos, video)


def set_qwen_frame_count(processor, count: int) -> None:
    if count <= 0:
        raise ValueError("Frame count must be positive")
    video = processor.video_processor
    video.min_frames = count
    video.max_frames = count
    if hasattr(video, "size"):
        size = video.size
        if isinstance(size, dict):
            size["shortest_edge"] = MIN_PIXELS * count
            size["longest_edge"] = MAX_PIXELS * count
        else:
            size.shortest_edge = MIN_PIXELS * count
            size.longest_edge = MAX_PIXELS * count
