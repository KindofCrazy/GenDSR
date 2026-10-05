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


def uniform_frame_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES).round().astype(int).tolist()
    return indices


def decode_video(path: str):
    """Return RGB frames, indices, fps, and source shape using decord."""
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("Video decoding requires the SFT or Wan environment") from exc
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    indices = uniform_frame_indices(len(reader))
    frames = reader.get_batch(indices).asnumpy()
    fps = float(reader.get_avg_fps())
    return frames, indices, fps, list(frames.shape[1:3])


def configure_qwen_video_processor(processor) -> None:
    """Force the same frame indices and pixel budget in Transformers."""
    video = getattr(processor, "video_processor", None)
    if video is None:
        raise ValueError("Qwen processor has no video_processor")

    def sample_frames(self, metadata, num_frames=None, fps=None, **kwargs):
        del self, num_frames, fps, kwargs
        indices = uniform_frame_indices(int(metadata.total_num_frames))
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
