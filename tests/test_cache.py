import json

import pytest
import torch

from gendsr.cache import FeatureCache, FeatureCacheWriter, WAN_IDENTITY, sha256_file
from gendsr.video import FRAME_POLICY_SHA256, dyn_policy_sha256, uniform_frame_indices
from gendsr.wan_schema import compute_wan_grid_shapes


def _metadata(video):
    return {
        **WAN_IDENTITY,
        "sample_id": "clip",
        "source_num_frames": 32,
        "source_frame_indices": uniform_frame_indices(10),
        "frame_policy_sha256": FRAME_POLICY_SHA256,
        "video_sha256": sha256_file(video),
        "video_mode": "normal",
        "generator_token_grid_shape": [1, 1, 1],
    }


def test_writer_reader_identity_and_coverage(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original")
    root = tmp_path / "cache"
    FeatureCacheWriter(root).write(torch.ones(1, 1, 1, 1536, dtype=torch.bfloat16), _metadata(video))
    cache = FeatureCache(root)
    cache.check_coverage([{"vgm_feature_key": "clip"}])
    assert cache.load("clip", video_path=video, frame_indices=uniform_frame_indices(10)).shape == (1, 1, 1, 1536)
    with pytest.raises(ValueError, match="frame indices"):
        cache.load("clip", frame_indices=[0] * 32)
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source video identity"):
        cache.load("clip", video_path=video)
    with pytest.raises(ValueError, match="sample count mismatch"):
        cache.check_coverage([{"vgm_feature_key": "another"}])
    (root / "summary.json").write_text(
        json.dumps({"annotation_rows": 2, "cached_videos": 1}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample count mismatch"):
        cache.check_coverage([{"vgm_feature_key": "clip"}])
    with pytest.raises(ValueError, match="video mode"):
        FeatureCache(root, expected_mode="reversed-video")


def test_wan_token_shape():
    shape = compute_wan_grid_shapes(
        source_num_frames=32, generator_input_height_width=(480, 832),
    )
    assert shape["generator_token_grid_shape"] == [8, 30, 52]
    assert shape["generator_num_tokens"] == 8 * 30 * 52


def test_official_dyn_frame_policy_is_checked(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    official = [0, 2, 4, 6, 8]
    metadata = _metadata(video) | {
        "frame_policy": "dyn_official",
        "frame_policy_sha256": dyn_policy_sha256(official),
        "official_frame_indices": official,
        "original_source_num_frames": 10,
        "source_num_frames": len(official),
        "source_frame_indices": official,
    }
    root = tmp_path / "dyn-cache"
    FeatureCacheWriter(root).write(
        torch.ones(1, 1, 1, 1536, dtype=torch.bfloat16), metadata,
    )
    cache = FeatureCache(root, expected_policy="dyn_official")
    row = {
        "sample_id": "qa", "vgm_feature_key": "clip",
        "frame_policy": "dyn_official", "official_frame_indices": official,
        "original_source_num_frames": 10,
    }
    cache.check_coverage([row])
    assert cache.load("clip", frame_indices=official).shape[-1] == 1536
    with pytest.raises(ValueError, match="frame policy"):
        FeatureCache(root)
    with pytest.raises(ValueError, match="official frame indices"):
        cache.check_coverage([{**row, "official_frame_indices": [0, 1]}])
