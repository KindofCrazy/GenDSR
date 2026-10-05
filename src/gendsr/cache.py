"""Portable Wan feature cache shared by extraction, training, and evaluation."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch

from .video import (
    FRAME_POLICY_SHA256, NUM_FRAMES, apply_video_mode, dyn_policy_sha256,
)

WAN_IDENTITY = {
    "generator_model": "Wan2.1-T2V-1.3B",
    "generator_task": "t2v-1.3B",
    "generator_layer": 20,
    "diffusion_timestep": 300,
    "generator_feat_dim": 1536,
    "generator_condition": "wan_t5_empty_prompt",
    "generator_scheduler_shift": 5.0,
    "generator_noise_seed": 42,
    "feature_layout": "T,H,W,C",
}


def sha256_file(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_file_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8192)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _safe_name(sample_id: str) -> str:
    if not sample_id or sample_id in {".", ".."} or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in sample_id
    ):
        raise ValueError(f"Unsafe videoID: {sample_id!r}")
    return sample_id


def _check_metadata(
    metadata: Mapping[str, Any], feature: torch.Tensor | None = None,
    expected_mode: str | None = None, expected_policy: str | None = None,
) -> None:
    for key, value in WAN_IDENTITY.items():
        if metadata.get(key) != value:
            raise ValueError(f"Wan cache identity mismatch: {key}={metadata.get(key)!r}, expected {value!r}")
    policy = metadata.get("frame_policy", "uniform32")
    if policy not in {"uniform32", "dyn_official"}:
        raise ValueError("Wan cache frame policy mismatch")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError("Wan cache frame policy mismatch")
    indices = metadata.get("source_frame_indices", [])
    if len(indices) != metadata.get("source_num_frames") or not indices:
        raise ValueError("Wan cache source frame indices/count mismatch")
    if policy == "uniform32":
        if metadata.get("source_num_frames") != NUM_FRAMES:
            raise ValueError("Wan cache frame count mismatch")
        if metadata.get("frame_policy_sha256") != FRAME_POLICY_SHA256:
            raise ValueError("Wan cache frame policy mismatch")
    else:
        official = metadata.get("official_frame_indices")
        if not isinstance(official, list) or not official:
            raise ValueError("Wan cache is missing official Dyn-Bench frame indices")
        original_total = metadata.get("original_source_num_frames")
        if not isinstance(original_total, int) or original_total <= max(official):
            raise ValueError("Wan cache official source frame count mismatch")
        if metadata.get("frame_policy_sha256") != dyn_policy_sha256(official):
            raise ValueError("Wan cache frame policy mismatch")
        if indices != apply_video_mode(official, metadata.get("video_mode", "normal")):
            raise ValueError("Wan cache official frame indices mismatch")
    if metadata.get("video_mode", "normal") not in {"normal", "repeat-first", "reversed-video"}:
        raise ValueError("Wan cache video mode mismatch")
    if expected_mode is not None and metadata.get("video_mode") != expected_mode:
        raise ValueError("Wan cache video mode mismatch")
    if not metadata.get("video_sha256"):
        raise ValueError("Wan cache is missing source video identity")
    shape = metadata.get("generator_token_grid_shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("Wan cache token grid is missing")
    if feature is not None:
        expected = tuple(shape) + (1536,)
        if tuple(feature.shape) != expected or feature.dtype != torch.bfloat16:
            raise ValueError(f"Wan cache feature shape or dtype mismatch: {feature.shape}, {feature.dtype}")


class FeatureCache:
    def __init__(
        self, root: str | Path, *, expected_mode: str = "normal",
        expected_policy: str = "uniform32",
    ):
        self.root = Path(root).resolve()
        self.expected_mode = expected_mode
        self.expected_policy = expected_policy
        self.index_path = self.root / "index.jsonl"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Wan cache index is missing: {self.index_path}")
        self.records: dict[str, dict[str, Any]] = {}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = _safe_name(str(record["sample_id"]))
            _check_metadata(record, expected_mode=expected_mode, expected_policy=expected_policy)
            if sample_id in self.records:
                raise ValueError(f"Duplicate Wan cache videoID: {sample_id}")
            self.records[sample_id] = record

    def _feature_path(self, record: Mapping[str, Any]) -> Path:
        relative = Path(str(record["feature_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Wan feature_path must be relative to cache root")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Wan feature_path escapes cache root")
        return path

    def load(
        self, sample_id: str, *, video_path: str | Path | None = None,
        frame_indices: list[int] | None = None,
    ) -> torch.Tensor:
        try:
            record = self.records[_safe_name(sample_id)]
        except KeyError as exc:
            raise FileNotFoundError(f"Wan cache is missing videoID={sample_id!r}") from exc
        if video_path is not None and sha256_file(video_path) != record["video_sha256"]:
            raise ValueError(f"Wan cache source video identity mismatch for {sample_id}")
        if frame_indices is not None and frame_indices != record["source_frame_indices"]:
            raise ValueError(f"Wan cache/Qwen frame indices mismatch for {sample_id}")
        path = self._feature_path(record)
        if sha256_file(path) != record.get("feature_sha256"):
            raise ValueError(f"Wan feature digest mismatch for {sample_id}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("metadata", {}).get("sample_id") != sample_id:
            raise ValueError(f"Wan payload identity mismatch for {sample_id}")
        feature = payload["feature"]
        _check_metadata(
            payload["metadata"], feature, expected_mode=self.expected_mode,
            expected_policy=self.expected_policy,
        )
        return feature

    def check_coverage(self, rows: list[dict[str, Any]], *, exact: bool = True) -> None:
        for row in rows:
            policy = row.get("frame_policy", "uniform32")
            if policy != self.expected_policy:
                raise ValueError(f"Wan cache/frame policy mismatch for {row['sample_id']}")
            if policy == "dyn_official":
                record = self.records.get(str(row["vgm_feature_key"]))
                if record is not None and row.get("official_frame_indices") != record.get("official_frame_indices"):
                    raise ValueError(f"Wan cache official frame indices mismatch for {row['sample_id']}")
                if record is not None and row.get("original_source_num_frames") != record.get("original_source_num_frames"):
                    raise ValueError(f"Wan cache official source frame count mismatch for {row['sample_id']}")
        expected = {str(row["vgm_feature_key"]) for row in rows}
        found = set(self.records)
        missing = expected - found
        extra = found - expected
        if missing or (exact and extra):
            raise ValueError(
                f"Wan cache/sample count mismatch: expected {len(expected)} videos, "
                f"found {len(found)}, missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )
        summary_path = self.root / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary["annotation_rows"]) != len(rows):
                raise ValueError(
                    f"Wan cache sample count mismatch: extracted for "
                    f"{summary['annotation_rows']} rows, received {len(rows)}"
                )
            if int(summary["cached_videos"]) != len(found):
                raise ValueError("Wan cache summary/video count mismatch")


class FeatureCacheWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        if self.index_path.exists():
            raise FileExistsError(f"Cache index already exists: {self.index_path}")

    def write(self, feature: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
        sample_id = _safe_name(str(metadata["sample_id"]))
        _check_metadata(metadata, feature)
        path = self.root / "features" / sample_id / f"{sample_id}.pt"
        if path.exists():
            raise FileExistsError(f"Wan cache already contains {sample_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(metadata)
        record["feature_path"] = path.relative_to(self.root).as_posix()
        torch.save({"feature": feature.cpu(), "metadata": record}, path)
        record["feature_sha256"] = sha256_file(path)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record
