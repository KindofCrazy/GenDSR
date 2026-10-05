"""Convert official DSR Suite annotations to local video SFT records."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .mcq import format_mcq_text

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading official Parquet annotations requires pyarrow") from exc
        rows = pq.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("data", value.get("train"))
    else:
        raise ValueError("Annotations must be .json, .jsonl, or .parquet")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Annotations must contain a list of records")
    return rows


def video_index(root: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in found:
            raise ValueError(f"Ambiguous videoID {path.stem!r}: {found[path.stem]} and {path}")
        found[path.stem] = path.resolve()
    return found


def convert_rows(
    rows: Iterable[dict[str, Any]], videos: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    converted: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    total = 0
    for row_number, row in enumerate(rows):
        total += 1
        video_id = str(row.get("videoID", "")).strip()
        if not video_id:
            raise ValueError(f"Annotation row {row_number} has no videoID")
        options = row.get("options")
        answer = str(row.get("answer", "")).strip().upper()
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError(f"Annotation row {row_number} has invalid answer {answer!r}")
        prompt = format_mcq_text(row.get("question", ""), options)
        path = videos.get(video_id)
        if path is None:
            missing.append({"videoID": video_id, "annotation_row": row_number})
            continue
        converted.append({
            "sample_id": f"{video_id}:{row_number}",
            "type": str(row.get("type", "")),
            "video": str(path),
            "vgm_feature_key": video_id,
            "question": str(row["question"]),
            "options": list(options),
            "answer": answer,
            "frame_policy": "uniform32",
            "conversations": [
                {"from": "human", "value": f"<video>\n{prompt}"},
                {"from": "gpt", "value": answer},
            ],
        })
    counts = {
        "annotation_rows": total,
        "converted_rows": len(converted),
        "missing_rows": len(missing),
        "unique_videos": len({row["vgm_feature_key"] for row in converted}),
    }
    if total != len(converted) + len(missing):
        raise RuntimeError("Annotation sample accounting mismatch")
    return converted, missing, counts


def convert_dyn_rows(
    rows: Iterable[dict[str, Any]], video_root: str | Path,
    sampling_archive: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Prepare Dyn-Bench QA Parquet against an extracted videos/<dataset>/ tree."""
    root = Path(video_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    sampling: dict[str, tuple[list[int], int]] = {}
    with zipfile.ZipFile(sampling_archive) as archive:
        for member in archive.namelist():
            if not member.endswith("_frame_sampling.json"):
                continue
            parts = Path(member).parts
            if len(parts) < 3:
                raise ValueError(f"Invalid Dyn-Bench sampling archive member: {member}")
            entry = json.loads(archive.read(member))
            dataset_name = parts[-2]
            video_name = str(entry.get("video_id") or Path(member).stem.removesuffix("_frame_sampling"))
            if video_name.startswith(dataset_name + "_"):
                video_name = video_name[len(dataset_name) + 1:]
            frame_indices = [int(value) for value in entry.get("frame_indices", [])]
            source_frames = int(entry.get("total_frames", 0))
            if not frame_indices or source_frames <= max(frame_indices) or min(frame_indices) < 0:
                raise ValueError(f"Invalid official frame indices in {member}")
            key = f"{dataset_name}/{video_name}"
            if key in sampling:
                raise ValueError(f"Duplicate Dyn-Bench sampling record: {key}")
            sampling[key] = (frame_indices, source_frames)
    if not sampling:
        raise ValueError("Dyn-Bench sampling archive has no frame records")
    converted: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    total = 0
    for row_number, row in enumerate(rows):
        total += 1
        dataset = str(row.get("dataset", "")).strip()
        name = str(row.get("video_name") or row.get("video") or "").strip()
        if not dataset or not name:
            raise ValueError(f"Dyn-Bench row {row_number} is missing dataset or video")
        if name.startswith(dataset + "/"):
            name = name[len(dataset) + 1:]
        if Path(name).name != name:
            raise ValueError(f"Dyn-Bench row {row_number} has unsafe video name")
        stem = Path(name).stem
        sample_key = f"{dataset}/{stem}"
        if sample_key not in sampling:
            raise ValueError(f"Dyn-Bench row {row_number} has no official frame indices: {sample_key}")
        official_indices, source_frames = sampling[sample_key]
        options = row.get("options")
        if isinstance(options, str):
            options = ast.literal_eval(options)
        answer = str(row.get("answer", "")).strip().upper()
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError(f"Dyn-Bench row {row_number} has invalid answer {answer!r}")
        prompt = format_mcq_text(row.get("question", ""), options)
        names = (
            [name] if Path(name).suffix.lower() in VIDEO_SUFFIXES
            else [f"{name}{suffix}" for suffix in sorted(VIDEO_SUFFIXES)]
        )
        candidates = [root / dataset / candidate for candidate in names]
        paths = [path.resolve() for path in candidates if path.is_file()]
        if len(paths) > 1:
            raise ValueError(f"Ambiguous Dyn-Bench video {dataset}/{name}")
        if not paths:
            missing.append({"videoID": f"{dataset}/{name}", "annotation_row": row_number})
            continue
        feature_key = "dyn_" + hashlib.sha256(
            f"{dataset}/{stem}".encode("utf-8")
        ).hexdigest()[:24]
        converted.append({
            "sample_id": f"{dataset}/{name}:{row_number}",
            "type": str(row.get("task", row.get("type", ""))),
            "video": str(paths[0]),
            "vgm_feature_key": feature_key,
            "question": str(row["question"]),
            "options": list(options),
            "answer": answer,
            "frame_policy": "dyn_official",
            "official_frame_indices": official_indices,
            "original_source_num_frames": source_frames,
            "conversations": [
                {"from": "human", "value": f"<video>\n{prompt}"},
                {"from": "gpt", "value": answer},
            ],
        })
    counts = {
        "annotation_rows": total, "converted_rows": len(converted),
        "missing_rows": len(missing),
        "unique_videos": len({row["vgm_feature_key"] for row in converted}),
    }
    return converted, missing, counts


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prepared(path: str | Path, *, expected_rows: int | None = None) -> list[dict[str, Any]]:
    rows = read_rows(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Sample count mismatch: expected {expected_rows}, found {len(rows)}")
    stats_path = Path(path).with_suffix(".stats.json")
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if int(stats["converted_rows"]) != len(rows):
            raise ValueError(
                f"Prepared sample count mismatch: stats say {stats['converted_rows']}, "
                f"JSONL has {len(rows)}"
            )
    for index, row in enumerate(rows):
        for field in ("video", "vgm_feature_key", "conversations", "answer"):
            if field not in row:
                raise ValueError(f"Prepared row {index} is missing {field}")
        policy = row.get("frame_policy", "uniform32")
        if policy not in {"uniform32", "dyn_official"}:
            raise ValueError(f"Prepared row {index} has unknown frame policy {policy!r}")
        if policy == "dyn_official":
            official = row.get("official_frame_indices")
            total = row.get("original_source_num_frames")
            if (
                not isinstance(official, list) or not official
                or any(type(value) is not int or value < 0 for value in official)
                or not isinstance(total, int) or total <= max(official)
            ):
                raise ValueError(f"Prepared row {index} has invalid official frame indices")
        if not Path(row["video"]).is_file():
            raise FileNotFoundError(f"Prepared row {index} video is missing: {row['video']}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--dataset", choices=["dsr", "dyn"], default="dsr")
    parser.add_argument("--sampling-archive", type=Path, help="Dyn-Bench multi_json.zip with official frame indices")
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--missing-output", required=True, type=Path)
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args(argv)
    annotations = read_rows(args.annotations)
    if args.dataset == "dyn" and args.sampling_archive is None:
        parser.error("--dataset dyn requires --sampling-archive")
    rows, missing, counts = (
        convert_rows(annotations, video_index(args.video_root))
        if args.dataset == "dsr"
        else convert_dyn_rows(annotations, args.video_root, args.sampling_archive)
    )
    write_jsonl(args.output, rows)
    write_jsonl(args.missing_output, missing)
    args.output.with_suffix(".stats.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(counts, sort_keys=True))
    if args.strict_missing and missing:
        parser.error(f"{len(missing)} annotation rows have no local video; see {args.missing_output}")


if __name__ == "__main__":
    main()
