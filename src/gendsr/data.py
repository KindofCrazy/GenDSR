"""Convert official DSR Suite annotations to local video SFT records."""

from __future__ import annotations

import argparse
import json
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
        if row.get("frame_policy", "uniform32") != "uniform32":
            raise ValueError(f"Prepared row {index} requires the uniform32 frame policy")
        if not Path(row["video"]).is_file():
            raise FileNotFoundError(f"Prepared row {index} video is missing: {row['video']}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--missing-output", required=True, type=Path)
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args(argv)
    annotations = read_rows(args.annotations)
    rows, missing, counts = convert_rows(annotations, video_index(args.video_root))
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
