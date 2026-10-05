import json
import zipfile

import pytest

from gendsr.data import convert_dyn_rows, convert_rows, load_prepared, video_index, write_jsonl


def _row(video_id, answer="B"):
    return {
        "videoID": video_id,
        "type": "rel_dir",
        "question": "What changed?",
        "options": ["A. Left", "B. Right", "C. Up", "D. Down"],
        "answer": answer,
    }


def test_official_annotation_conversion_and_missing_video(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").write_bytes(b"video")
    rows, missing, counts = convert_rows([_row("clip"), _row("lost")], video_index(videos))
    assert counts == {
        "annotation_rows": 2, "converted_rows": 1,
        "missing_rows": 1, "unique_videos": 1,
    }
    assert missing == [{"videoID": "lost", "annotation_row": 1}]
    assert rows[0]["vgm_feature_key"] == "clip"
    assert "A Left" in rows[0]["conversations"][0]["value"]
    assert rows[0]["conversations"][1]["value"] == "B"
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    assert load_prepared(path, expected_rows=1) == rows
    path.with_suffix(".stats.json").write_text('{"converted_rows": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared sample count mismatch"):
        load_prepared(path)
    with pytest.raises(ValueError, match="Sample count mismatch"):
        load_prepared(path, expected_rows=2)


def test_reject_ambiguous_and_bad_answers(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"a")
    (tmp_path / "clip.mkv").write_bytes(b"b")
    with pytest.raises(ValueError, match="Ambiguous"):
        video_index(tmp_path)
    with pytest.raises(ValueError, match="invalid answer"):
        convert_rows([_row("clip", answer="E")], {})


def test_dyn_bench_entry_and_missing_video(tmp_path):
    (tmp_path / "set_a").mkdir()
    (tmp_path / "set_a" / "clip.mp4").write_bytes(b"video")
    dyn = [
        {"dataset": "set_a", "video": "clip", "task": "motion", "question": "Where?",
         "options": ["A. L", "B. R", "C. U", "D. D"], "answer": "A"},
        {"dataset": "set_a", "video": "missing", "task": "motion", "question": "Where?",
         "options": ["A. L", "B. R", "C. U", "D. D"], "answer": "B"},
    ]
    archive = tmp_path / "multi_json.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in ("clip", "missing"):
            handle.writestr(
                f"root/set_a/set_a_{name}_frame_sampling.json",
                json.dumps({
                    "video_id": f"set_a_{name}",
                    "frame_indices": [0, 2, 4, 6, 8],
                    "total_frames": 10,
                }),
            )
    rows, missing, counts = convert_dyn_rows(dyn, tmp_path, archive)
    assert counts["converted_rows"] == 1
    assert counts["missing_rows"] == 1
    assert rows[0]["vgm_feature_key"].startswith("dyn_")
    assert rows[0]["official_frame_indices"] == [0, 2, 4, 6, 8]
    assert rows[0]["frame_policy"] == "dyn_official"
    assert missing[0]["videoID"] == "set_a/missing"
