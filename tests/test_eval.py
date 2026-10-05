from pathlib import Path

import pytest

from gendsr.evaluate import evaluate, parse_letter, summarize
from gendsr.video import apply_video_mode, uniform_frame_indices


def test_eval_count_and_answer_accounting():
    rows = [
        {"answer": "A", "prediction": "A"},
        {"answer": "B", "prediction": None},
    ]
    assert summarize(rows, expected_samples=2) == {
        "samples": 2, "correct": 1, "invalid": 1, "accuracy": 0.5,
    }
    with pytest.raises(ValueError, match="sample count mismatch"):
        summarize(rows, expected_samples=3)
    assert parse_letter("Answer: C.") == "C"
    assert parse_letter("unknown") is None


def test_cli_preflight_rejects_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(
            benchmark="dsr", condition="vgm", data_path=tmp_path / "missing.jsonl",
            checkpoint=tmp_path / "checkpoint", config_path=Path(__file__).parents[1] / "configs" / "gendsr.yaml",
            output=tmp_path / "out", expected_samples=1, cache_root=tmp_path / "cache",
        )


def test_video_modes_keep_frame_count():
    normal = uniform_frame_indices(10)
    assert len(normal) == 32
    assert uniform_frame_indices(10, "repeat-first") == [0] * 32
    assert uniform_frame_indices(10, "reversed-video") == list(reversed(normal))
    assert apply_video_mode([0, 2, 4], "repeat-first") == [0, 0, 0]
