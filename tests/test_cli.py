import pytest
from pathlib import Path

from gendsr.data import write_jsonl
from gendsr.evaluate import evaluate
from gendsr.train import train
from gendsr.wan import extract

CONFIG = Path(__file__).parents[1] / "configs" / "gendsr.yaml"


def test_training_preflight_checks_data_before_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(
            data_path=tmp_path / "missing.jsonl", cache_root=tmp_path / "cache",
            qwen_model=tmp_path / "model", output=tmp_path / "out",
            config_path=CONFIG, expected_samples=1,
        )


def test_evaluation_rejects_wrong_frame_policy_before_model(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    data = tmp_path / "benchmark.jsonl"
    write_jsonl(data, [{
        "sample_id": "qa", "video": str(video), "vgm_feature_key": "clip",
        "conversations": [], "answer": "A", "frame_policy": "other",
    }])
    with pytest.raises(ValueError, match="uniform32"):
        evaluate(
            data_path=data,
            checkpoint=tmp_path / "checkpoint", config_path=CONFIG,
            output=tmp_path / "out", expected_samples=1, cache_root=None,
        )


def test_extraction_preflight_checks_wan_paths_before_cuda(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    rows = [{"vgm_feature_key": "clip", "video": str(video), "frame_policy": "uniform32"}]
    with pytest.raises(FileNotFoundError, match="Wan checkpoint"):
        extract(
            rows=rows, cache_root=tmp_path / "cache",
            checkpoint_dir=tmp_path / "missing-checkpoint",
            wan_repo=tmp_path / "missing-source", config_path=CONFIG,
        )
