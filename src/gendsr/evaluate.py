"""Evaluate a GenDSR checkpoint on prepared DSR-Bench or Dyn-Bench MCQs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .cache import FeatureCache
from .checkpoint import load_fusion_strict
from .config import load_config
from .data import load_prepared, write_jsonl
from .qwen import QwenFusionAdapter, load_qwen, prepare_model_inputs

CONDITIONS = ("vgm", "no-vgm", "text-only", "repeat-first", "reversed-video")


def parse_letter(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def summarize(predictions: list[dict], *, expected_samples: int) -> dict[str, int | float]:
    if len(predictions) != expected_samples:
        raise ValueError(f"Evaluation sample count mismatch: expected {expected_samples}, got {len(predictions)}")
    if any("answer" not in row or "prediction" not in row for row in predictions):
        raise ValueError("Evaluation predictions are missing answer or prediction")
    correct = sum(row["prediction"] == row["answer"] for row in predictions)
    invalid = sum(row["prediction"] is None for row in predictions)
    return {
        "samples": len(predictions), "correct": correct, "invalid": invalid,
        "accuracy": correct / len(predictions) if predictions else 0.0,
    }


def evaluate(
    *, benchmark: str, condition: str, data_path: Path, checkpoint: Path,
    config_path: Path, output: Path, expected_samples: int,
    cache_root: Path | None,
) -> dict:
    load_config(config_path)
    if benchmark not in {"dsr", "dyn"}:
        raise ValueError("Benchmark must be dsr or dyn")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    rows = load_prepared(data_path, expected_rows=expected_samples)
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    expected_policy = "dyn_official" if benchmark == "dyn" else "uniform32"
    if any(row.get("frame_policy", "uniform32") != expected_policy for row in rows):
        raise ValueError(f"{benchmark} evaluation requires {expected_policy} frame policy")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GenDSR checkpoint directory does not exist: {checkpoint}")
    video_mode = condition if condition in {"repeat-first", "reversed-video"} else "normal"
    needs_cache = condition in {"vgm", "repeat-first", "reversed-video"}
    if needs_cache and cache_root is None:
        raise ValueError(f"Condition {condition} requires --cache-root")
    cache = (
        FeatureCache(cache_root, expected_mode=video_mode, expected_policy=expected_policy)
        if needs_cache else None
    )
    if cache is not None:
        cache.check_coverage(rows)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL-8B evaluation requires CUDA")
    model, processor = load_qwen(checkpoint, video_mode=video_mode)
    fusion = load_fusion_strict(checkpoint, config_path)
    adapter = QwenFusionAdapter(model, fusion)
    model.cuda().eval()
    predictions = []
    for row in rows:
        inputs, features = prepare_model_inputs(
            row, processor, cache, training=False,
            text_only=condition == "text-only",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode(), adapter.use(features, inputs):
            generated = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
            )
        continuation = generated[:, inputs["input_ids"].shape[-1]:]
        answer_text = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        predictions.append({
            "sample_id": row["sample_id"],
            "videoID": row["vgm_feature_key"],
            "type": row.get("type", ""),
            "answer": row["answer"],
            "prediction": parse_letter(answer_text),
            "raw_output": answer_text,
        })
    summary = summarize(predictions, expected_samples=expected_samples)
    summary.update({"benchmark": benchmark, "condition": condition})
    output.mkdir(parents=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["dsr", "dyn"], required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-samples", required=True, type=int)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(
        benchmark=args.benchmark, condition=args.condition, data_path=args.data,
        checkpoint=args.checkpoint, config_path=args.config, output=args.output,
        expected_samples=args.expected_samples, cache_root=args.cache_root,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
