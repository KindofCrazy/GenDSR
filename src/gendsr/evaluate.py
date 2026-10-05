"""Evaluate a GenDSR checkpoint on prepared DSR-Bench MCQs."""

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
    *, data_path: Path, checkpoint: Path,
    config_path: Path, output: Path, expected_samples: int,
    cache_root: Path,
) -> dict:
    load_config(config_path)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    rows = load_prepared(data_path, expected_rows=expected_samples)
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GenDSR checkpoint directory does not exist: {checkpoint}")
    cache = FeatureCache(cache_root)
    cache.check_coverage(rows)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL-8B evaluation requires CUDA")
    model, processor = load_qwen(checkpoint)
    fusion = load_fusion_strict(checkpoint, config_path)
    adapter = QwenFusionAdapter(model, fusion)
    model.cuda().eval()
    predictions = []
    for row in rows:
        inputs, features = prepare_model_inputs(
            row, processor, cache, training=False,
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
    summary["benchmark"] = "dsr"
    output.mkdir(parents=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-samples", required=True, type=int)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(
        data_path=args.data,
        checkpoint=args.checkpoint, config_path=args.config, output=args.output,
        expected_samples=args.expected_samples, cache_root=args.cache_root,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
