# GenDSR

GenDSR adds clip level Wan2.1 features to Qwen3-VL visual tokens for dynamic spatial reasoning. It uses two feature branches: the raw Wan grid and its first temporal difference. Both are aligned by temporal interpolation and spatial pooling, projected independently, and weighted by gates conditioned on Qwen visual tokens and the text prompt. The weighted sum is added to the native Qwen visual representation.

This repository contains code for data preparation, Wan feature extraction, supervised fine tuning, and multiple choice evaluation. It does not contain model weights, videos, cached features, or historical scores. The paper and results will be documented separately.

## Obtain the inputs

Acquire these independently:

- [DSR Suite annotations](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data), including `train_qa_pairs.parquet` for training and `benchmark.parquet` for DSR-Bench evaluation.
- The corresponding videos. The [dataset publisher's instructions](https://github.com/TencentARC/DSR_Suite/blob/main/data/README.md) describe their sources and availability. Some videos may be unavailable; `gendsr-prepare` records every missing annotation row. A new public-data run cannot be assumed to reconstruct the original private training sample set.
- [Qwen3-VL-8B-Instruct weights and processor](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- [Wan2.1-T2V-1.3B weights](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and the [official Wan2.1 source](https://github.com/Wan-Video/Wan2.1).

All paths below are local paths supplied by the user. The repository never downloads weights or videos at runtime.

## Environments

Use Python 3.11 and a CUDA machine for extraction and training. The Wan extractor and Qwen training stack use separate environments because their Transformers versions differ.

```bash
python3.11 -m venv .venv-wan
source .venv-wan/bin/activate
python -m pip install -r /path/to/Wan2.1/requirements.txt
python -m pip install -r requirements/wan.txt
python -m pip install -e . --no-deps
deactivate

python3.11 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install -e ".[sft]"
python -m pip install flash-attn --no-build-isolation
python -m pip install deepspeed
```

The `flash-attn` and DeepSpeed installs must match the local CUDA and PyTorch versions. For CPU unit tests, `python -m pip install -e ".[test]"` is enough.

## 1. Prepare annotations and local videos

```bash
export CONFIG=configs/gendsr.yaml
export ANNOTATIONS=/path/to/DSR_Suite-Data/train_qa_pairs.parquet
export VIDEO_ROOT=/path/to/dsr/videos
export DATA=/path/to/work/train.jsonl

gendsr-prepare \
  --annotations "$ANNOTATIONS" \
  --video-root "$VIDEO_ROOT" \
  --output "$DATA" \
  --missing-output /path/to/work/missing_videos.jsonl
```

The converter matches `videoID` to the exact local filename stem. It writes one SFT row per available QA, a missing-media JSONL file, and a `train.stats.json` count summary. Review the counts before extraction. The command also accepts the official JSON annotation file.

For Dyn-Bench, provide its `qa_tasks.parquet`, `multi_json.zip` sampling archive, and extracted `videos/<dataset>/<video>.mp4` directory:

```bash
gendsr-prepare --dataset dyn \
  --annotations /path/to/Dyn-Bench/qa_tasks.parquet \
  --sampling-archive /path/to/Dyn-Bench/multi_json.zip \
  --video-root /path/to/Dyn-Bench/videos \
  --output /path/to/work/dyn.jsonl \
  --missing-output /path/to/work/dyn_missing.jsonl
```

The Dyn-Bench rows retain each video's official frame indices. Extraction and Qwen evaluation read the same indices and reject mismatched caches.

## 2. Extract Wan features

Activate the Wan environment and set local model paths:

```bash
export WAN_CHECKPOINT=/path/to/Wan2.1-T2V-1.3B
export WAN_REPO=/path/to/Wan2.1
export CACHE=/path/to/work/wan-cache

gendsr-extract \
  --data "$DATA" \
  --cache-root "$CACHE" \
  --wan-checkpoint "$WAN_CHECKPOINT" \
  --wan-repo "$WAN_REPO" \
  --config "$CONFIG"
```

For DSR Suite, the extractor samples 32 endpoint inclusive frames. For prepared Dyn-Bench data, it uses the official indices in the JSONL. It then resizes and center crops to Wan's native landscape or portrait size, runs Wan2.1-T2V-1.3B in BF16, and captures DiT block 20 at the scheduler step closest to timestep 300. It computes the empty prompt condition using Wan's own T5 encoder and checkpoint. The cache has a portable `index.jsonl`, BF16 `[T,H,W,1536]` tensors, video hashes, exact source frame indices, and metadata checked by training and evaluation.

Each video is extracted once even when it has multiple questions. Plan for substantial cache storage.

## 3. Train

Activate the SFT environment. The fixed recipe uses 32 frames, a 230400 pixel per frame budget, global batch 32, one epoch, seed 42, and learning rates `2e-7` for the language model, `1e-6` for the Qwen visual merger, and `1e-5` for fusion. The vision tower stays frozen. Each device processes one sample per step; gradient accumulation is `32 / world_size`.

```bash
export QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
export OUTPUT=/path/to/work/gendsr-checkpoint
export EXPECTED_SAMPLES="$(python -c 'import json; print(json.load(open("/path/to/work/train.stats.json"))["converted_rows"])')"

accelerate launch --multi_gpu --num_processes 8 \
  --use_deepspeed --zero_stage 2 --gradient_accumulation_steps 4 \
  -m gendsr.train \
  --data "$DATA" --cache-root "$CACHE" \
  --qwen-model "$QWEN_MODEL" --output "$OUTPUT" \
  --config "$CONFIG" --expected-samples "$EXPECTED_SAMPLES"
```

The output contains Qwen weights, the processor, and a separate fusion state. Evaluation checks the fusion state and configuration digest strictly.

## 4. Evaluate

Prepare DSR-Bench `benchmark.parquet` with `gendsr-prepare` in the same way as training, using a separate output JSONL. Evaluate one condition per command:

```bash
gendsr-eval \
  --benchmark dsr --condition vgm \
  --data /path/to/work/benchmark.jsonl \
  --checkpoint "$OUTPUT" --cache-root /path/to/work/benchmark-wan-cache \
  --config "$CONFIG" \
  --expected-samples "$(python -c 'import json; print(json.load(open("/path/to/work/benchmark.stats.json"))["converted_rows"])')" \
  --output /path/to/work/eval-vgm
```

The five DSR-Bench conditions are `vgm`, `no-vgm`, `text-only`, `repeat-first`, and `reversed-video`. The last two require feature caches created with `gendsr-extract --video-mode repeat-first` or `--video-mode reversed-video` and evaluated with the matching `--condition`. `no-vgm` and `text-only` do not require a cache. A mode mismatch, missing feature, changed video, frame index mismatch, or wrong sample count stops evaluation with an error.

Use `--benchmark dyn` with a prepared Dyn-Bench JSONL for the Dyn-Bench entry. Each run writes `predictions.jsonl` and `summary.json` with counts computed from the provided samples.

## Verification

```bash
python -m pip install -e ".[test]"
bash -n scripts/*.sh
bash scripts/run_checks.sh
```

The CPU tests cover fusion gradients and video isolation, strict checkpoint loading, annotation conversion, missing media, cache identity and format, evaluation counts, and CLI preflight. Source-method forward and gradient parity was checked locally with synthetic inputs. Full Qwen/Wan training and benchmark reproduction require separately obtained weights, media, and a CUDA environment.

## License

Apache-2.0. See [LICENSE](LICENSE).
