# Reproducing GenDSR

## 1. Obtain inputs and set paths

Get the [DSR Suite annotations](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data)
and associated videos following the
[publisher's data instructions](https://github.com/TencentARC/DSR_Suite/blob/main/data/README.md).
Also obtain [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), and the
[official Wan2.1 source](https://github.com/Wan-Video/Wan2.1). The code does
not download weights or videos at runtime.

```bash
export CONFIG=configs/gendsr.yaml
export WORK=/path/to/work
export TRAIN_ANN=/path/to/DSR_Suite-Data/train_qa_pairs.parquet
export BENCH_ANN=/path/to/DSR_Suite-Data/benchmark.parquet
export TRAIN_VIDEO_ROOT=/path/to/train/videos
export BENCH_VIDEO_ROOT=/path/to/benchmark/videos
export QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
export WAN_CHECKPOINT=/path/to/Wan2.1-T2V-1.3B
export WAN_REPO=/path/to/Wan2.1
```

## 2. Prepare annotations in the SFT environment

```bash
python3.11 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install -e ".[sft]"
python -m pip install flash-attn --no-build-isolation
python -m pip install deepspeed

gendsr-prepare \
  --annotations "$TRAIN_ANN" --video-root "$TRAIN_VIDEO_ROOT" \
  --output "$WORK/train.jsonl" --missing-output "$WORK/train_missing.jsonl"

gendsr-prepare \
  --annotations "$BENCH_ANN" --video-root "$BENCH_VIDEO_ROOT" \
  --output "$WORK/benchmark.jsonl" --missing-output "$WORK/benchmark_missing.jsonl"

TRAIN_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["converted_rows"])' "$WORK/train.stats.json")"
BENCH_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["converted_rows"])' "$WORK/benchmark.stats.json")"
```

The converter matches each annotation's `videoID` to the exact local filename
stem. It writes one JSONL row per available QA, a missing-media JSONL, and a
count summary. It accepts the official JSON annotation file as well as Parquet.
Inspect `train_missing.jsonl`, `benchmark_missing.jsonl`, and the two
`*.stats.json` files. Use `--strict-missing` if any missing video should fail
preparation.

## 3. Extract Wan features in the Wan environment

```bash
deactivate
python3.11 -m venv .venv-wan
source .venv-wan/bin/activate
python -m pip install -r "$WAN_REPO/requirements.txt"
python -m pip install -r requirements/wan.txt
python -m pip install -e . --no-deps

gendsr-extract \
  --data "$WORK/train.jsonl" --cache-root "$WORK/train-wan-cache" \
  --wan-checkpoint "$WAN_CHECKPOINT" --wan-repo "$WAN_REPO" \
  --config "$CONFIG" --expected-samples "$TRAIN_COUNT"

gendsr-extract \
  --data "$WORK/benchmark.jsonl" --cache-root "$WORK/benchmark-wan-cache" \
  --wan-checkpoint "$WAN_CHECKPOINT" --wan-repo "$WAN_REPO" \
  --config "$CONFIG" --expected-samples "$BENCH_COUNT"
```

Each unique video is processed once. The extractor samples 32 endpoint-inclusive
frames, resizes and center crops to Wan's native landscape or portrait size,
encodes the clip with the Wan VAE, and captures the zero-indexed DiT block 20 at
the scheduler step closest to timestep 300. It uses the empty-prompt condition
from Wan's T5 encoder and saves BF16 `[T,H,W,1536]` tensors. Each cache contains
`index.jsonl`, video hashes, frame indices, and metadata checked by training and
evaluation. Plan for substantial cache storage.

## 4. Train in the SFT environment

```bash
deactivate
source .venv-sft/bin/activate

accelerate launch --multi_gpu --num_processes 8 \
  --use_deepspeed --zero_stage 2 --gradient_accumulation_steps 4 \
  -m gendsr.train \
  --data "$WORK/train.jsonl" --cache-root "$WORK/train-wan-cache" \
  --qwen-model "$QWEN_MODEL" --output "$WORK/gendsr-checkpoint" \
  --config "$CONFIG" --expected-samples "$TRAIN_COUNT"
```

The output contains Qwen weights, the processor, and a separate fusion state.
Evaluation checks the fusion state and configuration digest strictly.
`flash-attn` and DeepSpeed must match the local PyTorch and CUDA versions.

## 5. Evaluate on DSR-Bench

```bash
gendsr-eval \
  --data "$WORK/benchmark.jsonl" --cache-root "$WORK/benchmark-wan-cache" \
  --checkpoint "$WORK/gendsr-checkpoint" --config "$CONFIG" \
  --expected-samples "$BENCH_COUNT" --output "$WORK/eval"
```

Evaluation uses the standard video and Wan feature cache. It writes
`predictions.jsonl` and `summary.json` with sample-level accuracy. A missing
feature, changed video, frame-index mismatch, or wrong sample count stops the
run with an error. The evaluation output directory must not already exist.

## CPU checks

```bash
python -m pip install -e ".[test]"
bash -n scripts/*.sh
bash scripts/run_checks.sh
```

The CPU tests cover fusion gradients and video isolation, strict checkpoint
loading, annotation conversion and missing media, cache identity, evaluation
counts, and CLI preflight. Source-method forward and gradient parity was also
checked locally with synthetic inputs. End-to-end model validation requires
separately obtained weights, media, and a GPU environment.
