"""Train Qwen3-VL with the fixed GenDSR Wan visual residual."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .cache import FeatureCache
from .checkpoint import save_checkpoint
from .config import load_config
from .data import load_prepared
from .fusion import GenDSRFusion
from .qwen import QwenFusionAdapter, load_qwen, prepare_model_inputs, qwen_hidden_size, resolve_visual


class PreparedDataset(Dataset):
    def __init__(self, rows: list[dict], processor, cache: FeatureCache, max_length: int):
        self.rows = rows
        self.processor = processor
        self.cache = cache
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        inputs, features = prepare_model_inputs(
            self.rows[index], self.processor, self.cache, training=True,
        )
        length = int(inputs["input_ids"].shape[-1])
        if length > self.max_length:
            raise ValueError(
                f"Prepared sample {self.rows[index]['sample_id']} has {length} tokens; "
                f"model_max_length={self.max_length}"
            )
        return inputs, features


def _one(batch):
    if len(batch) != 1:
        raise ValueError("GenDSR uses one sample per device step")
    return batch[0]


def optimizer_groups(model, fusion: GenDSRFusion, config: dict):
    from transformers.trainer_pt_utils import get_parameter_names

    recipe = config["training"]
    projector = resolve_visual(model).merger
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    fusion_ids = {id(parameter) for parameter in fusion.parameters()}
    groups = {"base": [], "projector": [], "fusion": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in fusion_ids:
            groups["fusion"].append((name, parameter))
        elif id(parameter) in projector_ids:
            groups["projector"].append((name, parameter))
        else:
            groups["base"].append((name, parameter))
    if any(not values for values in groups.values()):
        raise RuntimeError("Base, projector, and fusion optimizer groups must all be non-empty")
    rates = {
        "base": recipe["base_learning_rate"],
        "projector": recipe["projector_learning_rate"],
        "fusion": recipe["fusion_learning_rate"],
    }
    decay_names = set(get_parameter_names(
        model, [torch.nn.LayerNorm],
        [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"],
    ))
    result = []
    for owner in ("base", "projector", "fusion"):
        for use_decay in (True, False):
            parameters = [
                parameter for name, parameter in groups[owner]
                if (name in decay_names and not name.endswith("bias")) == use_decay
            ]
            if parameters:
                result.append({
                    "params": parameters,
                    "lr": float(rates[owner]),
                    "weight_decay": float(recipe["weight_decay"]) if use_decay else 0.0,
                    "name": f"{owner}_{'decay' if use_decay else 'no_decay'}",
                })
    return result


def train(
    *, data_path: Path, cache_root: Path, qwen_model: Path,
    output: Path, config_path: Path, expected_samples: int | None,
) -> None:
    config = load_config(config_path)
    recipe = config["training"]
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    rows = load_prepared(data_path, expected_rows=expected_samples)
    if not rows:
        raise ValueError("Prepared dataset is empty")
    if any(row.get("frame_policy", "uniform32") != "uniform32" for row in rows):
        raise ValueError("The training recipe requires the uniform32 frame policy")
    cache = FeatureCache(cache_root)
    cache.check_coverage(rows)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup

    set_seed(int(recipe["seed"]))
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    global_batch = int(recipe["global_batch_size"])
    if global_batch % world_size:
        raise ValueError(f"Global batch {global_batch} must be divisible by world size {world_size}")
    accumulation = global_batch // world_size
    accelerator = Accelerator(gradient_accumulation_steps=accumulation, mixed_precision="bf16")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Qwen3-VL-8B training requires CUDA")
    model, processor = load_qwen(qwen_model)
    visual = resolve_visual(model)
    visual.requires_grad_(False)
    visual.merger.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    fusion = GenDSRFusion(qwen_hidden_size(model))
    adapter = QwenFusionAdapter(model, fusion)
    dataset = PreparedDataset(rows, processor, cache, int(recipe["model_max_length"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_one, num_workers=0)
    optimizer = torch.optim.AdamW(optimizer_groups(model, fusion, config))
    update_steps = math.ceil(len(rows) / global_batch) * int(recipe["epochs"])
    warmup_steps = math.ceil(update_steps * float(recipe["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, update_steps)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()
    for epoch in range(int(recipe["epochs"])):
        for inputs, features in loader:
            with accelerator.accumulate(model):
                with adapter.use(features, inputs):
                    loss = model(**inputs).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(recipe["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(unwrapped, processor, fusion, output, config_path)
        (output / "training_summary.json").write_text(
            json.dumps({
                "input_samples": len(rows), "unique_videos": len(cache.records),
                "epochs": recipe["epochs"], "global_batch_size": global_batch,
                "world_size": accelerator.num_processes,
            }, indent=2) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args(argv)
    train(
        data_path=args.data, cache_root=args.cache_root, qwen_model=args.qwen_model,
        output=args.output, config_path=args.config, expected_samples=args.expected_samples,
    )


if __name__ == "__main__":
    main()
