"""Narrow Qwen3-VL integration for the GenDSR visual residual."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cache import FeatureCache
from .fusion import GenDSRFusion
from .labels import labels_from_assistant_spans
from .mcq import build_sft_aligned_mcq_prompt
from .video import configure_qwen_video_processor


def resolve_visual(model: nn.Module) -> nn.Module:
    core = getattr(model, "model", model)
    visual = getattr(core, "visual", None)
    if visual is None:
        raise RuntimeError("Qwen3-VL visual module was not found")
    return visual


def resolve_video_feature_owner(model: nn.Module) -> nn.Module:
    core = getattr(model, "model", None)
    if core is None or not callable(getattr(core, "get_video_features", None)):
        raise RuntimeError("Qwen3-VL core must expose get_video_features")
    return core


def qwen_hidden_size(model: nn.Module) -> int:
    return int(model.config.text_config.hidden_size)


def prepare_model_inputs(
    row: dict[str, Any], processor, cache: FeatureCache, *, training: bool,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    prompt = row["conversations"][0]["value"].replace("<video>", "", 1).strip()
    if not training:
        prompt = build_sft_aligned_mcq_prompt(prompt)
    content = [
        {"type": "video", "video": str(row["video"])},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if training:
        messages.append({
            "role": "assistant", "content": [{"type": "text", "text": row["answer"]}],
        })
    if hasattr(processor, "video_processor"):
        video = processor.video_processor
        video._gendsr_video_metadata = []
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=not training,
        return_dict=True, return_tensors="pt",
    )
    inputs = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = torch.as_tensor(
            processor.create_mm_token_type_ids(inputs["input_ids"]), dtype=torch.long,
        )
    if training:
        labels = labels_from_assistant_spans(inputs["input_ids"], processor.tokenizer, -100)
        if not bool(labels.ne(-100).any()):
            raise ValueError(f"No assistant answer tokens found for {row['sample_id']}")
        inputs["labels"] = labels
    metadata = getattr(processor.video_processor, "_gendsr_video_metadata", [])
    if len(metadata) != 1:
        raise ValueError("Qwen processor must return metadata for exactly one video")
    item = metadata[0]
    indices = item.get("frames_indices") if isinstance(item, dict) else getattr(item, "frames_indices", None)
    if indices is None:
        raise ValueError("Qwen processor did not return source frame indices")
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    features = [cache.load(
        str(row["vgm_feature_key"]), video_path=row["video"],
        frame_indices=[int(value) for value in indices],
    )]
    return inputs, features


class QwenFusionAdapter:
    """Patch only Qwen's video feature return, preserving its native forward."""

    def __init__(self, model: nn.Module, fusion: GenDSRFusion):
        self.model = model
        self.fusion = fusion
        self.owner = resolve_video_feature_owner(model)
        self.original = self.owner.get_video_features
        self.features: list[torch.Tensor] | None = None
        self.task_context: torch.Tensor | None = None
        self.consumed = False
        reference = next(resolve_visual(model).parameters())
        fusion.to(device=reference.device, dtype=reference.dtype)
        model.vgm_add_residual = fusion
        self.owner.get_video_features = self._video_features

    @staticmethod
    def _task_context(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(input_ids)).bool()
        if "labels" in inputs:
            mask &= inputs["labels"].eq(-100)
        if "mm_token_type_ids" in inputs:
            mask &= inputs["mm_token_type_ids"].eq(0)
        if not bool(mask.any()):
            raise ValueError("No text prompt tokens are available for task conditioning")
        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids).float()
            mask = mask.to(device=embeds.device)
            return (
                (embeds * mask.unsqueeze(-1)).sum(dim=1)
                / mask.sum(dim=1, keepdim=True).clamp_min(1)
            ).detach()

    @contextmanager
    def use(self, features: list[torch.Tensor] | None, inputs: dict[str, torch.Tensor]):
        if self.features is not None:
            raise RuntimeError("Nested Qwen fusion calls are unsupported")
        self.features = features
        self.consumed = False
        self.task_context = self._task_context(self.model, inputs) if features is not None else None
        try:
            yield
            if features is not None and not self.consumed:
                raise RuntimeError("Qwen did not consume the Wan feature cache")
        finally:
            self.features = None
            self.task_context = None
            self.consumed = False

    def _video_features(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if self.features is None:
            return output
        if self.consumed:
            raise RuntimeError("Qwen consumed one Wan feature batch more than once")
        grid = kwargs.get("video_grid_thw", args[1] if len(args) > 1 else None)
        if grid is None:
            raise ValueError("Qwen get_video_features did not provide video_grid_thw")
        if len(self.features) != grid.shape[0]:
            raise ValueError("Wan cache/video count mismatch")

        def fuse(tokens):
            if isinstance(tokens, (list, tuple)):
                lengths = [value.shape[0] for value in tokens]
                merged = torch.cat(tuple(tokens), dim=0)
                fused = self.fusion(
                    merged, self.features, grid,
                    self.task_context.expand(len(self.features), -1),
                )
                split = torch.split(fused, lengths)
                return list(split) if isinstance(tokens, list) else split
            return self.fusion(
                tokens, self.features, grid,
                self.task_context.expand(len(self.features), -1),
            )

        if hasattr(output, "pooler_output"):
            output.pooler_output = fuse(output.pooler_output)
        elif isinstance(output, torch.Tensor):
            output = fuse(output)
        else:
            raise TypeError("Unsupported Qwen get_video_features return type")
        self.consumed = True
        return output


def load_qwen(model_path: str | Path):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Qwen model directory does not exist: {path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(str(path))
    configure_qwen_video_processor(processor)
    return model, processor
