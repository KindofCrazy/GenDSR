"""Raw and first-difference Wan features fused into Qwen visual tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _normalize(value: torch.Tensor, layer: nn.LayerNorm) -> torch.Tensor:
    return F.layer_norm(
        value.float(), layer.normalized_shape, layer.weight.float(),
        layer.bias.float(), layer.eps,
    )


def _align(feature: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(feature.shape[:3]) == shape:
        return feature
    t, h, w = shape
    temporal = F.interpolate(
        feature.permute(3, 0, 1, 2).unsqueeze(0),
        size=(t, feature.shape[1], feature.shape[2]),
        mode="trilinear", align_corners=False,
    )
    spatial = F.adaptive_avg_pool2d(
        temporal.squeeze(0).permute(1, 0, 2, 3), (h, w),
    )
    return spatial.permute(1, 0, 2, 3).permute(1, 2, 3, 0)


class _Identity(nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature


class _FirstDifference(nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(feature)
        if feature.shape[0] > 1:
            output[1:] = feature[1:] - feature[:-1]
        return output


class GenDSRFusion(nn.Module):
    """Add a conditioned Wan residual to each video's Qwen visual tokens.

    The module registration order and names match the source method's parameter
    layout. Each video is processed separately so gates cannot mix videos.
    """

    def __init__(self, hidden_size: int, feature_dim: int = 1536, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.feature_dim = int(feature_dim)
        self.spatial_merge_size = int(spatial_merge_size)
        self.temporal_operators = nn.ModuleDict({"raw": _Identity(), "delta": _FirstDifference()})
        self.projections = nn.ModuleDict({
            "raw": self._projection(),
            "delta": self._projection(),
        })
        self.gate_norms = nn.ModuleDict({
            "raw": nn.LayerNorm(hidden_size),
            "semantic": nn.LayerNorm(hidden_size),
            "task": nn.LayerNorm(hidden_size),
            "delta": nn.LayerNorm(hidden_size),
        })
        self.gate_heads = nn.ModuleDict({
            "delta": nn.Linear(4 * hidden_size, 1),
            "raw": nn.Linear(4 * hidden_size, 1),
        })
        nn.init.zeros_(self.gate_heads["delta"].weight)
        nn.init.constant_(self.gate_heads["delta"].bias, 0.1)
        nn.init.zeros_(self.gate_heads["raw"].weight)
        nn.init.constant_(self.gate_heads["raw"].bias, 1.0)
        self.branch_scales = nn.ParameterDict()
        self.learned_kernels = nn.ParameterDict()
        self.learned_betas = nn.ParameterDict()

    def _projection(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @staticmethod
    def _project(projection: nn.Module, feature: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        aligned = _align(feature, shape)
        parameter = next(projection.parameters())
        return projection(aligned.to(device=parameter.device, dtype=parameter.dtype))

    def _task(self, task: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if task is None:
            task = torch.zeros(self.hidden_size, device=reference.device, dtype=reference.dtype)
        task = task.to(device=reference.device, dtype=reference.dtype)
        if task.shape[-1] != self.hidden_size or task.numel() != self.hidden_size:
            raise ValueError("Task context must contain one hidden vector per video")
        return _normalize(task.reshape(1, 1, 1, -1), self.gate_norms["task"]).expand_as(reference.float())

    def _one(
        self, visual: torch.Tensor, feature: torch.Tensor,
        shape: tuple[int, int, int], task: torch.Tensor | None,
    ) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Wan feature must be [T,H,W,{self.feature_dim}]")
        feature = feature.to(device=visual.device, dtype=visual.dtype)
        raw = self._project(self.projections["raw"], feature, shape)
        delta = self.temporal_operators["delta"](feature)
        delta = self._project(self.projections["delta"], delta, shape)
        semantic = visual.reshape(*shape, self.hidden_size).to(dtype=raw.dtype)
        common = [
            _normalize(semantic, self.gate_norms["semantic"]),
            _normalize(raw, self.gate_norms["raw"]),
        ]
        task_grid = self._task(task, raw)
        delta_input = torch.cat(
            [*common, _normalize(delta, self.gate_norms["delta"]), task_grid], dim=-1,
        )
        raw_input = delta_input
        delta_weight = F.linear(
            delta_input, self.gate_heads["delta"].weight.float(),
            self.gate_heads["delta"].bias.float(),
        )
        raw_weight = F.linear(
            raw_input, self.gate_heads["raw"].weight.float(),
            self.gate_heads["raw"].bias.float(),
        )
        generator = raw_weight.to(raw.dtype) * raw + delta_weight.to(raw.dtype) * delta
        return (semantic + generator).reshape(-1, self.hidden_size)

    def forward(
        self, visual_embeds: torch.Tensor, features: Sequence[torch.Tensor],
        video_grid_thw: torch.Tensor, task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_embeds.ndim != 2 or visual_embeds.shape[-1] != self.hidden_size:
            raise ValueError("Qwen visual embeddings must be [tokens,hidden]")
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            raise ValueError("video_grid_thw must be [videos,3]")
        if len(features) != video_grid_thw.shape[0]:
            raise ValueError("Each video must have exactly one Wan feature record")
        if task_context is not None and task_context.shape != (len(features), self.hidden_size):
            raise ValueError("Task contexts must contain one row per video")
        chunks = []
        offset = 0
        for index, (feature, grid) in enumerate(zip(features, video_grid_thw.tolist(), strict=True)):
            t, h, w = [int(value) for value in grid]
            merge = self.spatial_merge_size
            if t <= 0 or h % merge or w % merge:
                raise ValueError(f"Invalid Qwen video grid: {(t, h, w)}")
            shape = (t, h // merge, w // merge)
            count = shape[0] * shape[1] * shape[2]
            chunk = visual_embeds[offset:offset + count]
            if chunk.shape[0] != count:
                raise ValueError("Qwen video token count does not match video_grid_thw")
            task = None if task_context is None else task_context[index]
            chunks.append(self._one(chunk, feature, shape, task))
            offset += count
        if offset != visual_embeds.shape[0]:
            raise ValueError("Qwen visual tokens exceed video_grid_thw accounting")
        return torch.cat(chunks, dim=0)

