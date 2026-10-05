from types import SimpleNamespace

import torch
from torch import nn

from gendsr.fusion import GenDSRFusion
from gendsr.qwen import QwenFusionAdapter


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(8, 8)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return SimpleNamespace(pooler_output=pixel_values_videos)


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Core()
        self.embeddings = nn.Embedding(10, 8)

    def get_input_embeddings(self):
        return self.embeddings


def test_video_hook_consumes_cache_and_preserves_native_condition():
    model = _Qwen()
    adapter = QwenFusionAdapter(model, GenDSRFusion(8, 6))
    grid = torch.tensor([[3, 2, 4]])
    visual = torch.randn(6, 8)
    feature = torch.randn(3, 4, 6, 6)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 4]]),
    }
    with adapter.use([feature], inputs):
        fused = model.model.get_video_features(visual, video_grid_thw=grid).pooler_output
    assert fused.shape == visual.shape
    assert not torch.equal(fused, visual)
    with adapter.use(None, inputs):
        native = model.model.get_video_features(visual, video_grid_thw=grid).pooler_output
    torch.testing.assert_close(native, visual)

