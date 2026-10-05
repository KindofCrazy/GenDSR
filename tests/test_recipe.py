from pathlib import Path

import pytest
from torch import nn

from gendsr.config import load_config
from gendsr.fusion import GenDSRFusion
from gendsr.train import optimizer_groups


def test_optimizer_uses_three_rates_and_excludes_norm_bias_from_decay():
    pytest.importorskip("transformers")

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = nn.Module()
            self.model.visual.tower = nn.Linear(8, 8)
            self.model.visual.merger = nn.Linear(8, 8)
            self.model.language = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))

    model = SmallModel()
    model.model.visual.requires_grad_(False)
    model.model.visual.merger.requires_grad_(True)
    fusion = GenDSRFusion(8, 6)
    model.vgm_add_residual = fusion
    config = load_config(Path(__file__).parents[1] / "configs" / "gendsr.yaml")
    groups = {group["name"]: group for group in optimizer_groups(model, fusion, config)}
    assert groups["base_decay"]["lr"] == 2e-7
    assert groups["projector_decay"]["lr"] == 1e-6
    assert groups["fusion_decay"]["lr"] == 1e-5
    assert groups["fusion_no_decay"]["weight_decay"] == 0.0
    assert groups["fusion_decay"]["weight_decay"] == 0.01
    assert id(fusion.gate_heads["delta"].bias) in {
        id(parameter) for parameter in groups["fusion_no_decay"]["params"]
    }
    assert id(fusion.projections["raw"][1].weight) in {
        id(parameter) for parameter in groups["fusion_decay"]["params"]
    }
