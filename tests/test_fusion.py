import pytest
import torch

from gendsr.checkpoint import config_digest, load_fusion_strict
from gendsr.fusion import GenDSRFusion


def test_forward_gradients_and_video_isolation():
    torch.manual_seed(42)
    model = GenDSRFusion(hidden_size=8, feature_dim=6)
    visual = torch.randn(12, 8, requires_grad=True)
    first = torch.randn(3, 4, 6, 6, requires_grad=True)
    second = torch.randn(3, 4, 6, 6, requires_grad=True)
    grid = torch.tensor([[3, 2, 4], [3, 2, 4]])
    task = torch.randn(2, 8)
    output = model(visual, [first, second], grid, task)
    assert output.shape == visual.shape
    changed = model(visual, [first, second + 1], grid, task)
    torch.testing.assert_close(output[:6], changed[:6], rtol=0, atol=0)
    assert not torch.equal(output[6:], changed[6:])
    output.square().sum().backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
    assert model.projections["raw"][1].weight.grad is not None
    assert model.gate_heads["delta"].weight.grad is not None


def test_fusion_state_load_is_strict():
    model = GenDSRFusion(hidden_size=8, feature_dim=6)
    state = model.state_dict()
    restored = GenDSRFusion(hidden_size=8, feature_dim=6)
    restored.load_state_dict(state, strict=True)
    with pytest.raises(RuntimeError):
        restored.load_state_dict({key: value for key, value in state.items() if key != "gate_heads.raw.bias"}, strict=True)


def test_checkpoint_rejects_missing_weights_and_wrong_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("method: fixed\n", encoding="utf-8")
    model = GenDSRFusion(hidden_size=8, feature_dim=6)
    payload = {
        "format_version": 1, "config_sha256": config_digest(config),
        "hidden_size": 8, "feature_dim": 6, "state_dict": model.state_dict(),
    }
    torch.save(payload, tmp_path / "gendsr_fusion.pt")
    load_fusion_strict(tmp_path, config)
    config.write_text("method: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration identity"):
        load_fusion_strict(tmp_path, config)
    config.write_text("method: fixed\n", encoding="utf-8")
    payload["state_dict"].pop("gate_heads.raw.bias")
    torch.save(payload, tmp_path / "gendsr_fusion.pt")
    with pytest.raises(RuntimeError):
        load_fusion_strict(tmp_path, config)


def test_fusion_rejects_video_count_mismatch():
    model = GenDSRFusion(hidden_size=8, feature_dim=6)
    with pytest.raises(ValueError, match="one Wan feature"):
        model(torch.zeros(6, 8), [], torch.tensor([[3, 2, 4]]))
