"""Strict checkpoint format for newly trained GenDSR models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .fusion import GenDSRFusion


def config_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(model, processor, fusion: GenDSRFusion, output: str | Path, config_path: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("vgm_add_residual.")
    }
    model.save_pretrained(output, state_dict=model_state, safe_serialization=True)
    processor.save_pretrained(output)
    torch.save(
        {
            "format_version": 1,
            "config_sha256": config_digest(config_path),
            "hidden_size": fusion.hidden_size,
            "feature_dim": fusion.feature_dim,
            "state_dict": {key: value.detach().cpu() for key, value in fusion.state_dict().items()},
        },
        output / "gendsr_fusion.pt",
    )
    (output / "gendsr_checkpoint.json").write_text(
        json.dumps({"format_version": 1, "config_sha256": config_digest(config_path)}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_fusion_strict(path: str | Path, config_path: str | Path) -> GenDSRFusion:
    payload = torch.load(Path(path) / "gendsr_fusion.pt", map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("config_sha256") != config_digest(config_path):
        raise ValueError("GenDSR checkpoint format or configuration identity mismatch")
    fusion = GenDSRFusion(
        hidden_size=int(payload["hidden_size"]), feature_dim=int(payload["feature_dim"]),
    )
    fusion.load_state_dict(payload["state_dict"], strict=True)
    return fusion

