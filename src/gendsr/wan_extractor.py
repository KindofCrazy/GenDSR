"""Fixed Wan2.1-T2V-1.3B feature extraction for GenDSR."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .wan_schema import WanFeatureConfig


class WanClipFeatureExtractor:
    """Wan2.1-T2V clip-level feature extractor.

    This backend requires a local Wan2.1 repo and checkpoint directory. It does
    not download anything. It encodes the sampled clip with WanVAE, applies
    fixed-timestep noise, hooks the requested DiT block, and returns the hidden
    token grid as [Tg,Hg,Wg,C].
    """

    backend_name = "wan2_1_t2v_clip"

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        wan_repo: str,
        config: WanFeatureConfig = WanFeatureConfig(),
        device: str = "cuda",
        noise_seed: int = 42,
        scheduler_shift: float = 5.0,
    ):
        checkpoint = Path(checkpoint_dir)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Wan checkpoint dir does not exist: {checkpoint_dir}")

        repo = Path(wan_repo)
        if not repo.exists():
            raise FileNotFoundError(f"Wan repo does not exist: {wan_repo}")

        import torch

        if config.generator_task != "t2v-1.3B":
            raise ValueError(f"unsupported generator task: {config.generator_task}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        sys.path.insert(0, str(repo))
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae import WanVAE
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        self.config = config
        self.device = torch.device(device)
        self.noise_seed = int(noise_seed)
        self.cache_dtype = torch.bfloat16
        self.scheduler_shift = float(scheduler_shift)

        wan_cfg = WAN_CONFIGS[config.generator_task]
        self.param_dtype = getattr(wan_cfg, "param_dtype", torch.bfloat16)
        self.text_len = int(getattr(wan_cfg, "text_len", 512))
        self.text_dim = int(getattr(wan_cfg, "text_dim", 4096))
        self.vae = WanVAE(
            vae_pth=str(checkpoint / wan_cfg.vae_checkpoint),
            dtype=self.param_dtype,
            device=self.device,
        )
        self.model = WanModel.from_pretrained(str(checkpoint))
        self.model.eval().requires_grad_(False).to(self.device)
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(wan_cfg.num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )
        self.scheduler.set_timesteps(
            int(wan_cfg.num_train_timesteps),
            device=self.device,
            shift=self.scheduler_shift,
        )
        self.selected_timestep = _select_timestep(self.scheduler.timesteps, config.diffusion_timestep)
        text_encoder = T5EncoderModel(
            text_len=wan_cfg.text_len,
            dtype=wan_cfg.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(checkpoint / wan_cfg.t5_checkpoint),
            tokenizer_path=str(checkpoint / wan_cfg.t5_tokenizer),
        )
        with torch.inference_mode():
            self.prompt_context = text_encoder([""], torch.device("cpu"))[0].detach().to(
                dtype=self.param_dtype
            ).contiguous()
        del text_encoder

    def extract(self, frames: np.ndarray, *, metadata: dict[str, Any]) -> torch.Tensor:
        import torch

        if self.config.generator_layer < 0 or self.config.generator_layer >= len(self.model.blocks):
            raise ValueError(f"generator_layer out of range: {self.config.generator_layer}")

        metadata["generator_selected_timestep"] = float(self.selected_timestep.item())
        metadata["generator_noise_schedule"] = "FlowUniPCMultistepScheduler"
        metadata["generator_scheduler_shift"] = self.scheduler_shift
        metadata["generator_noise_seed"] = self.noise_seed
        metadata["feature_dtype"] = str(self.cache_dtype).replace("torch.", "")
        video = _frames_to_wan_tensor(frames).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.param_dtype):
            latent = self.vae.encode([video])[0]
            noisy_latent = _add_scheduler_noise(
                latent,
                scheduler=self.scheduler,
                timestep=self.selected_timestep,
                seed=self.noise_seed,
            )
            seq_len = int(metadata["generator_num_tokens"])
            timestep = self.selected_timestep.reshape(1)
            context = [self.prompt_context.to(self.device)]
            captured: dict[str, torch.Tensor] = {}

            def hook(_module, _inputs, output):
                captured["feature"] = output.detach()

            handle = self.model.blocks[self.config.generator_layer].register_forward_hook(hook)
            try:
                self.model([noisy_latent], t=timestep, context=context, seq_len=seq_len)
            finally:
                handle.remove()

        if "feature" not in captured:
            raise RuntimeError("Wan DiT hook did not capture a feature tensor")

        feature = captured["feature"][0]
        token_t, token_h, token_w = metadata["generator_token_grid_shape"]
        expected_tokens = token_t * token_h * token_w
        if feature.shape[0] != expected_tokens:
            raise RuntimeError(f"Wan feature token count {feature.shape[0]} != expected {expected_tokens}")
        if feature.shape[-1] != self.config.generator_feat_dim:
            raise RuntimeError(
                f"Wan feature dim {feature.shape[-1]} != configured {self.config.generator_feat_dim}"
            )
        feature = feature.reshape(token_t, token_h, token_w, -1)
        return feature.to(dtype=self.cache_dtype).cpu()


def _frames_to_wan_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB [T,H,W,3] frames to Wan video tensor [3,T,H,W]."""
    import torch

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape [T,H,W,3]")
    tensor = torch.from_numpy(frames.astype(np.float32)).permute(3, 0, 1, 2)
    tensor = tensor / 127.5 - 1.0
    return tensor.contiguous()


def _select_timestep(timesteps: Any, target_timestep: int):
    import torch

    target = torch.tensor(int(target_timestep), device=timesteps.device, dtype=timesteps.dtype)
    return timesteps[torch.argmin(torch.abs(timesteps - target))]


def _add_scheduler_noise(latent: Any, *, scheduler: Any, timestep: Any, seed: int):
    import torch

    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    return scheduler.add_noise(latent.unsqueeze(0), noise.unsqueeze(0), timestep.reshape(1))[0]
