import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


def _enable_block_checkpointing(block_list: nn.ModuleList, start_idx: int = 0):
    for i, blk in enumerate(block_list):
        if i >= start_idx:
            orig_fwd = blk.forward
            def _ckpt_fwd(*args, _orig=orig_fwd, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                return _orig(*args, **kwargs)
            blk.forward = _ckpt_fwd


def _init_all_vggt_submodules(model: nn.Module):
    """
    Initializes all submodules and explicitly populates all LinearKMaskedBias
    `bias_mask` buffers (which are initialized to math.nan by default).
    """
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "init_weights") and callable(m.init_weights):
            m.init_weights()
        elif hasattr(m, "_init_weights") and callable(m._init_weights):
            m._init_weights()

    # Explicitly initialize all LinearKMaskedBias `bias_mask` buffers:
    # Q (first third) = 1.0, K (middle third) = 0.0, V (last third) = 1.0
    for name, buf in model.named_buffers():
        if name.endswith("bias_mask") or not torch.isfinite(buf).all():
            o = buf.numel()
            buf.fill_(1.0)
            if o % 3 == 0:
                buf[o // 3 : 2 * (o // 3)].fill_(0.0)

    for name, module in model.named_modules():
        if "patch_embed" in name:
            continue
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class VGGTOmegaAblation(nn.Module):
    def __init__(
        self,
        dinov3_ckpt_path: str = "/vulcanscratch/hughma/dinov3_vitl16_pretrain_lvd1689m.pth",
        freeze_dinov3: bool = False,
        start_checkpointing: int = 12,
    ):
        super().__init__()

        self.encoder_blocks = 24
        self.decoder_blocks = 48

        # 1. Instantiate full VGGT-Omega
        self.model = VGGTOmega(
            patch_size=16,
            embed_dim=1024,
            enable_camera=True,
            enable_depth=True,
            enable_alignment=False,
        )

        # 2. Initialize all submodules & fill the 52 NaN bias_mask buffers
        _init_all_vggt_submodules(self.model)

        # 3. Load pretrained DINOv3 weights onto patch_embed
        dinov3_state = torch.load(dinov3_ckpt_path, map_location="cpu", weights_only=True)
        if "model" in dinov3_state:
            dinov3_state = dinov3_state["model"]
        elif "state_dict" in dinov3_state:
            dinov3_state = dinov3_state["state_dict"]
        missing, unexpected = self.model.aggregator.patch_embed.load_state_dict(dinov3_state, strict=False)
        print(f"[DINOv3 Init] Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Verify zero NaN parameters or buffers remain after init
        bad_buffers = [n for n, b in self.model.named_buffers() if not torch.isfinite(b).all()]
        assert len(bad_buffers) == 0, f"Non-finite buffers remain: {bad_buffers}"

        # 4. Freeze or checkpoint DINOv3
        if freeze_dinov3:
            self.model.aggregator.patch_embed.requires_grad_(False)
            self.model.aggregator.patch_embed.eval()
        elif hasattr(self.model.aggregator.patch_embed, "blocks"):
            _enable_block_checkpointing(self.model.aggregator.patch_embed.blocks, start_idx=0)

        # 5. Enable gradient checkpointing on Aggregator blocks
        for attr in ("frame_blocks", "global_blocks", "inter_frame_blocks"):
            if hasattr(self.model.aggregator, attr):
                _enable_block_checkpointing(getattr(self.model.aggregator, attr), start_idx=start_checkpointing)

        # 6. Initialize final DenseHead output layers near 0 so exp(-s) = 1.0 at Step 0
        nn.init.normal_(self.model.dense_head.proj.weight, std=1e-3)
        nn.init.zeros_(self.model.dense_head.proj.bias)
        nn.init.normal_(self.model.dense_head.proj_conf.weight, std=1e-3)
        nn.init.zeros_(self.model.dense_head.proj_conf.bias)

    def forward(self, images: torch.Tensor, rope_x=None, rope_y=None, original_height=None, original_width=None) -> dict:
        B, L, C, H, W = images.shape

        preds = self.model(images)

        # 1. Direct unbounded log_depths and raw_uncertainty (s) from DenseHead
        log_depths = preds["depth"].squeeze(-1)
        raw_uncertainty = preds["depth_conf"]

        # 2. Safe pose decoding: prevent 0/0 quaternion NaN and tan(0) focal Inf
        pose_enc = preds["pose_enc"].to(torch.float32)
        t_enc = pose_enc[..., :3]
        q_enc = pose_enc[..., 3:7]
        q_norm = torch.linalg.vector_norm(q_enc, dim=-1, keepdim=True)
        q_safe = torch.where(
            q_norm < 1e-6,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=pose_enc.device, dtype=pose_enc.dtype),
            q_enc,
        )
        fov_safe = F.softplus(pose_enc[..., 7:]).clamp(min=0.05, max=3.0)
        safe_pose_enc = torch.cat([t_enc, q_safe, fov_safe], dim=-1)

        extrinsics, intrinsics = encoding_to_camera(safe_pose_enc, (H, W))

        current_R = extrinsics[:, :-1, :3, :3]
        next_R = extrinsics[:, 1:, :3, :3]
        relative_rotations = next_R @ current_R.transpose(-1, -2)

        current_t = extrinsics[:, :-1, :3, 3]
        next_t = extrinsics[:, 1:, :3, 3]
        relative_translations = next_t - current_t

        focal_length = intrinsics[:, :, 0, 0].mean(dim=-1).clamp(min=1.0, max=1e5)

        return {
            "log_depths": torch.clamp(log_depths, min=-15.0, max=15.0).to(torch.float32),
            "raw_uncertainty": torch.clamp(raw_uncertainty, min=-15.0, max=15.0).to(torch.float32),
            "focal_length": focal_length.to(torch.float32),
            "relative_camera_translations": relative_translations.to(torch.float32),
            "relative_camera_rotations": relative_rotations.to(torch.float32),
        }