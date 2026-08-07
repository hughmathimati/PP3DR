import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.v2 as transforms

from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT
from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
from models.PP3DR import PointwiseSwiGLU, GlobalBlock, LocalBlock, PP3DR
from models.ViTTT import ViTTT
from xformers.ops import SwiGLU
from timm.layers import DropPath

class DepthProj(nn.Module):
    """
    Down-proj into bilinear upscale up to 4x4 patches, then simple linear projection to get up to original resolution.
    8-2 -> hidden_dim = 8 * 8 = 64, upscale by 8x, proj to 2^2 = 4
    4-4 -> hidden_dim = 16 * 8 = 128, upscale by 4x, proj to 4^2 = 16
    2-8 -> hidden_dim = 64 * 8 = 512, upscale by 2x, proj to 8^2 = 64
    """
    def __init__(self, dim=1280):
        super().__init__()
        self.upscale_dim = 2
        self.proj_dim = 8
        out_dim = self.proj_dim**2
        hidden_dim = out_dim * 8
        self.initial_proj = PointwiseSwiGLU(in_features=dim, hidden_features=dim, out_features=hidden_dim)
        self.final_proj = PointwiseSwiGLU(in_features=hidden_dim, hidden_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        """
        x has shape (B, L, H // 16, W // 16, dim)
        """
        B = x.shape[0]
        x = rearrange(x, "B L H_p W_p dim -> (B L) dim H_p W_p")
        upscaled = F.interpolate(
            self.initial_proj(x), # (B * L, hidden_dim, H // 16, W // 16)
            scale_factor=self.upscale_dim, # (B * L, H // 4, W // 4, hidden_dim)
            mode='bilinear',
            align_corners=False
        )
        x = rearrange(
            self.final_proj(upscaled), # (B * L, 16, H // 4, W // 4)
            "(B L) (p_H p_W) H_p W_p -> B L (H_p p_H) (W_p p_W)",
            B=B, p_H=self.proj_dim, p_W=self.proj_dim
        )
        return x # (B, L, H, W)


class mixed_head(nn.Module):
    def __init__(
            self,
            input_dim=2 * 1280,
            hidden_dim=1280,
            num_registers=5,
            num_heads=20,
            blocks=4,  # Pi3 has 5 transformer blocks per decoder
            ffn_ratio=4,
    ):
        super().__init__()
        assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even for alternating global and frame-wise attention"
        self.blocks = blocks
        self.blocks_each = blocks // 2
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim

        self.global_blocks = nn.ModuleList(
            [GlobalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
        self.local_blocks = nn.ModuleList(
            [LocalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])

        self.registers_layer_norm = nn.LayerNorm(hidden_dim)  # Pre-projection layer norm.
        self.tokens_layer_norm = nn.LayerNorm(hidden_dim)  # Pre-projection layer norm.
        self.focal_proj = nn.Linear(num_registers * hidden_dim, 4)  # fx, fy, cx, cy
        self.depth_proj = DepthProj(dim=hidden_dim)
        self.num_registers = num_registers

    def forward(self, x: torch.Tensor, rope2d, rope3d, L, H, W) -> torch.Tensor:
        """
        Input to the model consists of the outputs of the last global block AND the last local block, concatenated
        along the last dimension. Thus, the input dimension is 2 * 1280.

        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        x = self.input_proj(x)
        for i in range(self.blocks_each):
            if self.training:
                # Global attention: absorb frame-length into patch-length dimension.
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        x = rearrange(x, "(B L) X dim -> B L X dim", L=L)  # (B, L, num_registers + HW // 256, dim)
        B = x.shape[0]
        registers = self.registers_layer_norm(x[:, :, :self.num_registers, :])  # (B, L, num_registers, dim)
        # (B, L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)
        # fx and fy will be passed into softplus to ensure they're always positive.
        fx, fy = F.softplus(fx), F.softplus(fy)
        # cx and cy will actually be the residuals to the image center point.
        cx, cy = cx + W / 2, cy + H / 2

        tokens = self.tokens_layer_norm(x[:, :, self.num_registers:, :])  # (B, L, HW // 256, dim)
        log_depths = self.depth_proj(tokens.view(B, L, H // 16, W // 16, self.hidden_dim))

        # log_depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
        return log_depths, fx, fy, cx, cy


@torch.compile()
class PP3DR_mixed_head(PP3DR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=mixed_head)