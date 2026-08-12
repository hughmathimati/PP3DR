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
from models.PP3DR import GlobalBlock, LocalBlock
from xformers.ops import SwiGLU
from timm.layers import DropPath


class PointwiseSwiGLU(nn.Module):
    """
    Mathematically identical to a standard Linear SwiGLU, but uses
    1x1 Convolutions so it can operate directly on (B, C, H, W) spatial maps.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gate_proj = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.val_proj = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_features, out_features, kernel_size=1)

    def forward(self, x):
        gate = self.gate_proj(x)
        val = self.val_proj(x)
        hidden = F.silu(gate) * val
        return self.out_proj(hidden)

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_dim):
        super().__init__()
        self.upscale_dim = upscale_dim
        self.initial_proj = PointwiseSwiGLU(in_features=in_channels, hidden_features=in_channels,
                                            out_features=out_channels)
        self.residual = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=upscale_dim,
                                           stride=upscale_dim)
        self.residual_ls = LayerScale([out_channels, 1, 1])  # Along the channel dimension

    def forward(self, x):
        upscaled = F.interpolate(
            self.initial_proj(x),
            scale_factor=self.upscale_dim,
            mode='bilinear',
            align_corners=False
        )
        return upscaled + self.residual_ls(self.residual(x))

class FocalHead(nn.Module):
    def __init__(self, num_registers=5):
        self.layer_norm = nn.LayerNorm(dim)
        self.focal_proj = nn.Linear(num_registers * hidden_dim, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        (B, L, num_registers, dim)

        Returns
        -------
        (B,)
        """
        registers = self.layer_norm(x)
        f_mult = self.focal_proj(registers.flatten(start_dim=2)) # (B, L, 1)
        f_mult = f_mult.squeeze(-1).mean(dim=-1) # (B,)
        f_mult = F.softplus(f_mult)
        return f_mult * max(H, W)

class DepthProj(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.initial_proj = [
            UpscaleBlock(dim, dim // 64, 8), # 1/2
            UpscaleBlock(dim, dim // 16, 4), # 1/4
            UpscaleBlock(dim, dim // 4, 2), # 1/8
            nn.Identity() # 1/16
        ]
        self.fusion_proj = [
            UpscaleBlock(dim, dim // 4, 2), # 1/8
            UpscaleBlock(dim // 4, dim // 16, 2), # 1/4
            UpscaleBlock(dim // 16, dim // 64, 2), # 1/2
        ]
        self.final_proj = PointwiseSwiGLU(in_features=dim // 64, hidden_features=dim // 64, out_features=4)

    def forward(self, x):
        """
        Parameters
        ----------
        4 x (B, L, dim, H // patch_size, W // patch_size)

        Returns
        -------
        (B, L, H, W)
        """
        a, b, c, d = [self.initial_proj[i](y) for i, y in enumerate(x)]
        x = F.silu(c + self.fusion_proj[0](d))
        x = F.silu(b + self.fusion_proj[1](d))
        x = F.silu(a + self.fusion_proj[2](d))
        return rearrange(
            self.final_proj(x),
            "B L (p1 p2) H_p W_p -> B L (H_p p1) (W_p p2)",
            p1 = 2, p2 = 2
        )

class PointHead(nn.Module):
    def __init__(
            self,
            dim=1280,
            num_registers=5,
    ):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.focal_head = FocalHead()
        self.token_layer_norms = [nn.LayerNorm(dim) for _ in range(4)]
        self.depth_head = DepthHead()

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        Inputs consist of:
        1. The output of block 6/12 (index 5) from ViTTT
        2. The output at the end of ViTTT
        3. The output of block 18/36 (index 17) from PP3DR
        4. The output at the end of PP3DR

        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """

        x = rearrange(x, "(B L) X dim -> B L X dim", L=L)  # (B, L, num_registers + HW // 256, dim)
        # 4 x (B, L, HW // 256, dim)
        tokens = [self.token_layer_norms[i](y[:, :, self.num_registers:, :]) for i, y in enumerate(x)]
        tokens = [rearrange(y, "B L (H_p W_p) dim -> B L dim H_p W_p", H_p=H // 16) for y in tokens]

        return self.focal_head(x[-1][:, :, :self.num_registers, :]), self.depth_head(tokens)