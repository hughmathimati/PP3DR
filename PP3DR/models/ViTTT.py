import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
from torch.utils.checkpoint import checkpoint

# try block contains imports for calling from trainer, and except block contains imports for running this file itself
try:
    from .BidirectionalLaCT import BidirectionalLaCT
    from .pos_embed import RopePositionEmbedding
except:
    from BidirectionalLaCT import BidirectionalLaCT
    from pos_embed import RopePositionEmbedding
from xformers.ops import SwiGLU
from timm.layers import DropPath

@torch.compile()
class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x

class SpatialDepthwiseConv(nn.Module):
    """
    Applies a residual 3x3 depthwise convolution strictly to the 2D patch tokens,
    leaving register tokens untouched. Zero-initialized to start as an identity.
    """
    def __init__(self, dim: int, num_registers: int = 5):
        super().__init__()
        self.num_registers = num_registers
        # Recall that our patch size is 16, meaning a 3x3 kernel actually takes into account nine 16x16 patches.
        self.depthwise_convolution = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        nn.init.zeros_(self.depthwise_convolution.weight)
        nn.init.zeros_(self.depthwise_convolution.bias)

    def forward(self, x: torch.Tensor, H_p: int, W_p: int) -> torch.Tensor:
        # x: (B * L, num_registers + H_p * W_p, dim)
        regs = x[:, :self.num_registers, :]
        patches = x[:, self.num_registers:, :]

        patches_2d = rearrange(patches, "BL (Hp Wp) D -> BL D Hp Wp", Hp=H_p, Wp=W_p)
        patches_2d = patches_2d + self.depthwise_convolution(patches_2d)
        patches = rearrange(patches_2d, "BL D Hp Wp -> BL (Hp Wp) D")

        return torch.cat([regs, patches], dim=1)

@torch.compile()
class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path = 0):
        super().__init__()
        self.conv1 = SpatialDepthwiseConv(dim)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = BidirectionalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.conv2 = SpatialDepthwiseConv(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope, H_p: int, W_p: int) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.TTT(
                self.conv1(self.layer_norm_1(x), H_p, W_p),
                rope
        )))
        x = x + self.drop_path(self.layer_scale_2(self.ffn(
            self.conv2(self.layer_norm_2(x), H_p, W_p)
        )))
        return x

@torch.compile()
class ViTTT(nn.Module):
    def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 20,
            blocks = 12, # DINOv3 H+ has 32 layers.
            ffn_ratio = 4,
            num_registers = 5,
            start_checkpointing = 12,
            drop_rates = None,
            final_layer_norm = True,
            output_blocks = None
    ):
        super().__init__()
        self.dim = dim
        self.rope = RopePositionEmbedding(dim, num_heads=num_heads)
        self.patch_conv = nn.Conv2d(3, dim, kernel_size=(16, 16), stride=(16, 16))

        drop_rates = [
            x.item() for x in
            (torch.linspace(0, 0.1, blocks) if drop_rates is None else drop_rates)
        ]
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ffn_ratio, drop_rates[i]) for i in range(blocks)
        ])

        self.class_and_registers = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.final_layer_norm = nn.LayerNorm(dim) if final_layer_norm else nn.Identity()
        self.start_checkpointing = start_checkpointing
        self.output_blocks = output_blocks

    def patch_embed(self, x):
        # B, 3, H, W -> B, dim, H // 16, W // 16 -> B, HW // 256, dim
        return self.patch_conv(x).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (batch size, RGB, height, width)

        Returns
        -------
        (batch size, 5 + height * width // 16**2, dim)
        """
        # We'll handle shapes not divisible by 16 during data processing, as DINO also doesn't handle this.
        # assert len(x.shape) == 4, f"x.shape should have length 4, but is instead {x.shape}"
        B, C, H, W = x.shape
        H_p, W_p = H // 16, W // 16
        x = self.patch_embed(x)
        rope = self.rope(H_p, W_p)
        repeated_class_and_registers = self.class_and_registers.unsqueeze(0).repeat(B, 1, 1)
        x = torch.cat((repeated_class_and_registers, x), dim = 1)
        outputs = []
        for i, block in enumerate(self.blocks):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(block, x, rope, H_p, W_p, use_reentrant = False)
            else:
                x = block(x, rope, H_p, W_p)
            if i in self.output_blocks:
                outputs.append(x)

        return self.final_layer_norm(x), outputs