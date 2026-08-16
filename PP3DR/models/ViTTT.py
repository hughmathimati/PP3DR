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
        self.scale = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

@torch.compile()
class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path = 0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = BidirectionalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope)))
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
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
            # In the ViTTT code found in the ViTTT folder, the default value for start_checkpointing is 8.
            # The reason it's increased to 12 here is so the non-checkpointed blocks take up around the same VRAM whether
            # freeze_feature_extractor is enabled or not, so I obtain maximum VRAM usage without having to adjust the
            # batch size every time I change freeze_feature_extractor.
            # Once I start doing long-sequence-length finetuning, I will almost certainly decrease this to maximize the
            # sequence length I can pass in.
            start_checkpointing = 12,
            drop_rates = None,
#            output_blocks=[1, 4, 11]
            output_blocks = list(range(12))
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
        self.final_layer_norm = nn.LayerNorm(dim)
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
        x = self.patch_embed(x)
        rope = self.rope(H // 16, W // 16)
        """
        Why is the shape (196, 64)?
        Our input tensor was 224x224; with 16x16 patches, it becomes 14x14 -> HW sequence length of 196 (we don't apply
        RoPE to special tokens).
        Our RoPE dimension is 64 because our total dimension is 1280 and we have 20 heads; 1280 / 20 = 64.
        """
        # print("RoPE shapes:", rope[0].shape, rope[1].shape)
        repeated_class_and_registers = self.class_and_registers.unsqueeze(0).repeat(B, 1, 1)
        x = torch.cat((repeated_class_and_registers, x), dim = 1)
        # assert x.shape == (B, 5 + H * W // 256, self.dim), f"x.shape should be {(B, 5 + H * W // 256, self.dim)} but is instead {x.shape}."
        outputs = []
        for i, block in enumerate(self.blocks):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(block, x, rope, use_reentrant = False)
            else:
                x = block(x, rope)
            if i in self.output_blocks:
                outputs.append(x)

        return self.final_layer_norm(x), outputs
