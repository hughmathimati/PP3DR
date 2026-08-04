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
from models.ViTTT import ViTTT
from models.PP3DR import GlobalBlock, LocalBlock, PP3DR, LayerScale
from xformers.ops import SwiGLU
from timm.layers import DropPath


class conv_upsample_head(nn.Module):
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
        self.num_registers = num_registers
        self.num_blocks = blocks

        self.upsample = nn.Upsample(scale_factor=2)
        self.transpose_convolutions = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim // 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=hidden_dim // 4, out_channels=hidden_dim // 16, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=hidden_dim // 16, out_channels=hidden_dim // 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=hidden_dim // 64, out_channels=1, kernel_size=2, stride=2),
        ])
        self.res_convolutions = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_dim // 4, out_channels=hidden_dim // 4, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=hidden_dim // 16, out_channels=hidden_dim // 16, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=hidden_dim // 64, out_channels=hidden_dim // 64, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        ])
        self.layer_scales = nn.ModuleList([
            LayerScale([hidden_dim // 4, 1, 1]),
            LayerScale([hidden_dim // 16, 1, 1]),
            LayerScale([hidden_dim // 64, 1, 1]),
            LayerScale([1, 1, 1]),
        ])

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
        registers = self.registers_layer_norm(x[:, :, :self.num_registers, :])  # (B, L, num_registers, dim)
        tokens = self.tokens_layer_norm(x[:, :, self.num_registers:, :])  # (B, L, HW // 256, dim)
        # (B, L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)
        # fx and fy will be passed into softplus to ensure they're always positive.
        fx, fy = F.softplus(fx), F.softplus(fy)
        # cx and cy will actually be the residuals to the image center point.
        cx, cy = cx + W / 2, cy + H / 2

        # We must take into account that each token represents the 16x16 patch at that location.
        # Naively rearranging leads to each 16x16 patch being mapped to a contiguous line of 256 pixels.
        tokens = rearrange(tokens, "B L (h_p w_p) dim -> (B L) dim h_p w_p", h_p=H // 16, w_p=W // 16)
        for i in range(4):
            tokens = self.transpose_convolutions[i](tokens)
            tokens = tokens + self.layer_scales[i](self.res_convolutions[i](tokens))
            if i < 3:
                tokens = F.gelu(tokens)
        log_depths = tokens.squeeze(1).view(x.shape[0], L, H, W)
        # log_depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
        return log_depths, fx, fy, cx, cy

@torch.compile
class conv_upsample(PP3DR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=conv_upsample_head)