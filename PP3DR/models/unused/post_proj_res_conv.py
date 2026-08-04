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
from models.PP3DR_Dino import PP3DR_Dino
from xformers.ops import SwiGLU
from timm.layers import DropPath


class res_conv_head(nn.Module):
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
        # self.depths_proj = nn.Linear(hidden_dim, 256)
        self.depths_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 256)
        )
        self.num_registers = num_registers
        self.num_blocks = blocks

        # self.layer_scale = LayerScale(1)
        # self.res_conv = nn.Sequential(
        #     # 31 is 1 less than 2 times the patch size, 16.
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5,  padding=2)
        # )

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
        B, dim = x.shape[0], x.shape[-1]
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
        log_depths = rearrange(
            self.depths_proj(tokens),  # (B, L, HW // 256, 256)
            "B L (h_p w_p) (p1 p2) -> B L (h_p p1) (w_p p2)",
            h_p=H // 16, w_p=W // 16, p1=16, p2=16
        ).flatten(start_dim=0, end_dim=1) # (B * L, H, W)
        # log_depths = log_depths + self.layer_scale(self.res_conv(log_depths.unsqueeze(1)).squeeze(1))
        # depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
        return log_depths.view(log_depths.shape[0] // L, L, *log_depths.shape[1:]), fx, fy, cx, cy

@torch.compile
class post_proj_res_conv(PP3DR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=res_conv_head)