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
from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT, BidirectionalLaCT_output_dim
from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
from models.ViTTT import ViTTT
from models.PP3DR import GlobalBlock, LocalBlock, PP3DR, LayerScale
from xformers.ops import SwiGLU
from timm.layers import DropPath


@torch.compile()
class ttt_depth_proj_head(nn.Module):
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
        self.pre_depth_proj_layer_norm = nn.LayerNorm(hidden_dim) # Pre-projection layer norm.
        self.post_depth_proj_layer_norm = nn.LayerNorm(256) # Post-projection layer norm.
        self.focal_proj = nn.Linear(num_registers * hidden_dim, 4)  # fx, fy, cx, cy

        # rope2d (and rope3d) are made for a head-dimension of 1280 / 20 = 64. However, with only 16 heads, the rope-
        # dimension is actually 1280 / 16 - 80. We'll solve this by simply not aqpplying RoPE to the last 16 dimensions.
        self.depth_proj = BidirectionalLaCT_output_dim(dim=hidden_dim, num_heads=16, v_dim=256)
        self.num_registers = num_registers
        self.num_blocks = blocks

    def forward(self, x: torch.Tensor, rope2d, rope3d, L, H, W) -> torch.Tensor:
        """
        Input to the model consists of the outputs of the last global block AND the last local block, concatenated
        along the last dimension. Thus, the input dimension is 2 * 1280.

        The depths are obtained by a down-projection via TTT block (1280 to 256). The 5 register tokens per frame are
        kept for the TTT block and discarded afterward, before passing the result into a layer norm.

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

        x = rearrange(x, "(B L) X dim -> B L X dim", L = L)
        registers = self.registers_layer_norm(x[:, :, :self.num_registers, :]) # (B, L, num_registers, dim)
        # (B * L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=-2)).unbind(-1)
        # fx and fy will be passed into softplus to ensure they're always positive.
        fx, fy = F.softplus(fx), F.softplus(fy)
        # cx and cy will actually be the residuals to the image center point.
        cx, cy = cx + W / 2, cy + H / 2

        sin, cos = rope2d # shape (B * L, HW, 64)
        # Padding with 16 zeroes on the left to a head-dim of 80.
        sin, cos = F.pad(sin, (16, 0)), F.pad(cos, (16, 0)) # (B * L, HW, 80)
        # Passing rope2d because the down-projection is a local operation.
        # (B, L, self.num_registers + HW // 256, 256)
        down_proj = self.depth_proj(self.pre_depth_proj_layer_norm(x), (sin, cos), L)
        # We must take into account that each token represents the 16x16 patch at that location.
        # Naively rearranging leads to each 16x16 patch being mapped to a contiguous line of 256 pixels.
        log_depths = rearrange(
            self.post_depth_proj_layer_norm(down_proj[:, :, self.num_registers:, :]), # (B, L, HW // 256, 256)
            "B L (h_p w_p) (p1 p2) -> B L (h_p p1) (w_p p2)",
            h_p=H // 16, w_p=W // 16, p1=16, p2=16
        )
        # depths has shape (B, L, H, W). fx, fy, cx, and cy have shape (B, L).
        return log_depths, fx, fy, cx, cy



@torch.compile
class ttt_depth_proj(PP3DR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=ttt_depth_proj_head)