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
from models.PP3DR import GlobalBlock, LocalBlock, PP3DR
from models.ViTTT import ViTTT
from xformers.ops import SwiGLU
from timm.layers import DropPath


class FactorizedSwiGLUINR(nn.Module):
    def __init__(self, token_dim, hidden_dim, out_dim, grid_dim=2):
        super().__init__()

        # 1. Factorize the Gate Projection
        self.gate_token = nn.Linear(token_dim, hidden_dim, bias=True)
        self.gate_grid = nn.Linear(grid_dim, hidden_dim, bias=False)  # Bias in token is enough

        # 2. Factorize the Value Projection
        self.val_token = nn.Linear(token_dim, hidden_dim, bias=True)
        self.val_grid = nn.Linear(grid_dim, hidden_dim, bias=False)

        # 3. Final Output Projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, tokens, grid):
        """
        tokens: (B, L, Patches, 1280)
        grid: (256, 2)
        """

        # --- COMPUTE THE GATE ---
        # Project isolated components
        gate_t = self.gate_token(tokens)  # (B, L, Patches, hidden_dim)
        gate_g = self.gate_grid(grid)  # (256, hidden_dim)

        # Broadcast Addition: Fuses them into (B, L, Patches, 256, hidden_dim)
        gate = gate_t.unsqueeze(-2) + gate_g.view(1, 1, 1, 256, -1)

        # --- COMPUTE THE VALUE ---
        val_t = self.val_token(tokens)
        val_g = self.val_grid(grid)
        val = val_t.unsqueeze(-2) + val_g.view(1, 1, 1, 256, -1)

        # --- SWIGLU FUSION ---
        hidden = F.silu(gate) * val

        # --- FINAL OUTPUT ---
        # Shape: (B, L, Patches, 256, out_dim)
        return self.out_proj(hidden)


@torch.compile()
class FullSwiGLUHead(nn.Module):
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
        self.depth_proj = FactorizedSwiGLUINR(hidden_dim, 256, 1)

        self.num_registers = num_registers
        self.register_buffer("local_grid", self.generate_local_grid(16))

    def generate_local_grid(self, patch_size):
        # Generates a grid from -1.0 to 1.0 for the 16x16 patch
        coords = torch.linspace(-1.0, 1.0, steps=patch_size)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        # Shape: (256, 2)
        grid = torch.stack([x, y], dim=-1)#.reshape(-1, 2)
        return grid

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
        # (B, L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)
        # fx and fy will be passed into softplus to ensure they're always positive.
        fx, fy = F.softplus(fx), F.softplus(fy)
        # cx and cy will actually be the residuals to the image center point.
        cx, cy = cx + W / 2, cy + H / 2

        tokens = self.tokens_layer_norm(x[:, :, self.num_registers:, :])  # (B, L, HW // 256, dim)
        # self.tokens_proj(tokens) is (B, L, HW // 256, 256). self.grid_proj(self.local_grid) is (256, 256).
        log_depths = self.depth_proj(tokens, self.local_grid) # (B, L, HW // 256, 256, 1)
        # We must take into account that each token represents the 16x16 patch at that location.
        # Naively rearranging leads to each 16x16 patch being mapped to a contiguous line of 256 pixels.
        log_depths = rearrange(
            log_depths.squeeze(-1), # (B, L, HW // 256, 256)
            "B L (h_p w_p) (p1 p2) -> B L (h_p p1) (w_p p2)",
            h_p=H // 16, w_p=W // 16, p1=16, p2=16
        )
        # log_depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
        return log_depths, fx, fy, cx, cy


@torch.compile()
class FullSwiGLU(PP3DR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=FullSwiGLUHead)