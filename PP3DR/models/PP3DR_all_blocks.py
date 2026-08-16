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
from xformers.ops import SwiGLU
from timm.layers import DropPath

from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT
from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
from models.PP3DR import PointwiseSwiGLU, UpscaleBlock, FocalHead, PP3DR
from models.ViTTT import ViTTT


class DecomposedSwiGLU(nn.Module):
    def __init__(self, num_tensors=30, input_dim=1280, hidden_dim=4 * 1280, out_dim=1280):
        super().__init__()

        # Instead of one massive 38400 x hidden_dim matrix, we create a ModuleList
        # of 30 smaller linear layers for the gate and the up projection.
        self.gate_projs = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_tensors)])
        self.up_projs = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_tensors)])

        # We only need a single set of biases for the accumulated result
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.up_bias = nn.Parameter(torch.zeros(hidden_dim))

        # The final down projection remains identical
        self.down_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, tensor_list):
        # 1. Initialize the accumulators (requires very little memory)
        # Using the first tensor's projection directly to avoid a zero-tensor allocation
        H_gate = self.gate_projs[0](tensor_list[0])
        H_up = self.up_projs[0](tensor_list[0])

        # 2. Iteratively accumulate the projections of the remaining 29 tensors
        for i in range(1, len(tensor_list)):
            # Using out-of-place addition so autograd can still track gradients properly
            H_gate = H_gate + self.gate_projs[i](tensor_list[i])
            H_up = H_up + self.up_projs[i](tensor_list[i])

        # Divide by len(tensor_list) to help keep the variance under control
        H_gate = (H_gate + self.gate_bias) / len(tensor_list)
        H_up = (H_up + self.up_bias) / len(tensor_list)

        # 3. Apply the non-linearity to the completely aggregated context
        H_mid = F.silu(H_gate) * H_up

        # 4. Project back down to 1280
        return self.down_proj(H_mid)

class DepthHead(nn.Module):
    def __init__(self, dim=1280, ffn_ratio=4):
        super().__init__()
        total_num_blocks = 30
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(total_num_blocks)])
        self.projection = DecomposedSwiGLU(total_num_blocks, dim, ffn_ratio * dim, dim)
        self.upscale = nn.Sequential(
            UpscaleBlock(dim, dim // 16, 4),
            nn.SiLU(inplace=True),
            UpscaleBlock(dim // 16, dim // 256, 4),
            nn.SiLU(inplace=True),
        )
        self.final_proj = PointwiseSwiGLU(in_features=dim // 256, hidden_features=dim // 256, out_features=2)

    def forward(self, x, L, H, W):
        """
        Parameters
        ----------
        (12 + 18) x (B * L, H_p * W_p, dim)

        Returns
        -------
        (B, L, H, W)
        """
        x = [self.layer_norms[i](y) for i, y in enumerate(x)]
        x = self.projection(x) # (B * L, H_p * W_p, dim)
        x = rearrange(x, "BL (H_p W_p) dim -> BL dim H_p W_p", H_p=H // 16, W_p=W // 16)
        x = self.upscale(x)
        x = self.final_proj(x)
        return rearrange(x, "(B L) two H W -> B L two H W", L=L).unbind(2)

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
        self.depth_head = DepthHead()

    def forward(self, x: torch.Tensor, L, H, W) -> torch.Tensor:
        """
        Inputs consist of:
        The outputs of all 12 ViTTT blocks + the outputs of all 18 PP3DR LocalBlocks. That's 30 feature maps total.
        Each feature map has shape (B * L, X, dim)

        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        # 4 x (B * L, HW // 256, dim)
        tokens = [y[:, self.num_registers:, :] for y in x]
        return self.focal_head(x[-1][:, :self.num_registers, :], L, H, W), *self.depth_head(tokens, L, H, W)


@torch.compile()
class PP3DR_all_blocks(PP3DR):
    """
    NOTE: You'll have to decrease start_checkpointing. You'll also have to pass in range(num_blocks) for ViTTT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, point_head_class=PointHead, output_blocks=list(range(18)))