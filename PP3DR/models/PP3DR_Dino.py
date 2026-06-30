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
# try block contains imports for calling from trainer, and except block contains imports for running this file itself
try:
    from .BidirectionalLaCT import GlobalLaCT, LocalLaCT
    from .pos_embed import RopePositionEmbedding, Rope3D
    from .Dinov3 import load_dinov3, obtain_features
except:
    from BidirectionalLaCT import GlobalLaCT, LocalLaCT
    from pos_embed import RopePositionEmbedding, Rope3D
    from Dinov3 import load_dinov3, obtain_features
from xformers.ops import SwiGLU


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = GlobalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)

    def forward(self, x: torch.Tensor, rope3d, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: [(B L) X dim]
        rope3d: 2 * [LHW, D_head]
        L

        Returns
        -------
        [B (L X) dim]
        """
        # First addition needs a rearrange, since x gets rearranged in the TTT.
        x = rearrange(x, "(B L) X dim -> B (L X) dim", L = L) + self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope3d, L))
        x = x + self.layer_scale_2(self.ffn(self.layer_norm_2(x)))
        return x


class LocalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = LocalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)

    def forward(self, x: torch.Tensor, rope, L) -> torch.Tensor:
        # First addition needs a rearrange, since x gets rearranged in the TTT.
        x = rearrange(x, "B (L X) dim -> (B L) X dim", L = L) + self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope, L))
        x = x + self.layer_scale_2(self.ffn(self.layer_norm_2(x)))
        return x


class PointHead(nn.Module):
    def __init__(
            self,
            dim,
            num_registers,
            num_heads = 20,
            blocks = 4,  # Pi3 has 5 transformer blocks per decoder
            ffn_ratio = 4,
            output_dim = 3,
    ):
        super().__init__()
        assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even for alternating global and frame-wise attention"
        self.blocks_each = blocks // 2
        self.dim = dim
        self.global_blocks = nn.ModuleList([GlobalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        self.local_blocks = nn.ModuleList([LocalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        self.layer_norm = nn.LayerNorm(dim) # Pre-projection layer norm.
        self.dim_proj = nn.Linear(dim, 16**2 * output_dim)
        self.num_registers = num_registers
        self.output_dim = output_dim
        self.num_blocks = blocks

    def forward(self, x: torch.Tensor, rope2d, rope3d, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        for i in range(self.blocks_each):
            if self.training:
                # Global attention: absorb frame-length into patch-length dimension.
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant = False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant = False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        x = rearrange(x, "(B L) X dim -> B L X dim", L=L)
        # x.shape == (B, L, num_registers + HW // 256, dim)
        # Drop the register tokens (recall that they're concatenated along the sequence dimension, not the embedding dimension).
        x = x[:, :, self.num_registers:, :]
        # x.shape == (B, L, HW // 256, dim)
        x = self.layer_norm(x)
        print(x.shape)
        # Project so we end up with the correct number of dimensions at the end
        x = self.dim_proj(x)
        print(x.shape)
        # x.shape == (B, L, HW // 256, 256 * output_dim)
        x = rearrange(x, "B L X (Y output_dim) -> B L (X Y) output_dim", output_dim = self.output_dim)
        # x.shape == (B, L, HW, output_dim)
        return x


class PoseHead(nn.Module):
    """
    Predict the relative camera pose per frame via the register (special) tokens.
    You might ask: If we use the register tokens for our output, do we need to add even more register tokens to serve
    the original purpose of the register tokens? The answer is: no, because now the register and "normal" tokens have
    swapped purposes!
    """
    def __init__(
            self,
            dim,
            num_registers,
            num_heads = 20,
            blocks = 4,  # Pi3 has 5 transformer blocks per decoder. We need to use more.
            ffn_ratio = 4,
            output_dim = 9, # (x, y, z) + 6D continuous rotation representation. This is per-frame.
    ):
        super().__init__()
        assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even for alternating global and frame-wise attention"
        self.blocks_each = blocks // 2
        self.dim = dim
        self.global_blocks = nn.ModuleList([GlobalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        self.local_blocks = nn.ModuleList([LocalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        self.layer_norm = nn.LayerNorm(num_registers * dim) # Pre-projection layer norm.
        self.dim_proj = nn.Linear(num_registers * dim, output_dim)
        self.num_registers = num_registers
        self.output_dim = output_dim
        self.num_blocks = blocks

    def forward(self, x: torch.Tensor, rope2d, rope3d, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        for i in range(self.blocks_each):
            if self.training:
                # Global attention: absorb frame-length into patch-length dimension.
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant = False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant = False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        # x.shape == (B * L, num_registers + HW // 256, dim)
        # Keep only the register tokens (recall that they're concatenated along the sequence dimension, not the embedding dimension).
        x = x[:, :self.num_registers, :]
        # x.shape == (B * L, num_registers, dim)
        # Flatten the register tokens (and separate the batch size and frame dimensions):
        x = rearrange(x, "(B L) reg dim -> B L (reg dim)", L=L)
        x = self.layer_norm(x)
        # Project so we end up with the correct number of dimensions at the end
        x = self.dim_proj(x)
        # x.shape == (B, L, output_dim)
        return x


def preprocess(image):
    """
    Crops the input image so its dimensions are evenly divisible by the patch size 16.

    Parameters
    ----------
    image

    Returns
    -------
    Cropped image, new H, new W
    """
    new_H, new_W = image.shape[-2] // 16, image.shape[-1] // 16
    return transforms.functional.center_crop(image, (new_H, new_W)), new_H, new_W

# @torch.compile(dynamic = True)
class PP3DR(nn.Module):
    def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 20,
            encoder_blocks: int = 24, # Pi3 is 36 ViT blocks
            decoder_blocks: int = 24, # Pi3 is 36 decoder blocks.
            ffn_ratio: int = 4,
            num_registers: int = 5,
            start_checkpointing = 6,
    ):
        super().__init__()
        assert decoder_blocks % 2 == 0, f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention"
        self.blocks_each = decoder_blocks // 2
        self.start_checkpointing = start_checkpointing

        # General decoder
        self.dim = dim
        self.global_blocks = nn.ModuleList([GlobalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        self.local_blocks = nn.ModuleList([LocalBlock(dim, num_heads, ffn_ratio)] * self.blocks_each)
        # We don't store num_registers here. It gets stored in the decoder heads.
        self.rope2d = RopePositionEmbedding(embed_dim = dim, num_heads = num_heads, device = "cuda")
        self.rope3d = Rope3D(embed_dim = dim, num_heads = num_heads, device = "cuda")

        self.processor, self.dino = load_dinov3()
        self.dino = self.dino.to("cuda").eval()

        # ViTTT puts in the registers for me.
        self.num_registers = num_registers

        # Per-task decoders
        """
        The point decoder will predict a normalised XY ray direction, along with inverse depth.
        """
        self.point_decoder = PointHead(dim, num_registers, num_heads)
        """
        The camera decoder will predict the relative SE3 transformation to the next frame.
        We are parameterizing our camera with three scalars for the translation and six scalars for the rotation.
        Details of how the rotation prediction works are in the forward() method.
        """
        self.pose_decoder = PoseHead(dim, num_registers, num_heads)

    def ViT(self, image):
        return obtain_features(self.processor, self.dino, image, remove_registers = False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Call preprocess() on the image first!

        Parameters
        ----------
        images: (batch size, RGB, height, width)

        Returns
        -------
        "XY_ray": Normalized XY ray direction for each pixel's point

        "inverse_depth": Inverse depth for each pixel's point (ReLU'ed to keep non-negative)

        "relative_camera_translation": (x, y, z) translation between this camera pose and the next one

        "relative_camera_rotation": quaternion rotation between this camera pose and the next one
        """
        B, L, C, H, W = x.shape

        # Step 1: ViT
        # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into ViTTT.
        x = self.ViT(x.flatten(0, 1))
        # x.shape == (B * L, num_registers + HW // 256, dim)
        """
        We will now perform alternating-attention, starting with global attention and ending with frame-wise attention.
        """
        rope2d, rope3d = self.rope2d(H // 16, W // 16), self.rope3d(L, H // 16, W // 16)
        for i in range(self.blocks_each):
            if self.training and i >= self.start_checkpointing:
                # Global attention: absorb frame-length into patch-length dimension.
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant = False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant = False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        # x.shape == (B * L, num_registers + HW // 256, dim)
        # Step 2: Per-task decoders
        points = self.point_decoder(x, rope2d, rope3d, L) # (B, L, HW, 3)
        poses = self.pose_decoder(x, rope2d, rope3d, L) # (B, L, 9)

        # Get the rotation matrix by orthogonalizing the first two 3D vectors, then taking the cross product for the third.
        rotation = torch.empty(B, L, 3, 3) # going to take the transpose at the end
        a = F.normalize(poses[:, :, 3:6], dim = -1)
        rotation[:, :, 0] = a
        b = poses[:, :, 6:]
        # Unsqueeze after the dot product so the resulting scalars broadcast correctly against the vector a.
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim = -1).unsqueeze(-1) * a, dim = -1)
        rotation[:, :, 1] = b
        rotation[:, :, 2] = torch.linalg.cross(a, b, dim = -1)

        return {
            "XY_ray": F.normalize(points[...,:2], dim = -1), # Normalize the XY ray direction
            "inverse_depth": F.relu(points[...,2]), # ReLU the inverse depth to keep it positive
            "relative_camera_translation": poses[..., :3], # 3 scalars, (x, y, z)
            "relative_camera_rotation": rotation.transpose(-1, -2),
        }


if __name__ == "__main__":
    import time
    from transformers.image_utils import load_image
    import torchvision.transforms.v2 as transforms
    model = PP3DR().to("cuda").eval() # You should really rename this
    # (batch size, num frames, # channels, image height, image width)
    fake_input = torch.randn(2, 4, 3, 200, 400).to("cuda")
    output = model(fake_input)
    for key in output:
        print(key)
        print(output[key].shape)