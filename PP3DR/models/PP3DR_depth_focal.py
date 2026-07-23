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
    from .pos_embed import RopePositionEmbedding, Rope3D, Rope2D
    from .ViTTT import ViTTT
except:
    from BidirectionalLaCT import GlobalLaCT, LocalLaCT
    from pos_embed import RopePositionEmbedding, Rope3D, Rope2D
    from ViTTT import ViTTT
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
class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path = 0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = GlobalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

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
        x = rearrange(x, "(B L) X dim -> B (L X) dim", L = L) + self.drop_path(
            self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope3d, L))
        )
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
        return x

# @torch.compile()
class LocalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path = 0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = LocalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope, L) -> torch.Tensor:
        # First addition needs a rearrange, since x gets rearranged in the TTT.
        x = rearrange(x, "B (L X) dim -> (B L) X dim", L = L) + self.drop_path(
            self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope, L))
        )
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
        return x

@torch.compile()
class DepthFocalHead(nn.Module):
    """
    Predicts depths for all points + fx, fy, cx, cy per frame.
    """
    def __init__(
            self,
            dim,
            num_registers,
            num_heads = 20,
            blocks = 4,  # Pi3 has 5 transformer blocks per decoder
            ffn_ratio = 4,
    ):
        super().__init__()
        assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even for alternating global and frame-wise attention"
        self.blocks_each = blocks // 2
        self.dim = dim

        self.global_blocks = nn.ModuleList([GlobalBlock(dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
        self.local_blocks = nn.ModuleList([LocalBlock(dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])

        self.registers_layer_norm = nn.LayerNorm(dim) # Pre-projection layer norm.
        self.tokens_layer_norm = nn.LayerNorm(dim) # Pre-projection layer norm.
        self.focal_proj = nn.Linear(num_registers * dim, 4, bias=False) # fx, fy, cx, cy
        self.depths_proj = nn.Linear(dim, 256, bias=False)
        self.num_registers = num_registers
        self.num_blocks = blocks

    def forward(self, x: torch.Tensor, rope2d, rope3d, L, H, W) -> torch.Tensor:
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

        x = rearrange(x, "(B L) X dim -> B L X dim", L=L) # (B, L, num_registers + HW // 256, dim)
        registers = self.registers_layer_norm(x[:, :, :self.num_registers, :]) # (B, L, num_registers, dim)
        tokens = self.tokens_layer_norm(x[:, :, self.num_registers:, :]) # (B, L, HW // 256, dim)
        # (B, L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)
        # fx and fy will be passed into softplus to ensure they're always positive.
        fx, fy = F.softplus(fx), F.softplus(fy)
        # cx and cy will actually be the residuals to the image center point.
        cx, cy = cx + W / 2, cy + H / 2

        # We must take into account that each token represents the 16x16 patch at that location.
        # Naively rearranging leads to each 16x16 patch being mapped to a contiguous line of 256 pixels.
        depths = rearrange(
            self.depths_proj(tokens), # (B, L, HW // 256, 256)
            "B L (h_p w_p) (p1 p2) -> B L (h_p p1) (w_p p2)",
            h_p=H // 16, w_p=W // 16, p1=16, p2=16
        )
        # depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
        return depths, fx, fy, cx, cy

@torch.compile()
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
        self.global_blocks = nn.ModuleList([GlobalBlock(dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
        self.local_blocks = nn.ModuleList([LocalBlock(dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
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


@torch.compile()
class PP3DR_depth_focal(nn.Module):
    def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 20,
            encoder_blocks: int = 12, # Pi3 is 36 ViT blocks
            decoder_blocks: int = 36, # Pi3 is 36 decoder blocks.
            ffn_ratio: int = 4,
            num_registers: int = 5,
            start_checkpointing = 7,
            freeze_feature_extractor = True,
            PP3DR_drop_rates = None,
            ViTTT_drop_rates = None
    ):
        super().__init__()
        assert decoder_blocks % 2 == 0, f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention"
        self.blocks_each = decoder_blocks // 2
        self.start_checkpointing = start_checkpointing

        # General decoder
        self.dim = dim
        drop_rates = [
            x.item() for x in
            (torch.linspace(0, 0.1, decoder_blocks) if PP3DR_drop_rates is None else PP3DR_drop_rates)
        ]
        self.global_blocks = nn.ModuleList([
            GlobalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i])
            for i in range(self.blocks_each)
        ])
        self.local_blocks = nn.ModuleList([
            LocalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i + 1])
            for i in range(self.blocks_each)
        ])

        # We don't store num_registers here. It gets stored in the decoder heads.
        self.rope2d = Rope2D(embed_dim = dim, num_heads = num_heads, device = "cuda")
        self.rope3d = Rope3D(embed_dim = dim, num_heads = num_heads, device = "cuda")

        self.ViTTT = ViTTT(dim, num_heads, encoder_blocks, ffn_ratio, drop_rates=ViTTT_drop_rates)  # Import ViTTT!
        self.ViTTT.load_state_dict(torch.load("/vulcanscratch/hughma/ViT/best_checkpoint.pth",
                                              weights_only=True,
                                              map_location="cpu"))
        self.freeze_feature_extractor = freeze_feature_extractor
        if self.freeze_feature_extractor:
            for parameter in self.ViTTT.parameters():
                parameter.requires_grad = False
            self.ViTTT.eval()
        # ViTTT puts in the registers for me.
        self.num_registers = num_registers

        # Per-task decoders
        self.depth_focal_decoder = DepthFocalHead(dim, num_registers, num_heads)
        self.pose_decoder = PoseHead(dim, num_registers, num_heads)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_feature_extractor:
            self.ViTTT.eval()
        return self

    def ViT(self, image):
        return self.ViTTT(image)

    def forward(self, x: torch.Tensor, rope_x, rope_y) -> dict:
        """
        The input must be pre-processed so its height and width are multiples of the patch size.

        rope_x and rope_y come from the dataloader.

        Parameters
        ----------
        images: (batch size, RGB, height, width)
        rope_x: The 2D x-coordinates to be fed into RoPE (B, L, HW)
        rope_y: The 2D y-coordinates to be fed into RoPE; same shape as above.

        Returns
        -------
        "XY_ray": Normalized XY ray direction for each pixel's point

        "log_depth": Log depth for each pixel's point

        "relative_camera_translation": (x, y, z) translation between this camera pose and the next one

        "relative_camera_rotation": quaternion rotation between this camera pose and the next one
        """
        B, L, C, H, W = x.shape

        # Step 1: ViT
        # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into Dino.
        x = self.ViT(x.flatten(0, 1))
        # x.shape == (B * L, num_registers + HW // 256, dim)

        """
        We will now perform alternating-attention, starting with global attention and ending with frame-wise attention.
        
        The RoPE coordinates have been pre-calculated and are provided by the dataloader. This is because they're
        dependent on the dimensions of the raw images.
        
        rope2d has shape (B * L, HW, head_dim), and rope3d has shape (B, L, HW, head_dim).
        """
        rope2d, rope3d = self.rope2d(rope_x, rope_y), self.rope3d(rope_x, rope_y)
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
        depths, fx, fy, cx, cy = self.depth_focal_decoder(x, rope2d, rope3d, L, H, W)
        # We're predicting the relative pose from this frame to the next one, which is why our sequence length is L - 1.
        poses = self.pose_decoder(x, rope2d, rope3d, L)[:, :-1] # (B, L - 1, 9)

        # Get the rotation matrix by orthogonalizing the first two 3D vectors, then taking the cross product for the third.
        # We're going to construct the actual columns as rows of the current matrix, then take the transpose at the end.
        # eps is increased from 1e-12 to 1e-6 in an attempt to increase stability.
        a = F.normalize(poses[:, :, 3:6], dim = -1, eps=1e-6)
        b = poses[:, :, 6:]
        # Unsqueeze after the dot product so the resulting scalars broadcast correctly against the vector a.
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim = -1).unsqueeze(-1) * a, dim = -1, eps=1e-6)
        c = torch.linalg.cross(a, b, dim = -1)

        return {
            # e^-80 to e^80 is safely within the range of bfloat16.
            "log_depths": torch.clamp(depths, min=-80, max=80),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "relative_camera_translations": poses[:, :, :3], # (B, L - 1, 3) -> 3 scalars, (x, y, z)
            "relative_camera_rotations": torch.stack([a, b, c], dim=-1) # (B, L - 1, 3, 3)
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