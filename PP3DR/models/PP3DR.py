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
from models.ViTTT import ViTTT


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path=0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = GlobalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features=dim, hidden_features=dim * ffn_ratio)
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
        x = rearrange(x, "(B L) X dim -> B (L X) dim", L=L) + self.drop_path(
            self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope3d, L))
        )
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
        return x

class LocalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path=0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = LocalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features=dim, hidden_features=dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope, L) -> torch.Tensor:
        # First addition needs a rearrange, since x gets rearranged in the TTT.
        x = rearrange(x, "B (L X) dim -> (B L) X dim", L=L) + self.drop_path(
            self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope, L))
        )
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
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
            dim=1280,
            num_registers=5,
            output_dim=9,  # (x, y, z) + 6D continuous rotation representation. This is per-frame.
    ):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(num_registers * dim)  # Pre-projection layer norm.
        self.dim_proj = nn.Linear(num_registers * dim, output_dim)
        self.num_registers = num_registers
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
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
    def __init__(self, dim=1280, num_registers=5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.focal_proj = nn.Linear(num_registers * dim, 1)

    def forward(self, x, L, H, W):
        """
        Parameters
        ----------
        (B * L, num_registers, dim)

        Returns
        -------
        (B,)
        """
        registers = self.layer_norm(x)
        f_mult = self.focal_proj(registers.flatten(start_dim=-2)) # (B * L, 1)
        f_mult = rearrange(f_mult.squeeze(-1), "(B L) -> B L", L=L).mean(dim=-1) # (B,)
        f_mult = F.softplus(f_mult)
        return f_mult * max(H, W)

class DepthHead(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.initial_proj = nn.ModuleList([
            UpscaleBlock(dim, dim // 64, 8), # 1/2
            UpscaleBlock(dim, dim // 16, 4), # 1/4
            UpscaleBlock(dim, dim // 4, 2), # 1/8
            nn.Identity() # 1/16
        ])
        self.fusion_proj = nn.ModuleList([
            UpscaleBlock(dim, dim // 4, 2), # 1/8
            UpscaleBlock(dim // 4, dim // 16, 2), # 1/4
            UpscaleBlock(dim // 16, dim // 64, 2), # 1/2
        ])
        self.final_upscale = UpscaleBlock(dim // 64, dim // 256, 2) # original dimensions
        self.final_proj = PointwiseSwiGLU(in_features=dim // 256, hidden_features=dim // 256, out_features=2)

    def forward(self, x, L):
        """
        Parameters
        ----------
        4 x (B * L, dim, H // patch_size, W // patch_size)

        Returns
        -------
        log_depths (B, L, H, W)
        raw_uncertainty (B, L, H, W)
        """
        a, b, c, d = [self.initial_proj[i](y) for i, y in enumerate(x)]
        x = F.silu(c + self.fusion_proj[0](d))
        x = F.silu(b + self.fusion_proj[1](x))
        x = F.silu(a + self.fusion_proj[2](x))
        x = F.silu(self.final_upscale(x)) # (B * L, 2, H, W)
        return rearrange(self.final_proj(x), "(B L) two H W -> B L two H W", L=L).unbind(2)

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
        # These LayerNorms are for the depth head only. The FocalHead has its own layer norm.
        self.token_layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.depth_head = DepthHead()

    def forward(self, x: torch.Tensor, L, H, W) -> torch.Tensor:
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
        # 4 x (B * L, HW // 256, dim)
        tokens = [self.token_layer_norms[i](y[:, self.num_registers:, :]) for i, y in enumerate(x)]
        tokens = [rearrange(y, "BL (H_p W_p) dim -> BL dim H_p W_p", H_p=H // 16) for y in tokens]

        return self.focal_head(x[-1][:, :self.num_registers, :], L, H, W), *self.depth_head(tokens, L)


@torch.compile()
class PP3DR(nn.Module):
    def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 20,
            encoder_blocks: int = 12,  # Pi3 is 36 ViT blocks
            decoder_blocks: int = 36,  # Pi3 is 36 decoder blocks.
            ffn_ratio: int = 4,
            num_registers: int = 5,
            start_checkpointing=11,  # How many blocks EACH not to checkpoint (total # is twice as many).
            freeze_feature_extractor=True,
            point_head_class=PointHead,
            pose_head_class=PoseHead,
            output_blocks = [17] # These are block_each indices. The highest block_each index is 17.
    ):
        super().__init__()
        assert decoder_blocks % 2 == 0, f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention"
        self.decoder_blocks = decoder_blocks
        self.blocks_each = decoder_blocks // 2
        # If we're unfreezing ViTTT, we still want to maximise VRAM usage.
        # To keep VRAM around 99% with the same batch size as freeze_feature_extractor=True, we're going to checkpoint
        # *none* of the ViTTT blocks and 6 of the PP3DR blocks.
        # Keep in mind that this varies wildly with the specific architecture we're using, so you should check each time.
        self.start_checkpointing = start_checkpointing if freeze_feature_extractor else 4

        # Drop rates
        if freeze_feature_extractor:
            ViTTT_drop_rates = None
            PP3DR_drop_rates = torch.linspace(0, 0.1, decoder_blocks)
        else:
            rates = torch.linspace(0, 0.1, encoder_blocks + decoder_blocks)
            ViTTT_drop_rates = rates[:encoder_blocks]
            PP3DR_drop_rates = rates[encoder_blocks:]

        # General decoder
        self.dim = dim
        drop_rates = [x.item() for x in PP3DR_drop_rates]
        self.global_blocks = nn.ModuleList([
            GlobalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i])
            for i in range(self.blocks_each)
        ])
        self.local_blocks = nn.ModuleList([
            LocalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i + 1])
            for i in range(self.blocks_each)
        ])

        # We don't store num_registers here. It gets stored in the decoder heads.
        self.rope2d = Rope2D(embed_dim=dim, num_heads=num_heads, device="cuda")
        self.rope3d = Rope3D(embed_dim=dim, num_heads=num_heads, device="cuda")

        self.ViTTT = ViTTT(dim, num_heads, encoder_blocks, ffn_ratio, drop_rates=ViTTT_drop_rates)
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
        self.point_head = point_head_class()
        """
        The camera decoder will predict the relative SE3 transformation to the next frame.
        We are parameterizing our camera with three scalars for the translation and six scalars for the rotation.
        Details of how the rotation prediction works are in the forward() method.
        """
        self.pose_head = pose_head_class()

        self.output_blocks = output_blocks

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
        "log_depth": Log depth for each pixel's point (B, L, H, W)

        "focal_length": Unified focal length per sequence (B,)

        "relative_camera_translation": (x, y, z) translation between this camera pose and the next one (B, L - 1, 3)

        "relative_camera_rotation": 3x3 rotation matrix between this camera pose and the next one (B, L - 1, 3, 3)
        """
        B, L, C, H, W = x.shape

        # Step 1: ViT
        # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into Dino.
        x, outputs = self.ViT(x.flatten(0, 1))
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
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)
            if i in self.output_blocks:
                outputs.append(x)
        # x.shape == (B * L, num_registers + HW // 256, 2 * dim)

        # Step 2: Per-task decoders
        """
        If you're observant, you'll notice that every tensor added to `outputs` was the output of a LocalBlock.
        This is fine, because we're doing dense depth prediction. This would be harder to justify for the PoseHead.
        """
        focal_length, log_depths, raw_uncertainty = self.point_head(outputs, L, H, W)
        # We're predicting the relative pose from this frame to the next one, which is why our sequence length is L - 1.
        poses = self.pose_head(x, L)[:, :-1]  # (B, L - 1, 9)

        # Get the rotation matrix by orthogonalizing the first two 3D vectors, then taking the cross product for the third.
        # We're going to construct the actual columns as rows of the current matrix, then take the transpose at the end.
        # eps is increased from 1e-12 to 1e-6 in an attempt to increase stability.
        a = F.normalize(poses[:, :, 3:6], dim=-1, eps=1e-6)
        b = poses[:, :, 6:]
        # Unsqueeze after the dot product so the resulting scalars broadcast correctly against the vector a.
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim=-1).unsqueeze(-1) * a, dim=-1, eps=1e-6)
        c = torch.linalg.cross(a, b, dim=-1)

        return {
            # e^-80 to e^80 is safely within the range of bfloat16.
            "log_depths": torch.clamp(log_depths, min=-80, max=80).to(torch.float32),
            "raw_uncertainty": torch.clamp(raw_uncertainty, min=-80, max=80).to(torch.float32),
            "focal_length": focal_length.to(torch.float32),
            "relative_camera_translations": poses[:, :, :3].to(torch.float32),  # (B, L - 1, 3) -> 3 scalars, (x, y, z)
            "relative_camera_rotations": torch.stack([a, b, c], dim=-1).to(torch.float32)  # (B, L - 1, 3, 3)
        }


if __name__ == "__main__":
    import time
    from transformers.image_utils import load_image
    import torchvision.transforms.v2 as transforms
    from Dinov3 import low_rank, write_to_image
    from datasets.nrgbd_dataset import nrgbd_dataset

    model = PP3DR().to("cuda").eval()  # You should really rename this
    loaded_state_dict = torch.load("/vulcanscratch/hughma/PP3DR/finetune/PP3DR.pth", weights_only=True)
    # The lines here are necessary if all the loaded state dict entries begin with an extra "module."
    new_state_dict = {}
    for key in loaded_state_dict:
        new_state_dict[key[7:]] = loaded_state_dict[key]
    model.load_state_dict(new_state_dict)
    print("loaded state dict")

    dataset = nrgbd_dataset()
    data = dataset[0]
    for key in data:
        data[key] = data[key].unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = model(data['images'].to("cuda"), data['rope_x'].to("cuda"), data['rope_y'].to("cuda"))
    print("Done.")