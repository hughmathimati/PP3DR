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
from models.PP3DR import PP3DR, GlobalBlock, LocalBlock, PoseHead, FocalHead
from models.ViTTT import ViTTT
from models.vggt_omega_depth_head import DenseHead


class PointHead(nn.Module):
    def __init__(
            self,
            dim=1280,
            num_registers=5,
    ):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.focal_head = FocalHead(dim=dim)
        self.depth_head = DenseHead(dim_in=2 * dim)

    def forward(
            self, x, outputs: torch.Tensor, L, H, W, original_height, original_width
    ) -> torch.Tensor:
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
        four_dim = [rearrange(y, "(B L) X dim -> B L X dim", L=L) for y in outputs]
        focal = self.focal_head(x[:, :self.num_registers, :], L, H, W)
        log_depths, raw_depth_confidence = self.depth_head(
            four_dim, four_dim[0].shape[0], L,
            H // 16, W // 16,
            original_height, original_width,
            self.num_registers
        )
        return focal, log_depths, raw_depth_confidence


@torch.compile()
class PP3DR_DenseHead(PP3DR):
    def __init__(
            self,
            dim: int = 1024,
            num_heads: int = 16,
            # VGGT-Omega uses 24 encoder blocks and 48 decoder blocks.
            encoder_blocks: int = 16,
            decoder_blocks: int = 36,
            ffn_ratio: int = 4,
            num_registers: int = 5,
            # How many blocks EACH not to checkpoint (total # is twice as many).
            start_checkpointing=16,
            point_head_class=PointHead,
            pose_head_class=PoseHead,
            # These are block_each indices.
            output_blocks=None
    ):
        # super().__init__(*args, **kwargs, point_head_class=PointHead)
        nn.Module.__init__(self)
        assert decoder_blocks % 2 == 0, \
            f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention."
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."

        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.blocks_each = decoder_blocks // 2
        self.start_checkpointing = start_checkpointing

        # Drop rates
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

        # Don't checkpoint the ViTTT blocks. Checkpointing the PP3DR blocks saves more time.
        self.ViTTT = ViTTT(
            dim,
            num_heads,
            encoder_blocks,
            ffn_ratio,
            drop_rates=ViTTT_drop_rates,
            start_checkpointing=0,
            final_layer_norm=False,
            # For 16 blocks, this is [3, 15].
            # output_blocks=[encoder_blocks // 4 - 1, encoder_blocks - 1]
            output_blocks = [3, 4, 14, 15]
        )
        self.num_registers = num_registers

        # Per-task decoders
        self.point_head = point_head_class(dim=dim)
        """
        The camera decoder will predict the relative SE3 transformation to the next frame.
        We are parameterizing our camera with three scalars for the translation and six scalars for the rotation.
        Details of how the rotation prediction works are in the forward() method.
        """
        self.pose_head = pose_head_class(dim=dim)

        # For 18 blocks each (36 blocks total), this is [8, 17].
        self.output_blocks = [self.blocks_each // 2 - 1,
                              self.blocks_each - 1] if output_blocks is None else output_blocks

    def forward(self, x: torch.Tensor, rope_x, rope_y, original_height, original_width) -> dict:
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
        x, outputs = self.ViTTT(x.flatten(0, 1))
        # x.shape == (B * L, num_registers + HW // 256, dim)
        """
        NOTE:
        We're doing a funny thing here where I want to be able to just import my ViTTT code but I also want to
        concatenate ViTTT feature maps to give the right feature dimension for the head.
        What I'm going to do is have ViTTT return the feature maps I want to concatenate, then concatenate them myself.
        Currently, I'm having ViTTT return four feature maps. The first two will be concatenated, and so will the last two.
        """
        concatenated_outputs = [torch.cat(outputs[:2], dim=-1), torch.cat(outputs[2:], dim=-1)]
        outputs = concatenated_outputs
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
                global_output = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], global_output, rope2d, L, use_reentrant=False)
            else:
                global_output = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](global_output, rope2d, L)
            if i in self.output_blocks:
                outputs.append(torch.cat([x, global_output.view(B * L, -1, self.dim)], dim=-1))
        # x.shape == (B * L, num_registers + HW // 256, 2 * dim)

        # Step 2: Per-task decoders
        """
        If you're observant, you'll notice that every tensor added to `outputs` was the output of a LocalBlock.
        This is fine, because we're doing dense depth prediction. This would be harder to justify for the PoseHead.
        """
        focal_length, log_depths, raw_uncertainty = self.point_head(
            x, outputs, L, H, W, original_height, original_width
        )
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