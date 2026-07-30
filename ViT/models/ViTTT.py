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
    from .Dinov3 import low_rank, write_to_image
except:
    from BidirectionalLaCT import BidirectionalLaCT
    from pos_embed import RopePositionEmbedding
    from Dinov3 import low_rank, write_to_image
from xformers.ops import SwiGLU
from timm.layers import DropPath

class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

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
            blocks = 24, # DINOv3 H+ has 32 layers, but I'm using more to compensate for the weaker performance of TTT.
            ffn_ratio = 4,
            num_registers = 5,
            start_checkpointing = 6
    ):
        super().__init__()
        self.dim = dim
        self.rope = RopePositionEmbedding(dim, num_heads=num_heads)
        self.patch_conv = nn.Conv2d(3, dim, kernel_size=(16, 16), stride=(16, 16))

        drop_rates = [x.item() for x in torch.linspace(0, 0.1, blocks)]
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ffn_ratio, drop_rates[i]) for i in range(blocks)
        ])

        self.class_and_registers = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.start_checkpointing = start_checkpointing
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
        x = (x - self.mean) / self.std
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
        for i, block in enumerate(self.blocks):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(block, x, rope, use_reentrant = False)
            else:
                x = block(x, rope)
        return self.final_layer_norm(x)

if __name__ == "__main__":
    import time
    from transformers.image_utils import load_image
    import torchvision.transforms.v2 as transforms
    torch.set_float32_matmul_precision('high')
    import torchvision

    model = ViTTT(blocks = 12).to("cuda")
    print("loading state dict...")
    loaded_state_dict = torch.load("/vulcanscratch/hughma/ViT/12_blocks_finetune/ViTTT.pth", weights_only=True)

    # The lines here are necessary if all the loaded state dict entries begin with an extra "module."
    new_state_dict = {}
    for key in loaded_state_dict:
        new_state_dict[key[7:]] = loaded_state_dict[key]
    model.load_state_dict(new_state_dict)

    model.eval()
    image1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
    image2 = transforms.functional.to_dtype(torchvision.io.decode_image("/vulcanscratch/hughma/data/sintel/training/final/alley_1/frame_0001.png"), torch.float32, scale=True)
    # image = image.unsqueeze(0)
    # print(f"original shape: {image.size}")
    # image = image.crop((0, 0, image.size[0] // 16 * 16, image.size[1] // 16 * 16))
    image_to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    # Unsqueeze for the batch dimension.
    image1 = image_to_tensor(image1).unsqueeze(0)
    image2 = image_to_tensor(image2).unsqueeze(0)
    # print(f"resized shape: {image.shape}")
    with torch.no_grad():
        start = time.time()
        features1 = model(image1.to("cuda", non_blocking = True))
        features2 = model(image2.to("cuda", non_blocking = True))
        end = time.time()
    print("Elapsed time (slight overestimation):", end - start)
    features1, features2 = features1[:,5:], features2[:,5:]
    print(features1.shape, features2.shape)
    features1, features2 = low_rank(features1), low_rank(features2)
    print(f"lowrank: {features1.shape}, {features2.shape}")
    write_to_image(features1, 42, 60, name = "images/ViTTT_cat_12_blocks_finetune best_checkpoint.png")
    write_to_image(features2, 27, 64, name = "images/ViTTT_sintel_12_blocks_finetune best_checkpoint.png")
