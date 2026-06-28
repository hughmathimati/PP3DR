import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
# try block contains imports for calling from trainer, and except block contains imports for running this file itself
try:
    from .BidirectionalLaCT import BidirectionalLaCT
    from .pos_embed import RopePositionEmbedding
    # from .Dinov3 import low_rank, write_to_image
except:
    from BidirectionalLaCT import BidirectionalLaCT
    from pos_embed import RopePositionEmbedding
    # from Dinov3 import low_rank, write_to_image
from xformers.ops import SwiGLU

class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = BidirectionalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features = dim, hidden_features = dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        x = x + self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope))
        x = x + self.layer_scale_2(self.ffn(self.layer_norm_2(x)))
        return x

@torch.compile(dynamic = True) # If you compile this module, comment the assert first.
class ViTTT(nn.Module):
    def __init__(
            self,
            dim: int = 1280,
            num_heads: int = 20,
            blocks = 48, # DINOv3 H+ has 32 layers, but I'm using more to compensate for the weaker performance of TTT.
            ffn_ratio = 4,
    ):
        super().__init__()
        self.dim = dim
        self.rope = RopePositionEmbedding(dim, num_heads=num_heads)
        self.patch_conv = nn.Conv2d(3, dim, kernel_size=(16, 16), stride=(16, 16))
        self.blocks = nn.ModuleList([Block(dim, num_heads, ffn_ratio)] * blocks)
        self.class_and_registers = nn.Parameter(torch.randn(5, dim) * 0.02)
        self.final_layer_norm = nn.LayerNorm(dim)

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
        for block in self.blocks:
            x = block(x, rope)
        return self.final_layer_norm(x)


if __name__ == "__main__":
    print(ViTTT())
    # import time
    # from transformers.image_utils import load_image
    # import torchvision.transforms.v2 as transforms
    #
    # model = ViTTT()
    # print("loading state dict...")
    # model.load_state_dict(torch.load("/vulcanscratch/hughma/ViT/Failed attempts/Flawed ViTTT train/ViTTT.pth", weights_only=True, map_location={"cuda:1": "cuda:0"},))
    # model.eval()
    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
    # print(f"original shape: {image.size}")
    # image = image.crop((0, 0, image.size[0] // 16 * 16, image.size[1] // 16 * 16))
    # # Unsqueeze for the batch dimension.
    # image_to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    # image = image_to_tensor(image).unsqueeze(0)
    # print(f"resized shape: {image.size}")
    # with torch.no_grad():
    #     start = time.time()
    #     features = model(image)
    #     end = time.time()
    # print("Elapsed time (slight overestimation):", end - start)
    # features = features[:,5:]
    # print(features.shape)
    # features = low_rank(features)
    # print(f"lowrank: {features.shape}")
    # write_to_image(features, 42, 60, name = "ViTTT_image.png")