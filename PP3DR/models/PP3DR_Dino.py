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
    from .BidirectionalLaCT import GlobalLaCT, LocalLaCT, BidirectionalLaCT
    from .pos_embed import RopePositionEmbedding, Rope3D
    from .Dinov3 import load_dinov3, obtain_features
    from .PP3DR import PP3DR
except:
    from BidirectionalLaCT import GlobalLaCT, LocalLaCT, BidirectionalLaCT
    from pos_embed import RopePositionEmbedding, Rope3D
    from Dinov3 import load_dinov3, obtain_features
    from PP3DR import PP3DR
from xformers.ops import SwiGLU


@torch.compile()
class PP3DR_Dino(PP3DR):
    """
    PP3DR Baseline, which uses Dinov3 features instead of the 12-block ViTTT feature extractor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor, self.dino = load_dinov3()
        self.dino = self.dino.to("cuda").eval()
        for parameter in self.dino.parameters():
            parameter.requires_grad = False

    def ViT(self, image):
        return obtain_features(self.processor, self.dino, image, remove_registers = False)


if __name__ == "__main__":
    model = PP3DR_Dino().to('cuda')
    print(model.dino.parameters())

    # import time
    # from transformers.image_utils import load_image
    # import torchvision.transforms.v2 as transforms
    # model = PP3DR_Dino().to("cuda").eval() # You should really rename this
    # (batch size, num frames, # channels, image height, image width)
    # fake_input = torch.randn(2, 4, 3, 200, 400).to("cuda")
    # output = model(fake_input)
    # for key in output:
    #     print(key)
    #     print(output[key].shape)