import torch
from models.PP3DR import PP3DR
from models.Dinov3 import load_dinov3, obtain_features

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

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def ViT(self, image):
        return obtain_features(self.processor, self.dino, image, remove_registers = False)