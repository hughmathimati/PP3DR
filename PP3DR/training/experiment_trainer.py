from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.ZipMap_backbone import ZipMapPP3DRAblation
from models.VGGT_Omega_full import VGGTOmegaAblation
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="PP3DR-new-design",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/16-36_7-15-8-17_new-losses/PP3DR.pth",
        # checkpoint="/vulcanscratch/hughma/PP3DR/16-36_residual-ls/epoch 51"
        # strict=False
        batch_size=1
    )