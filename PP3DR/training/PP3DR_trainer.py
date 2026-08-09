from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="dino",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/double-upscale/PP3DR.pth",
        # freeze_feature_extractor=False,
        use_muon=False,
    )