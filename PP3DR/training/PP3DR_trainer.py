from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.PP3DR_pretrained_depth import PP3DR_DenseHead
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="2-11_8-17_resconv_pos-embed",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/2-11_8-17_pos-embed/PP3DR.pth",
        # strict=False,
        freeze_feature_extractor=False,
    )