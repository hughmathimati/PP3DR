from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.PP3DR_all_blocks import PP3DR_all_blocks
from models.PP3DR_pretrained_depth import PP3DR_pretrained_depth
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR_pretrained_depth,
        PP3DR_loss,
        name="pretrained_depth",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/dpt_all-blocks/PP3DR.pth",
        freeze_feature_extractor=False,
    )