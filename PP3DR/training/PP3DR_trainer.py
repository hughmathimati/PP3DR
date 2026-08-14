from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="dpt_1-4-11_17",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/dpt-finetune/PP3DR.pth",
        # freeze_feature_extractor=False,
    )