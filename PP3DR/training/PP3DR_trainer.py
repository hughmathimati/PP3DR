from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.PP3DR_pretrained_depth import PP3DR_DenseHead
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="16-36_7-15-8-17_new-losses",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/16-36_dpt-simpler-weighting/PP3DR.pth",
        # checkpoint="/vulcanscratch/hughma/PP3DR/16-36_residual-ls/epoch 51"
        # strict=False
    )