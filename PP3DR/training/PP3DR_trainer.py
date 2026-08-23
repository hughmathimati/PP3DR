from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.PP3DR_pretrained_depth import PP3DR_DenseHead
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="bigger-part-3",
        pretrained_path="/vulcanscratch/hughma/PP3DR/5-23_11-23_dim-1024_100-more-epochs/PP3DR.pth",
        # checkpoint=""
    )