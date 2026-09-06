from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from training.PP3DR_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        PP3DR,
        PP3DR_loss,
        name="16-36_pixel-shuffle-icnr",
        # pretrained_path="/vulcanscratch/hughma/PP3DR/16-36_hughber/PP3DR.pth",
        # checkpoint=""
        # strict=False
    )