from training.base_trainer import BaseTrainer
from models.post_proj_res_conv import post_proj_res_conv
from training.PP3DR_depth_focal_loss import PP3DR_loss

if __name__ == "__main__":
    BaseTrainer(
        post_proj_res_conv,
        PP3DR_loss
    )