from training.base_trainer import BaseTrainer
from models.PP3DR_depth_focal import PP3DR
from training.PP3DR_depth_focal_loss import PP3DR_loss
from datasets.nrgbd_dataset import nrgbd_dataset
import torch

class SanityTrainer(BaseTrainer):
    def per_epoch(self):
        if self.state.epoch % self.checkpoint_every == 0:
            self.accelerator.save_state(output_dir=f"/vulcanscratch/hughma/PP3DR/{self.name}/epoch {self.state.epoch}",
                                   total_limit=3)
            if self.accelerator.is_local_main_process:
                print("Checkpoint saved.")

        if self.accelerator.is_local_main_process:
            print(f"Epoch {self.state.epoch}/{self.epochs}:")

        self.PP3DR_model.train()
        with torch.profiler.record_function("training"):
            self.train_on_dataset("Combined train set", self.train_iter, self.train_dataloader)

        if self.state.epoch != self.epochs:
            self.train_iter = iter(self.train_dataloader)

        self.state.epoch += 1

if __name__ == "__main__":
    SanityTrainer(
        model=PP3DR,
        loss=PP3DR_loss,
        name="TTT-depth-proj",
        epochs=1000,
        checkpoint_every=500,
        batch_size=5,
        gradient_accumulation_steps=1,
        start_checkpointing=6,
        train_datasets=[nrgbd_dataset],
    )