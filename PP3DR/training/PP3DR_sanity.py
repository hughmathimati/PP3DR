from training.base_trainer import BaseTrainer
from models.PP3DR import PP3DR
from models.mixed_head import PP3DR_mixed_head
from training.PP3DR_loss import PP3DR_loss
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.dtu_dataset import dtu_dataset
from datasets.dynamic_replica_dataset import dynamic_replica_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.flying_things_3d_dataset import flying_things_3d_dataset
from datasets.sintel_dataset import sintel_dataset

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
        self.train_on_dataset("Combined train set", self.train_iter, self.train_dataloader)

        if self.state.epoch != self.epochs:
            self.train_iter = iter(self.train_dataloader)

        self.state.epoch += 1

if __name__ == "__main__":
    SanityTrainer(
        model=PP3DR,
        loss=PP3DR_loss,
        name="finetune-double-upscale",
        epochs=1000,
        checkpoint_every=1000, # MUST checkpoint before saving files. Also just good practice.
        pretrained_path="/vulcanscratch/hughma/PP3DR/LDD-1000-double-upscale/PP3DR.pth",
        freeze_feature_extractor=False,
        use_muon=False,
        gradient_accumulation_steps=1,
        train_datasets=[nrgbd_dataset],
    )