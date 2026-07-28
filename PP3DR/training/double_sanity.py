from base_trainer import Basetrainer
from models.PP3DR_double import PP3DR_double
from training.PP3DR_depth_focal_loss import PP3DR_loss
from datasets.nrgbd_dataset import nrgbd_dataset
import torch

class DoubleSanity(Basetrainer):
    def get_pp3dr_param_groups(self,
                               model: nn.Module,
                               adamw_lr: float = 1e-5,
                               muon_lr: float = 5e-3,
                               weight_decay: float = 0.04,
                               layer_decay: float = 0.95,
                               # Double blocks means global/local blocks of the same index are actually on the same layer.
                               # Thus, we have 18 (main) + 2 (heads) = 20 layers.
                               num_layers: int = 20
                               ):
        """
        Modified for the double-block structure.
        """
        AdamW_params, Muon_params = {}, {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # ==========================================
            # 1. Selective Weight Decay
            # ==========================================
            # Excludes 1D tensors, biases, norms, and embeddings from weight decay
            is_1d = param.dim() < 2
            is_special = any(k in name for k in ["bias", "norm", "gamma", "token", "pos_embed", "patch_embed"])

            if is_1d or is_special:
                wd = 0.0
            else:
                wd = weight_decay

            # ==========================================
            # 2. Layer-wise Learning Rate Decay (LLRD)
            # ==========================================
            """
            If the model has N blocks in total, it has N // 2 local and global blocks each.
            Immediately after feature extraction, we start with a global block, then a local block, then repeat.
            Thus, the "true" index of a global block is 2 * i, and the "true" index of a local block is 2 * i + 1.
            """
            # Calculate the depth multiplier (0.0 to 1.0 scale)
            offset = 0
            if "pose_decoder." in name or "point_decoder." in name:
                offset += 18
            if "global_blocks." in name:
                try:
                    layer_id = int(name.split("blocks.")[1].split(".")[0])
                    lr_mult = layer_decay ** (num_layers - (layer_id + offset))
                except ValueError:
                    lr_mult = 1.0
            elif "local_blocks." in name:
                try:
                    layer_id = int(name.split("blocks.")[1].split(".")[0])
                    lr_mult = layer_decay ** (num_layers - (layer_id + offset))
                except ValueError:
                    lr_mult = 1.0
            elif any(k in name for k in ["patch_embed", "pos_embed", "token"]):
                lr_mult = layer_decay ** (num_layers + 1)
            else:
                lr_mult = 1.0

            # ==========================================
            # 3. Route to Optimizers with Decoupled Base LRs
            # ==========================================
            if param.dim() == 2:
                # Muon strictly uses the massive base LR, scaled by the decay multiplier
                lr = muon_lr * lr_mult
                group_key = (lr, wd)

                if group_key not in Muon_params:
                    Muon_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
                Muon_params[group_key]["params"].append(param)

            else:
                # AdamW strictly uses the tiny base LR, scaled by the decay multiplier
                lr = adamw_lr * lr_mult
                group_key = (lr, wd)

                if group_key not in AdamW_params:
                    AdamW_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
                AdamW_params[group_key]["params"].append(param)

        return list(AdamW_params.values()), list(Muon_params.values())

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
    DoubleSanity(
        model=PP3DR_double,
        loss=PP3DR_loss,
        name="depth-focal-double-sanity",
        checkpoint="/vulcanscratch/hughma/PP3DR/depth-focal-double-sanity/epoch 500",
        epochs=1000,
        checkpoint_every=500,
        batch_size=2,
        gradient_accumulation_steps=2,
        start_checkpointing=2,
        train_datasets=[nrgbd_dataset],
    )