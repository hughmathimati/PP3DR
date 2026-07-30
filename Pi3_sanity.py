import sys
import os
import torch
from torch import nn
from PP3DR.training.PP3DR_sanity import SanityTrainer
from PP3DR.training.base_trainer import State
from PP3DR.training.PP3DR_loss import PP3DR_loss
from PP3DR.datasets.nrgbd_dataset import nrgbd_dataset
from Pi3.pi3.models.pi3_training import Pi3
from tqdm import tqdm

@torch.compile()
class Pi3_loss(PP3DR_loss):
    def initialize(self, pred, gt):
        pred['camera_poses'] = pred['camera_poses'][..., :3, :]
        pred['relative_camera_rotations'], pred['relative_camera_translations'] = self.obtain_gt_relative_poses(pred['camera_poses'])
        pred['log_depths'] = torch.log(pred['local_points'][..., 2])
        return super().initialize(pred, gt)

    def obtain_pred_3D_points(self, pred):
        return pred['local_points']

class Pi3Sanity(SanityTrainer):
    def initialize(self, model, pretrained_path, start_checkpointing):
        PP3DR_model = Pi3(decoder_size='large', load_vggt=False, freeze_encoder=True)
        if pretrained_path is not None:
            PP3DR_model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))
            print("Loaded pretrained weights from", pretrained_path)

        def init_vit_weights(module):
            if isinstance(module, nn.Linear):
                # Truncated normal tightly bounds the initial weights
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        PP3DR_model.apply(init_vit_weights)

        AdamW_params, Muon_params = self.get_param_groups(PP3DR_model)

        AdamW = torch.optim.AdamW(AdamW_params, betas=(0.9, 0.99), foreach=True)
        Muon = torch.optim.Muon(Muon_params)
        state = State(self.epochs, self.accelerator.device)
        self.accelerator.register_for_checkpointing(state)
        return state, PP3DR_model, AdamW, Muon

    def get_param_groups(
        self,
        model: nn.Module,
        adamw_lr: float = 1e-4,
        muon_lr: float = 1e-3,
        weight_decay: float = 0.04,
        layer_decay: float = 0.95,
        num_encoder_layers: int = 24,  # DINOv2 ViT-L has 24 blocks
        num_decoder_layers: int = 36  # Pi3 'large' decoder has 36 blocks
    ):
        """
            Separates Pi3 parameters into AdamW and Muon groups.
            Applies unified Layer-wise Learning Rate Decay (LLRD) across the 24 encoder and 36 decoder blocks.
            """
        AdamW_params, Muon_params = {}, {}
        total_layers = num_encoder_layers + num_decoder_layers  # 60 layers total

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # ==========================================
            # 1. Selective Weight Decay
            # ==========================================
            is_1d = param.dim() < 2
            is_special = any(k in name for k in ["bias", "norm", "gamma", "token", "pos_embed", "patch_embed"])

            wd = 0.0 if (is_1d or is_special) else weight_decay

            # ==========================================
            # 2. Unified Layer-wise Learning Rate Decay (LLRD)
            # ==========================================

            # A. Decoder Heads (Absolute max LR)
            if any(k in name for k in
                   ["point_decoder", "point_head", "camera_decoder", "camera_head", "conf_decoder", "global_"]):
                layer_id = total_layers

            # B. DINOv2 Encoder Blocks (Layers 0 to 23)
            elif "encoder.blocks." in name:
                idx = int(name.split("encoder.blocks.")[1].split(".")[0])
                layer_id = idx

            # C. Pi3 Decoder Blocks (Layers 24 to 59)
            elif "decoder." in name:
                idx = int(name.split("decoder.")[1].split(".")[0])
                layer_id = num_encoder_layers + idx

            # D. Stem Embeddings & Registers (Absolute lowest LR)
            elif any(k in name for k in ["patch_embed", "pos_embed", "register_token", "cls_token"]):
                layer_id = -1

            else:
                # Safely default any custom auxiliary weights to the maximum un-decayed LR
                layer_id = total_layers

                # Calculate the exponential decay multiplier
            lr_mult = layer_decay ** (total_layers - layer_id)

            # ==========================================
            # 3. Route to Optimizers
            # ==========================================
            # Muon exclusively optimizes >= 2D matrices (Standard Linear weights)
            if param.dim() >= 2 and not is_special:
                lr = muon_lr * lr_mult
                group_key = (lr, wd)
                if group_key not in Muon_params:
                    Muon_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
                Muon_params[group_key]["params"].append(param)
            else:
                lr = adamw_lr * lr_mult
                group_key = (lr, wd)
                if group_key not in AdamW_params:
                    AdamW_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
                AdamW_params[group_key]["params"].append(param)

        return list(AdamW_params.values()), list(Muon_params.values())

    def train_on_dataset(self, name, iterator, dataloader):
        """
        dataloader parameter is solely for the tqdm progress bar.

        Losses will be logged to WandB every batch. This notably differs from the validation logging behaviour.
        """
        # I'm basically just expecting python to correctly reference the non-parameter variables.
        for batch in tqdm(
                iterator, desc=name, disable=not self.accelerator.is_local_main_process, total=len(dataloader),
                mininterval=1
        ):
            with self.accelerator.accumulate(self.PP3DR_model):
                # Accelerate automatically handles autocast and automatically moves the batch's tensors to the right GPU.
                pred = self.PP3DR_model(batch['images'])
                loss, loss_dict = self.metric(pred, batch)
                self.state.train_losses[self.state.epoch - 1] += loss.detach()
                with torch.autocast(device_type=self.accelerator.device.type, enabled=False):
                    self.AdamW.zero_grad()
                    self.Muon.zero_grad()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(self.PP3DR_model.parameters(), max_norm=1.0)
                    self.AdamW.step()
                    self.Muon.step()
                    self.AdamW_scheduler.step()
                    self.Muon_scheduler.step()

            # Log to WandB once per batch
            if self.accelerator.is_local_main_process:
                self.accelerator.log({f"train/{key}": value for key, value in loss_dict.items()}, step=self.global_step)
                self.global_step += 1


if __name__ == "__main__":
    Pi3Sanity(
        model=Pi3,
        loss=Pi3_loss,
        name="pi3-sanity",
        epochs=1000,
        checkpoint_every=500,
        batch_size=5,
        gradient_accumulation_steps=1,
        train_datasets=[nrgbd_dataset],
    )