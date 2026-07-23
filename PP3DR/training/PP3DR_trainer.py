name = "no-scale"
import transformers.optimization
from models.PP3DR import PP3DR
from models.PP3DR_Dino import PP3DR_Dino
from PP3DR_loss import PP3DR_loss
from adapted_pi3_loss import Adapted_Pi3_loss
from models.Dinov3 import load_dinov3, obtain_features

from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.dtu_dataset import dtu_dataset
from datasets.dynamic_replica_dataset import dynamic_replica_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.flying_things_3d_dataset import flying_things_3d_dataset
from datasets.sintel_dataset import sintel_dataset
from datasets.nrgbd_dataset import nrgbd_dataset

import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
from tqdm import tqdm, trange
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import accelerate
from accelerate import Accelerator, ProfileKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from torch.utils.data import ConcatDataset, DataLoader
import wandb
import logging

# This will hopefully solve "ValueError: too many fds".
torch.multiprocessing.set_sharing_strategy('file_system')

accelerator = Accelerator(
    # kwargs_handlers=[ProfileKwargs(activities=["cpu", "cuda"])],
    dataloader_config=DataLoaderConfiguration(non_blocking=True),
    log_with="wandb", project_dir="/vulcanscratch/hughma/PP3DR/tensorboard",
    gradient_accumulation_steps=8
)
if accelerator.is_local_main_process:
    accelerator.init_trackers(project_name="PP3DR")
torch.cuda.set_device(accelerator.device)


class State:
    def __init__(self, epochs):
        self.epoch = 1
        self.train_losses = torch.zeros((epochs), device=accelerator.device)
        self.val_losses = torch.zeros((epochs), device=accelerator.device)

    def state_dict(self):
        # This is called automatically by accelerator.save_state()
        return {
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

    def load_state_dict(self, state):
        # This is called automatically by accelerator.load_state()
        self.epoch = state["epoch"]
        self.train_losses = state["train_losses"]
        self.val_losses = state["val_losses"]


def prepare_dataloaders():
    # I have arranged the datasets roughly in order of the time they take to initialize.
    datasets = [dynamic_replica_dataset, flying_things_3d_dataset, nrgbd_dataset, dtu_dataset, eth3d_dataset]
    constructed = []
    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(dataset) for dataset in datasets]

        # Concurrently load Sintel while we're waiting.
        val_dataloader = DataLoader(
            sintel_dataset(),
            batch_size=6,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        for completed in as_completed(tasks):
            constructed.append(completed.result())

    train_dataloader = DataLoader(
        ConcatDataset(constructed),
        # Batch size of 6 sequences, each with 10 images (60 images total)
        batch_size=6,
        shuffle=True,  # Critical: Shuffles across all domains!
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader

# "/vulcanscratch/hughma/PP3DR/no-scale/PP3DR.pth"
def initialize(epochs, pretrained_path=None):
    PP3DR_model = PP3DR()
    # PP3DR_model = PP3DR_Dino()
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

    AdamW_params, Muon_params = get_pp3dr_param_groups(
        model=PP3DR_model,
        num_layers=40  # 36 blocks + 4 blocks in each per-task head
    )

    AdamW = torch.optim.AdamW(AdamW_params, betas=(0.9, 0.99), foreach=True)
    Muon = torch.optim.Muon(Muon_params)
    state = State(epochs)
    accelerator.register_for_checkpointing(state)
    return state, PP3DR_model, AdamW, Muon


def get_pp3dr_param_groups(model: nn.Module, adamw_lr: float = 1e-5, muon_lr: float = 5e-3,
                           weight_decay: float = 0.04, layer_decay: float = 0.95,
                           num_layers: int = 36):
    """
    Separates model parameters into AdamW and Muon parameter groups,
    applying Layer-wise LR Decay (LLRD) and selective weight decay independently to both scales.
    Returns: (AdamW parameter groups, Muon parameter groups)
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
            offset += 36
        if "global_blocks." in name:
            try:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                lr_mult = layer_decay ** (num_layers - (2 * layer_id + offset))
            except ValueError:
                lr_mult = 1.0
        elif "local_blocks." in name:
            try:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                lr_mult = layer_decay ** (num_layers - (2 * layer_id + 1 + offset))
            except ValueError:
                lr_mult = 1.0
        elif any(k in name for k in ["patch_embed", "pos_embed", "token"]):
            lr_mult = layer_decay ** (num_layers + 1)
        else:
            lr_mult = 1.0
        # print(name, lr_mult)

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


def train_on_dataset(name, iterator, dataloader):
    """
    dataloader parameter is solely for the tqdm progress bar.

    Losses will be logged to WandB every batch. This notably differs from the validation logging behaviour.
    """
    global global_step
    # I'm basically just expecting python to correctly reference the non-parameter variables.
    for batch in tqdm(
            iterator, desc=name, disable=not accelerator.is_local_main_process, total=len(dataloader), mininterval=1
    ):
        with accelerator.accumulate(PP3DR_model):
            # Accelerate automatically handles autocast and automatically moves the batch's tensors to the right GPU.
            pred = PP3DR_model(batch['images'], batch['rope_x'], batch['rope_y'])
            loss, loss_dict = metric(pred, batch)
            state.train_losses[state.epoch - 1] += loss.detach()
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                AdamW.zero_grad()
                Muon.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(PP3DR_model.parameters(), max_norm=1.0)
                AdamW.step()
                Muon.step()
                AdamW_scheduler.step()
                Muon_scheduler.step()
        # Log to WandB once per batch
        if accelerator.is_local_main_process:
            accelerator.log({f"train/{key}": value for key, value in loss_dict.items()}, step=global_step)
            global_step += 1


def val_on_dataset(name, iterator, dataloader):
    """
    dataloader parameter is solely for the tqdm progress bar.

    Average validation loss per epoch getes logged. This notably differs from the training logging behaviour.
    """
    global global_step
    sum_loss_dict = dict(total_loss=0, point_loss=0, translation_loss=0, rotation_loss=0, normal_loss=0)
    len_dataloader = len(dataloader)
    for batch in tqdm(iterator, desc=f"Validation {name}", disable=not accelerator.is_local_main_process,
                      total=len_dataloader):
        # Accelerate automatically handles autocast.
        pred = PP3DR_model(batch['images'], batch['rope_x'], batch['rope_y'])
        loss, loss_dict = metric(pred, batch)
        state.val_losses[state.epoch - 1] += loss.detach()
        for k, v in loss_dict.items():
            sum_loss_dict[k] += v.item()

    # Log to Tensorboard once per epoch
    if accelerator.is_local_main_process:
        loss_dict = {f"val/{key}": value / len_dataloader for key, value in sum_loss_dict.items()}
        accelerator.log(loss_dict, step=global_step)
        global_step += 1


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # DEBUG
    # Failed to reload cubin file statically launchable autotuner triton_poi_fused_arange_div_expand_mul_stack_sub_unsqueeze_view_0
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_rank_{os.environ.get("LOCAL_RANK", "0")}"
    torch.set_float32_matmul_precision('high')

    epochs = 50
    checkpoint_every = 10
    checkpoint = "/vulcanscratch/hughma/PP3DR/no-scale/epoch 50"

    # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
    with ThreadPoolExecutor() as executor:
        a = executor.submit(prepare_dataloaders)
        b = executor.submit(initialize, epochs)

        metric = PP3DR_loss(scale=False)
        # metric = Adapted_Pi3_loss()

        train_dataloader, val_dataloader = a.result()
        state, PP3DR_model, AdamW, Muon = b.result()

    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)
    total_training_steps = len(train_dataloader) * epochs

    AdamW_scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        AdamW,
        total_training_steps // 10,
        total_training_steps
    )
    Muon_scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        Muon,
        total_training_steps // 10,
        total_training_steps
    )
    # Register the LR schedulers
    accelerator.register_for_checkpointing(AdamW_scheduler, Muon_scheduler)
    PP3DR_model, AdamW, Muon, AdamW_scheduler, Muon_scheduler = accelerator.prepare(
        PP3DR_model, AdamW, Muon, AdamW_scheduler, Muon_scheduler
    )

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        if accelerator.is_local_main_process:
            print(f"Loaded checkpoint from {checkpoint}")
        # For some reason, accelerate seems to load the `state` tensors on the cpu. I don't know why this is, but I'll
        # just move them back.
        state.train_losses = state.train_losses.to(accelerator.device, non_blocking=True)
        state.val_losses = state.val_losses.to(accelerator.device, non_blocking=True)
    else:
        accelerate.utils.set_seed(42)
        if accelerator.is_local_main_process:
            print("Starting from scratch, with seed 42.")

    train_iter = iter(train_dataloader)

    # Training loop
    start = state.epoch
    # global_step exists for the sole purpose of tensorboard.
    global_step = 0
    # with accelerator.profile() as prof:
    while state.epoch <= epochs:
        val_iter = iter(val_dataloader)
        if state.epoch % checkpoint_every == 0:
            accelerator.save_state(output_dir=f"/vulcanscratch/hughma/PP3DR/{name}/epoch {state.epoch}",
                                   total_limit=3)
            if accelerator.is_local_main_process:
                print("Checkpoint saved.")

        if accelerator.is_local_main_process:
            print(f"Epoch {state.epoch}/{epochs}:")

        PP3DR_model.train()
        with torch.profiler.record_function("training"):
            train_on_dataset("Combined train set", train_iter, train_dataloader)

        # Pre-load train iterator for the next epoch.
        if state.epoch != epochs:
            train_iter = iter(train_dataloader)

        # Validation on Sintel
        PP3DR_model.eval()
        # with torch.profiler.record_function("validation"):
        with torch.no_grad():
            val_on_dataset("Sintel", val_iter, val_dataloader)

        state.epoch += 1

    # if accelerator.is_local_main_process:
    #     prof.export_chrome_trace(f"trace.json")
    #     print("Saved trace")
    # accelerator.end_training()

    # End of training loop; write losses to file
    if accelerator.is_local_main_process:
        print("Training done.")
    accelerator.reduce(state.train_losses, "sum")
    accelerator.reduce(state.val_losses, "sum")

    if accelerator.is_local_main_process:
        with open(f"/vulcanscratch/hughma/PP3DR/{name}/train_losses.pkl", "wb") as f:
            pickle.dump(state.train_losses.numpy(force=True), f)
        print("Saved train_losses")
        with open(f"/vulcanscratch/hughma/PP3DR/{name}/val_losses.pkl", "wb") as f:
            pickle.dump(state.val_losses.numpy(force=True), f)
        print("Saved val_losses")
        torch.save(PP3DR_model.state_dict(), f"/vulcanscratch/hughma/PP3DR/{name}/PP3DR.pth")
        print("Saved model")
    if accelerator.is_local_main_process:
        accelerator.end_training()
