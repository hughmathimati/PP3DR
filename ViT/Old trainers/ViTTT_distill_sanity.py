import transformers.optimization
from models.ViTTT import ViTTT
from models.Dinov3 import load_dinov3, obtain_features
from datasets.sintel_dataset import sintel_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.scannet_dataset import scannet_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.dtu_dataset import dtu_dataset
import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
from tqdm import tqdm, trange
import os
import threading
import queue
import concurrent.futures
from functools import partial
import pickle
import accelerate
from accelerate import Accelerator, ProfileKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
accelerator = Accelerator(
    kwargs_handlers = [ ProfileKwargs( activities = ["cpu", "cuda"] ) ],
    dataloader_config = DataLoaderConfiguration(non_blocking=True),
    # project_config = ProjectConfiguration(
    #     project_dir = "/vulcanscratch/hughma/ViT/",
        # automatic_checkpoint_naming = True, # STRICTLY REQUIRED for total_limit to work
        # total_limit = 3
    # )
)


class State:
    def __init__(self, epochs):
        self.epoch = 1
        self.train_losses = torch.zeros((epochs), device = accelerator.device)
        self.val_losses = torch.zeros((epochs), device = accelerator.device)

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


def train_on_dataset(name, iterator, dataloader):
    """dataloader parameter is solely for the tqdm progress bar."""
    # I'm basically just expecting python to correctly reference the non-parameter variables.
    for batch in tqdm(iterator, desc=name, disable=not accelerator.is_local_main_process, total = len(dataloader)):
        # Accelerate automatically handles autocast.
        # torch.no_grad() is included inside obtain_features().
        # with torch.profiler.record_function("dino_train_inference"):
        gt = obtain_features(processor, dino, batch.to(accelerator.device, non_blocking=True), remove_registers=False)
        # with torch.profiler.record_function("ViTTT_train_inference"):
        pred = ViTTT_model(batch.to(accelerator.device, non_blocking=True))
        loss = metric(pred, gt)
        state.train_losses[state.epoch - 1] += loss.detach()
        accelerator.backward(loss)
        # Clamps the total norm of the gradients to 1.0
        torch.nn.utils.clip_grad_norm_(ViTTT_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def val_on_dataset(name, iterator, dataloader):
    for batch in tqdm(iterator, desc=f"Validation {name}", disable=not accelerator.is_local_main_process,
                      total=len(dataloader)):
        # Accelerate automatically handles autocast.
        # torch.no_grad() is included inside obtain_features().
        # with torch.profiler.record_function("dino_val_inference"):
        gt = obtain_features(processor, dino, batch.to(accelerator.device, non_blocking=True),
                             remove_registers=False)
        # with torch.profiler.record_function("ViTTT_val_inference"):
        pred = ViTTT_model(batch.to(accelerator.device, non_blocking=True))
        state.val_losses[state.epoch - 1] += metric(pred, gt).detach()


def get_vittt_param_groups(model: nn.Module, base_lr: float, weight_decay: float = 0.04, layer_decay: float = 0.9,
                           num_layers: int = 12):
    """
    Separates model parameters into optimized groups with Layer-wise LR Decay (LLRD)
    and selective weight decay.
    """
    # Group parameters by their (lr, weight_decay) configuration
    group_dict = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # ==========================================
        # 1. Selective Weight Decay
        # ==========================================
        # A highly reliable PyTorch heuristic: any tensor with < 2 dimensions is a
        # bias or a normalization parameter. We also explicitly check strings for
        # LayerScale gammas and special tokens to match DINOv3.
        is_1d = param.dim() < 2
        is_special = any(k in name for k in ["bias", "norm", "gamma", "token", "pos_embed", "patch_embed"])

        if is_1d or is_special:
            wd = 0.0
        else:
            wd = weight_decay

        # ==========================================
        # 2. Layer-wise Learning Rate Decay (LLRD)
        # ==========================================
        # Calculate the depth of the parameter to scale down its learning rate.
        if "blocks." in name:
            try:
                # Extract the index from names like "blocks.5.TTT.weight"
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                # Layers closer to the input (lower layer_id) get heavier decay
                lr_mult = layer_decay ** (num_layers - layer_id)
            except ValueError:
                lr_mult = 1.0
        elif any(k in name for k in ["patch_embed", "pos_embed", "token"]):
            # The input stem / embeddings get the heaviest possible decay
            lr_mult = layer_decay ** (num_layers + 1)
        else:
            # The final prediction head gets the full, un-decayed base learning rate
            lr_mult = 1.0

        # Calculate final layer-specific learning rate
        lr = base_lr * lr_mult

        # ==========================================
        # 3. Batch into Shared Optimizer Groups
        # ==========================================
        group_key = (lr, wd)
        if group_key not in group_dict:
            group_dict[group_key] = {"params": [], "lr": lr, "weight_decay": wd}

        group_dict[group_key]["params"].append(param)

    return list(group_dict.values())


def initialize(epochs):
    ViTTT_model = ViTTT()

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

    ViTTT_model.apply(init_vit_weights)

    param_groups = get_vittt_param_groups(
        model=ViTTT_model,
        base_lr=1e-4,  # Set your peak learning rate here
        weight_decay=0.04,  # DINO default
        layer_decay=0.9,  # DINO default LLRD
        num_layers=48  # Update to match your model depth
    )

    optimizer = torch.optim.AdamW(param_groups, betas = (0.9, 0.99), foreach = True) # `foreach` doesn't even help, empirically.
    state = State(epochs)
    accelerator.register_for_checkpointing(state)
    return state, ViTTT_model, optimizer


def prepare_dataloaders():
    # 1024 x 436
    sintel_dataloader = torch.utils.data.DataLoader(sintel_dataset(), batch_size=16, shuffle=False, num_workers=3,
                                                 persistent_workers=True, pin_memory=True)

    return sintel_dataloader


@torch.compile(dynamic = True)
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity(dim = -1) # default dim is 1, so we actually do have to explicitly pass this parameter.

    def forward(self, pred, gt):
        return 1 - self.sim(pred, gt).mean()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # DEBUG
    # torch.set_float32_matmul_precision('high')

    epochs = 50
    checkpoint_every = 5
    # Make sure to rename checkpoints folder to old_checkpoints!!!
    checkpoint = None
    jobs = [
        partial(initialize, epochs),
        load_dinov3,
        prepare_dataloaders,
    ]
    # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(job) for job in jobs]
        metric = CosineLoss()
        # futures.as_completed returns futures in the order they complete, not their original order.
        for future in concurrent.futures.as_completed(futures):
            match futures.index(future):
                case 0:
                    state, ViTTT_model, optimizer = future.result()
                case 1:
                    processor, dino = future.result()
                case 2:
                    sintel_dataloader = future.result()

    # Immediately start loading the data.
    sintel_dataloader = accelerator.prepare(sintel_dataloader)
    total_training_steps = len(sintel_dataloader) * epochs
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, total_training_steps // 20, total_training_steps)
    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)
    processor, dino, ViTTT_model, optimizer, scheduler = accelerator.prepare(processor, dino.eval(), ViTTT_model, optimizer, scheduler)

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        if accelerator.is_local_main_process:
            print(f"Loaded checkpoint from {checkpoint}")
        # For some reason, accelerate seems to load the `state` tensors on the cpu. I don't know why this is, but I'll
        # just move them back.
        state.train_losses = state.train_losses.to(accelerator.device, non_blocking = True)
        state.val_losses = state.val_losses.to(accelerator.device, non_blocking = True)
    else:
        accelerate.utils.set_seed(42)
        if accelerator.is_local_main_process:
            print("Starting from scratch, with seed 42.")

    # with accelerator.profile() as prof:
    # Training loop
    start = state.epoch
    while state.epoch <= epochs:
        sintel_iter = iter(sintel_dataloader) # Start pre-loading validation data as early as possible.
        if accelerator.is_local_main_process:
            print(f"Epoch {state.epoch}/{epochs}:")
        ViTTT_model.train()
        train_on_dataset("Sintel", sintel_iter, sintel_dataloader)
        state.epoch += 1

    # End of training loop; write losses to file
    if accelerator.is_local_main_process:
        print("Training done.")
    accelerator.reduce(state.train_losses, "sum")

    if accelerator.is_local_main_process:
        with open("/vulcanscratch/hughma/ViT/sanity_2/train_losses.pkl", "wb") as f:
            pickle.dump(state.train_losses.numpy(force=True), f)
        print("Saved train_losses")
        torch.save(ViTTT_model.state_dict(), "/vulcanscratch/hughma/ViT/sanity_2/ViTTT.pth")
        print("Saved model")

    # if accelerator.is_local_main_process:
    #     prof.export_chrome_trace(f"trace.json")
    #     print("Saved trace")