name = "12_blocks"
import transformers.optimization
from models.ViTTT import ViTTT
from models.Dinov3 import load_dinov3, obtain_features
from datasets.sintel_dataset import sintel_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.flying_things_3d_dataset import flying_things_3d_dataset
from datasets.dynamic_replica_dataset import dynamic_replica_dataset
from datasets.mega_depth_dataset import mega_depth_dataset
from datasets.object_net_dataset import object_net_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.from_games_dataset import from_games_dataset
from datasets.open_images_dataset import open_images_dataset
from datasets.youtube_vis_dataset import youtube_vis_dataset
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
from torch.utils.data import ConcatDataset, DataLoader

accelerator = Accelerator(
    kwargs_handlers=[ProfileKwargs(activities=["cpu", "cuda"])],
    dataloader_config=DataLoaderConfiguration(non_blocking=True),
    # project_config = ProjectConfiguration(
    #     project_dir = "/vulcanscratch/hughma/ViT/",
    # automatic_checkpoint_naming = True, # STRICTLY REQUIRED for total_limit to work
    # total_limit = 3
    # )
)


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


def train_on_dataset(name, iterator, dataloader):
    """dataloader parameter is solely for the tqdm progress bar."""
    # I'm basically just expecting python to correctly reference the non-parameter variables.
    for batch in tqdm(
            iterator, desc=name, disable=not accelerator.is_local_main_process, total=len(dataloader), mininterval = 1
    ):
        # Accelerate automatically handles autocast.
        # torch.no_grad() is included inside obtain_features().
        # with torch.profiler.record_function("dino_train_inference"):
        gt = obtain_features(processor, dino, batch.to(accelerator.device, non_blocking=True))
        # with torch.profiler.record_function("ViTTT_train_inference"):
        pred = ViTTT_model(batch.to(accelerator.device, non_blocking=True))[:, 5:] # Discard register tokens
        loss = metric(pred, gt)
        state.train_losses[state.epoch - 1] += loss.detach()
        AdamW.zero_grad()
        Muon.zero_grad()
        accelerator.backward(loss)
        # Clamps the total norm of the gradients to 1.0
        torch.nn.utils.clip_grad_norm_(ViTTT_model.parameters(), max_norm=1.0)
        AdamW.step()
        Muon.step()
        AdamW_scheduler.step()
        Muon_scheduler.step()


def val_on_dataset(name, iterator, dataloader):
    for batch in tqdm(
            iterator, desc=f"Validation {name}", disable=not accelerator.is_local_main_process, total=len(dataloader), mininterval = 1
    ):
        # Accelerate automatically handles autocast.
        # torch.no_grad() is included inside obtain_features().
        # with torch.profiler.record_function("dino_val_inference"):
        gt = obtain_features(processor, dino, batch.to(accelerator.device, non_blocking=True))
        # with torch.profiler.record_function("ViTTT_val_inference"):
        pred = ViTTT_model(batch.to(accelerator.device, non_blocking=True))[:, 5:] # Discard register tokens
        state.val_losses[state.epoch - 1] += metric(pred, gt).detach()


def get_vittt_param_groups(model: nn.Module, adamw_lr: float = 1e-4, muon_lr: float = 1e-3,
                           weight_decay: float = 0.04, layer_decay: float = 0.9,
                           num_layers: int = 24):
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
        # Calculate the depth multiplier (0.0 to 1.0 scale)
        if "blocks." in name:
            try:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                lr_mult = layer_decay ** (num_layers - layer_id)
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


def initialize(epochs, pretrained_path = None):
    ViTTT_model = ViTTT(blocks = 12)
    if pretrained_path is not None:
        ViTTT_model.load_state_dict(torch.load(pretrained_path, weights_only=True))
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

    ViTTT_model.apply(init_vit_weights)

    AdamW_params, Muon_params = get_vittt_param_groups(
        model=ViTTT_model,
        num_layers=12  # Update to match your model depth
    )

    AdamW = torch.optim.AdamW(AdamW_params, betas=(0.9, 0.99), foreach=True)
    Muon = torch.optim.Muon(Muon_params)
    state = State(epochs)
    accelerator.register_for_checkpointing(state)
    return state, ViTTT_model, AdamW, Muon


def prepare_dataloaders():
    train_dataloader = DataLoader(
        ConcatDataset([
            nrgbd_dataset(),
            flying_things_3d_dataset(),
            dynamic_replica_dataset(),
            mega_depth_dataset(),
            object_net_dataset(),
            eth3d_dataset(),
            from_games_dataset(),
            open_images_dataset(),
            youtube_vis_dataset()
        ]),
        batch_size=128, # is 128 any better than 64?
        shuffle=True,  # Critical: Shuffles across all domains!
        num_workers=16,
        pin_memory=True
    )

    # 1024 x 436
    val_dataloader = DataLoader(sintel_dataset(), batch_size=16, shuffle=False, num_workers=4,
                                                 persistent_workers=True, pin_memory=True)

    return train_dataloader, val_dataloader


@torch.compile()
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity(
            dim=-1)  # default dim is 1, so we actually do have to explicitly pass this parameter.

    def forward(self, pred, gt):
        # Ignore the register tokens. pca_lowrank() to lower dino output to ViTTT dim.
        return 1 - self.sim(pred, gt).mean()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # DEBUG
    # Failed to reload cubin file statically launchable autotuner triton_poi_fused_arange_div_expand_mul_stack_sub_unsqueeze_view_0
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_rank_{os.environ.get("LOCAL_RANK", "0")}"
    torch.set_float32_matmul_precision('high')

    epochs = 8
    checkpoint_every = 1
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
                    state, ViTTT_model, AdamW, Muon = future.result()
                case 1:
                    processor, dino = future.result()
                case 2:
                    train_dataloader, val_dataloader = future.result()

    # Immediately start loading the data.
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
    processor, dino, ViTTT_model, AdamW, Muon, AdamW_scheduler, Muon_scheduler = accelerator.prepare(
        processor, dino.eval(), ViTTT_model, AdamW, Muon, AdamW_scheduler, Muon_scheduler
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

    # with accelerator.profile() as prof:
    # Training loop
    start = state.epoch
    while state.epoch <= epochs:
        val_iter = iter(val_dataloader)
        if state.epoch % checkpoint_every == 0 or state.epoch == 1:
            accelerator.save_state(output_dir=f"/vulcanscratch/hughma/ViT/{name}/epoch {state.epoch}",
                                   total_limit=3)
            if accelerator.is_local_main_process:
                print("Checkpoint saved.")

        if accelerator.is_local_main_process:
            print(f"Epoch {state.epoch}/{epochs}:")

        ViTTT_model.train()
        train_on_dataset("Combined train set", train_iter, train_dataloader)

        # Pre-load NRGBD iterator for the next epoch.
        if state.epoch != epochs:
            train_iter = iter(train_dataloader)

        # Validation on Sintel
        ViTTT_model.eval()
        with torch.no_grad():
            val_on_dataset("Sintel", val_iter, val_dataloader)

        state.epoch += 1

    # End of training loop; write losses to file
    if accelerator.is_local_main_process:
        print("Training done.")
    accelerator.reduce(state.train_losses, "sum")
    accelerator.reduce(state.val_losses, "sum")

    if accelerator.is_local_main_process:
        with open(f"/vulcanscratch/hughma/ViT/{name}/train_losses.pkl", "wb") as f:
            pickle.dump(state.train_losses.numpy(force=True), f)
        print("Saved train_losses")
        with open(f"/vulcanscratch/hughma/ViT/{name}/val_losses.pkl", "wb") as f:
            pickle.dump(state.val_losses.numpy(force=True), f)
        print("Saved val_losses")
        torch.save(ViTTT_model.state_dict(), f"/vulcanscratch/hughma/ViT/{name}/ViTTT.pth")
        print("Saved model")

    # if accelerator.is_local_main_process:
    #     prof.export_chrome_trace(f"trace.json")
    #     print("Saved trace")
