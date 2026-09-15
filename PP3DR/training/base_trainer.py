from datasets.dynamic_replica.dynamic_replica_dataset import dynamic_replica_dataset
from datasets.dynamic_replica.dynamic_replica_val_dataset import dynamic_replica_val_dataset
from datasets.dynamic_replica.dynamic_replica_test_dataset import dynamic_replica_test_dataset
from datasets.flying_things_3d.flying_things_3d_dataset import flying_things_3d_dataset
from datasets.flying_things_3d.flying_things_3d_test_dataset import flying_things_3d_test_dataset
from datasets.rtmv.rtmv_dataset import rtmv_dataset
from datasets.rtmv.rtmv_test_dataset import rtmv_test_dataset
from datasets.interior_net_dataset import interior_net_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.dtu_dataset import dtu_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.sintel.sintel_dataset import sintel_dataset

import transformers.optimization
import torch
from torch import nn
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import accelerate
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import ConcatDataset, DataLoader
import os
import math
from torch.utils.data import default_collate
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def debug_collate(batch):
    """
    A drop-in replacement for PyTorch's default_collate.
    If a shape mismatch or View trap triggers a crash, this will intercept it
    and print a highly readable summary of the batch shapes so you can find the culprit!
    """
    try:
        return default_collate(batch)
    except Exception as e:
        print("\n" + "=" * 60)
        print("CRASH: BATCH SHAPE MISMATCH OR VIEW DETECTED!")
        print("=" * 60)

        for i, item in enumerate(batch):
            print(f"\n--- Batch Item {i} ---")
            for key, val in item.items():
                if isinstance(val, torch.Tensor):
                    # We also print .is_contiguous() because Views (Trap 1)
                    # are often flagged as non-contiguous memory blocks!
                    print(f"{key}: shape {list(val.shape)} | dtype: {val.dtype} | contiguous: {val.is_contiguous()}")
                else:
                    print(f"{key}: {val}")

        print("=" * 60 + "\n", flush=True)

        # Re-raise the error so the script still halts safely
        raise e

class State:
    def __init__(self):
        self.epoch = 1
        self.global_step = 0

    def state_dict(self):
        # This is called automatically by accelerator.save_state()
        return {
            "epoch": self.epoch,
            "global_step": self.global_step
        }

    def load_state_dict(self, state):
        # This is called automatically by accelerator.load_state()
        self.epoch = state['epoch']
        self.global_step = state['global_step']

class BaseTrainer:
    """
    Base trainer class, so I don't have a dozen trainer scripts with mostly the same code.
    """

    def __init__(self,
                 model,
                 loss,
                 name="checkpoints",
                 epochs=100,
                 checkpoint_every=51,
                 pretrained_path=None,
                 strict=True,
                 ema = False,
                 batch_size=4,
                 gradient_accumulation_steps=8,
                 checkpoint=None,
                 # start_checkpointing is handled inside PP3DR itself.
                 train_datasets=[
                     dynamic_replica_dataset, dynamic_replica_val_dataset, dynamic_replica_test_dataset,
                     flying_things_3d_dataset, flying_things_3d_test_dataset,
                     rtmv_dataset, rtmv_test_dataset,
                     interior_net_dataset, nrgbd_dataset, dtu_dataset, eth3d_dataset
                 ],
                 val_dataset=sintel_dataset,
                 ):
        # This will hopefully solve "ValueError: too many fds".
        torch.multiprocessing.set_sharing_strategy('file_system')
        # Failed to reload cubin file statically launchable autotuner triton_poi_fused_arange_div_expand_mul_stack_sub_unsqueeze_view_0
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_rank_{os.environ.get("LOCAL_RANK", "0")}"
        # If we do perform a float32 matmul, use TensorFloat20 operations instead.
        torch.set_float32_matmul_precision('high')
        # Set this to True to help debug invalid values (slows things down)
        # torch.autograd.set_detect_anomaly(True) # DEBUG

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(non_blocking=True),
            log_with="wandb",
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        if self.accelerator.is_local_main_process:
            self.accelerator.init_trackers(project_name="PP3DR")
        torch.cuda.set_device(self.accelerator.device)
        if self.accelerator.is_local_main_process:
            print("\033[95m" + f"Train job: {name}" + "\033[0m")

        self.name = name
        self.epochs = epochs
        self.pretrained_path = pretrained_path
        self.checkpoint_every = checkpoint_every

        # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
        with ThreadPoolExecutor() as executor:
            a = executor.submit(
                self.prepare_dataloaders,
                train_datasets,
                val_dataset,
                batch_size
            )
            b = executor.submit(
                self.initialize,
                model,
                pretrained_path,
                strict,
            )
            self.metric = loss(scale=False)
            train_dataloader, val_dataloader = a.result()
            train_model, AdamW = b.result()

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)

        total_training_steps = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps) * epochs
        AdamW_scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            AdamW,
            total_training_steps // 10,
            total_training_steps
        )
        self.train_model, self.AdamW, self.AdamW_scheduler, = self.accelerator.prepare(
            train_model, AdamW, AdamW_scheduler
        )

        self.ema = ema
        if ema:
            # train_model is the unwrapped model, and self.train_model is the accelerator-prepared wrapped model.
            self.eval_model = AveragedModel(train_model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
            self.eval_model = self.eval_model.to(self.accelerator.device)
            # self.eval_model.eval() will get called right before the validation for each epoch.
            for param in self.eval_model.parameters():
                param.requires_grad = False
            self.accelerator.register_for_checkpointing(self.eval_model)
        else:
            # Safely point the eval reference to the prepared training model
            self.eval_model = self.train_model
        self.state = State()
        self.accelerator.register_for_checkpointing(self.state)

        if checkpoint is not None:
            self.accelerator.load_state(checkpoint)
            if self.accelerator.is_local_main_process:
                print(f"Loaded checkpoint from {checkpoint}.")
        else:
            accelerate.utils.set_seed(42)
            if self.accelerator.is_local_main_process:
                print("Starting from scratch, with seed 42.")

        self.train_iter = iter(self.train_dataloader)

        # Training loop
        while self.state.epoch <= epochs:
            self.per_epoch()

        if self.accelerator.is_local_main_process:
            print("Training done.")
            torch.save(self.eval_model.state_dict(), f"/vulcanscratch/hughma/PP3DR/{name}/PP3DR.pth")
            print("Saved model.")
        self.accelerator.end_training()

    def prepare_dataloaders(self, train_datasets, val_dataset, batch_size):
        constructed = []
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(dataset) for dataset in train_datasets]

            # Concurrently load Sintel while we're waiting.
            val_dataloader = DataLoader(
                val_dataset(),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            for completed in as_completed(tasks):
                constructed.append(completed.result())

        train_dataloader = DataLoader(
            ConcatDataset(constructed),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            # collate_fn=debug_collate
        )

        return train_dataloader, val_dataloader

    def initialize(self, model, pretrained_path, strict):
        # Drop rates get handled inside the model itself.
        train_model = model()

        # 1. Initialize ONLY the new LaCT blocks and Decoder Heads!
        for name, module in train_model.named_modules():
            if isinstance(module, nn.Linear):
                # Truncated normal tightly bounds the initial weights
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if pretrained_path is not None:
            loaded_state_dict = torch.load(pretrained_path, weights_only=True, map_location="cpu")

            # 2. Fix the DDP "module." prefix trap!
            clean_state_dict = {}
            for k, v in loaded_state_dict.items():
                if k.startswith("module."):
                    k = k[7:]

                    # if "final_upscale" in k:
                    #     print(f"Skipping poisoned weight: {k}")
                    #     clean_state_dict[k] = torch.zeros_like(v)
                    #     continue

                clean_state_dict[k] = v

            # 3. Load the fine-tuned weights ON TOP of the initialization.
            missing, unexpected = train_model.load_state_dict(clean_state_dict, strict=strict)
            print(f"Loaded pretrained weights from {pretrained_path}")

            # Print a helpful debug summary so strict=False never blinds us again
            if not strict and self.accelerator.is_local_main_process:
                print(f"  -> Missing keys (New Layers initialized from scratch): {missing}")
                print(f"  -> Unexpected keys (Old Layers discarded): {unexpected}")

        AdamW_params = self.get_param_groups(
            train_model,
            encoder_blocks=train_model.encoder_blocks,
            decoder_blocks=train_model.decoder_blocks,
        )
        AdamW = torch.optim.AdamW(AdamW_params, betas=(0.9, 0.99), foreach=True)
        return train_model, AdamW

    def get_param_groups(self,
                         model: nn.Module,
                         encoder_blocks: int,
                         decoder_blocks: int,
                         adamw_lr: float = 1e-5, # 1e-4
                         weight_decay: float = 0.04,
                         layer_decay: float = 0.95,
                         ):
        """
        Separates model parameters into parameter groups,
        applying Layer-wise LR Decay (LLRD) and selective weight decay independently to both scales.
        Returns: parameter groups
        """
        AdamW_params = {}

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
            # Hardcoded value representing the # of blocks in the feature extractor.
            num_layers = encoder_blocks + decoder_blocks
            if "blocks" in name:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                if "global_blocks." in name:
                    lr_mult = layer_decay ** (num_layers - (2 * layer_id + encoder_blocks))
                elif "local_blocks." in name:
                    lr_mult = layer_decay ** (num_layers - (2 * layer_id + 1 + encoder_blocks))
                else:
                    lr_mult = layer_decay ** (num_layers - layer_id)
            elif any(k in name for k in ["patch_conv", "class_and_registers","block_weights"]):
                lr_mult = layer_decay ** (num_layers + 1)
            elif "final_layer_norm" in name: # ViTTT blocks only
                lr_mult = layer_decay ** (num_layers - encoder_blocks)  # It comes after all the ViTTT blocks.
            else:
                # This comprises the final projections/LayerNorms in the per-task decoder heads.
                # Recall that RoPE does not have trainable parameters.
                lr_mult = 1.0

            # ==========================================
            # 3. Route to Optimizers with Decoupled Base LRs
            # ==========================================
            lr = adamw_lr * lr_mult
            group_key = (lr, wd)
            if group_key not in AdamW_params:
                AdamW_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
            AdamW_params[group_key]["params"].append(param)

        return list(AdamW_params.values())

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
            with self.accelerator.accumulate(self.train_model):
                # Accelerate automatically handles autocast and automatically moves the batch's tensors to the right GPU.
                pred = self.train_model(
                    batch['images'],
                    batch['rope_x'],
                    batch['rope_y'],
                    batch['original_height'],
                    batch['original_width']
                )
                loss, loss_dict = self.metric(pred, batch)

                with torch.autocast(device_type=self.accelerator.device.type, enabled=False):
                    self.AdamW.zero_grad()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), max_norm=1.0)
                        if self.ema:
                            self.eval_model.update_parameters(self.train_model)
                    self.AdamW.step()
                    self.AdamW_scheduler.step()

            # Log to WandB once per batch
            if self.accelerator.is_local_main_process:
                self.accelerator.log(
                    {f"train/{key}": value for key, value in loss_dict.items()},
                    step=self.state.global_step
                )
                self.state.global_step += 1

    def val_on_dataset(self, name, iterator, dataloader):
        """
        dataloader parameter is solely for the tqdm progress bar.

        Average validation loss per epoch getes logged. This notably differs from the training logging behaviour.
        """
        sum_loss_dict = dict(
            total_loss=0,
            point_loss=0,
            depth_loss=0,
            normal_loss=0,
            gradient_matching_loss=0,
            translation_loss=0,
            rotation_loss=0,
        )
        len_dataloader = len(dataloader)
        for batch in tqdm(iterator, desc=f"Validation {name}", disable=not self.accelerator.is_local_main_process,
                          total=len_dataloader):
            # Accelerate automatically handles autocast.
            pred = self.eval_model(
                batch['images'],
                batch['rope_x'],
                batch['rope_y'],
                batch['original_height'],
                batch['original_width']
            )
            loss, loss_dict = self.metric(pred, batch)
            for k, v in loss_dict.items():
                sum_loss_dict[k] += v.item()

        # Log to WandB once per epoch
        if self.accelerator.is_local_main_process:
            loss_dict = {f"val/{key}": value / len_dataloader for key, value in sum_loss_dict.items()}
            self.accelerator.log(loss_dict, step=self.state.global_step)
            self.state.global_step += 1

    def per_epoch(self):
        val_iter = iter(self.val_dataloader)
        if self.state.epoch % self.checkpoint_every == 0:
            self.accelerator.save_state(output_dir=f"/vulcanscratch/hughma/PP3DR/{self.name}/epoch {self.state.epoch}",
                                        total_limit=3)
            if self.accelerator.is_local_main_process:
                print("Checkpoint saved.")

        if self.accelerator.is_local_main_process:
            print(f"Epoch {self.state.epoch}/{self.epochs}:")

        self.train_model.train()
        self.train_on_dataset("Combined train set", self.train_iter, self.train_dataloader)

        # Pre-load train iterator for the next epoch.
        if self.state.epoch != self.epochs:
            self.train_iter = iter(self.train_dataloader)

        # Validation on Sintel
        self.eval_model.eval() # This is necessary if ema is False.
        with torch.no_grad():
            self.val_on_dataset("Sintel", val_iter, self.val_dataloader)

        self.state.epoch += 1
