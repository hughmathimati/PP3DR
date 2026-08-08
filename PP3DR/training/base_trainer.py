import transformers.optimization

from datasets.dynamic_replica_dataset import dynamic_replica_dataset
from datasets.dynamic_replica_val_dataset import dynamic_replica_val_dataset
from datasets.dynamic_replica_test_dataset import dynamic_replica_test_dataset
from datasets.flying_things_3d_dataset import flying_things_3d_dataset
from datasets.flying_things_3d_test_dataset import flying_things_3d_test_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.dtu_dataset import dtu_dataset
from datasets.eth3d_dataset import eth3d_dataset
from datasets.sintel_dataset import sintel_dataset
from datasets.nrgbd_dataset import nrgbd_dataset

import torch
from torch import nn
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import accelerate
from accelerate import Accelerator, ProfileKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from torch.utils.data import ConcatDataset, DataLoader
import os
import math


class State:
    def __init__(self, epochs, device):
        self.epoch = 1
        self.train_losses = torch.zeros((epochs), device=device)
        self.val_losses = torch.zeros((epochs), device=device)

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


class BaseTrainer:
    """
    Base trainer class, so I don't have a dozen trainer scripts with mostly the same code.
    """

    def __init__(self,
                 model,
                 loss,
                 name="checkpoints",
                 epochs=50,
                 checkpoint_every=20,
                 pretrained_path=None,
                 checkpoint=None,
                 strict=True,
                 freeze_feature_extractor=True,
                 use_muon=False, # In my limited testing, Muon underperforms AdamW.
                 batch_size=6,
                 gradient_accumulation_steps=8,
                 start_checkpointing=6,
                 train_datasets=[
                     dynamic_replica_dataset, dynamic_replica_val_dataset, dynamic_replica_test_dataset,
                     flying_things_3d_dataset, flying_things_3d_test_dataset,
                     nrgbd_dataset, dtu_dataset, eth3d_dataset
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
            # kwargs_handlers=[ProfileKwargs(activities=["cpu", "cuda"])],
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
        self.pretrained_path = pretrained_path
        self.epochs = epochs
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
                start_checkpointing,
                strict,
                freeze_feature_extractor,
                use_muon
            )
            self.metric = loss(scale=False)
            train_dataloader, val_dataloader = a.result()
            self.state, PP3DR_model, AdamW, Muon = b.result()

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)
        total_training_steps = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps) * epochs

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
        self.accelerator.register_for_checkpointing(AdamW_scheduler, Muon_scheduler)
        self.PP3DR_model, self.AdamW, self.Muon, self.AdamW_scheduler, self.Muon_scheduler = self.accelerator.prepare(
            PP3DR_model, AdamW, Muon, AdamW_scheduler, Muon_scheduler
        )

        if checkpoint is not None:
            self.accelerator.load_state(checkpoint)
            if self.accelerator.is_local_main_process:
                print(f"Loaded checkpoint from {checkpoint}")
            # For some reason, accelerate seems to load the `state` tensors on the cpu. I don't know why this is, but I'll
            # just move them back.
            self.state.train_losses = self.state.train_losses.to(self.accelerator.device, non_blocking=True)
            self.state.val_losses = self.state.val_losses.to(self.accelerator.device, non_blocking=True)
        else:
            accelerate.utils.set_seed(42)
            if self.accelerator.is_local_main_process:
                print("Starting from scratch, with seed 42.")

        self.train_iter = iter(self.train_dataloader)

        # Training loop
        # global_step exists for the sole purpose of tensorboard.
        self.global_step = 0
        # with accelerator.profile() as prof:
        while self.state.epoch <= self.epochs:
            self.per_epoch()

        # if accelerator.is_local_main_process:
        #     prof.export_chrome_trace(f"trace.json")
        #     print("Saved trace")
        # accelerator.end_training()

        # End of training loop; write losses to file
        if self.accelerator.is_local_main_process:
            print("Training done.")
        self.accelerator.reduce(self.state.train_losses, "sum")
        self.accelerator.reduce(self.state.val_losses, "sum")

        if self.accelerator.is_local_main_process:
            with open(f"/vulcanscratch/hughma/PP3DR/{name}/train_losses.pkl", "wb") as f:
                pickle.dump(self.state.train_losses.numpy(force=True), f)
            print("Saved train_losses")
            with open(f"/vulcanscratch/hughma/PP3DR/{name}/val_losses.pkl", "wb") as f:
                pickle.dump(self.state.val_losses.numpy(force=True), f)
            print("Saved val_losses")
            torch.save(PP3DR_model.state_dict(), f"/vulcanscratch/hughma/PP3DR/{name}/PP3DR.pth")
            print("Saved model")
        if self.accelerator.is_local_main_process:
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
            # Batch size of 6 sequences, each with 10 images (60 images total)
            batch_size=batch_size,
            shuffle=True,  # Critical: Shuffles across all domains!
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

        return train_dataloader, val_dataloader

    def initialize(self, model, pretrained_path, start_checkpointing, strict, freeze_feature_extractor, use_muon):
        PP3DR_model = model(start_checkpointing=start_checkpointing, freeze_feature_extractor=freeze_feature_extractor)

        # 1. Initialize ONLY the new LaCT blocks and Decoder Heads!
        for name, module in PP3DR_model.named_modules():
            # Explicitly protect the backbone from randomization!
            if "ViTTT" in name or "ViT" in name:
                continue

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
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v

            # 3. Load the fine-tuned weights ON TOP of the initialization.
            missing, unexpected = PP3DR_model.load_state_dict(clean_state_dict, strict=strict)
            print(f"Loaded pretrained weights from {pretrained_path}")

            # Print a helpful debug summary so strict=False never blinds us again
            if not strict and self.accelerator.is_local_main_process:
                print(f"  -> Missing keys (New Layers initialized from scratch): {missing}")
                print(f"  -> Unexpected keys (Old Layers discarded): {unexpected}")

        AdamW_params, Muon_params = self.get_param_groups(
            PP3DR_model,
            decoder_blocks=PP3DR_model.decoder_blocks,
            head_blocks=PP3DR_model.point_head.blocks,
            freeze_feature_extractor=freeze_feature_extractor,
            use_muon=use_muon
        )
        AdamW = torch.optim.AdamW(AdamW_params, betas=(0.9, 0.99), foreach=True)
        Muon = torch.optim.Muon(Muon_params)
        state = State(self.epochs, self.accelerator.device)
        self.accelerator.register_for_checkpointing(state)
        return state, PP3DR_model, AdamW, Muon

    def get_param_groups(self,
                         model: nn.Module,
                         decoder_blocks: int = 24,
                         head_blocks: int = 8,
                         freeze_feature_extractor=True,
                         use_muon: bool = True,
                         adamw_lr: float = 1e-4,
                         muon_lr: float = 1e-3,
                         weight_decay: float = 0.04,
                         layer_decay: float = 0.95,
                         ):
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
            # Hardcoded value representing the # of blocks in the feature extractor.
            encoder_offset = 0 if freeze_feature_extractor else 12
            num_layers = decoder_blocks + head_blocks + encoder_offset
            offset = encoder_offset
            if "pose_head" in name or "point_head" in name:
                offset += decoder_blocks
            if "blocks" in name:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
                if "global_blocks." in name:
                    lr_mult = layer_decay ** (num_layers - (2 * layer_id + offset))
                elif "local_blocks." in name:
                    lr_mult = layer_decay ** (num_layers - (2 * layer_id + 1 + offset))
                else:
                    lr_mult = layer_decay ** (num_layers - layer_id)
            elif any(k in name for k in ["patch_conv", "class_and_registers"]): # ViTTT blocks only
                lr_mult = layer_decay ** (num_layers + 1)
            elif "final_layer_norm" in name: # ViTTT blocks only
                lr_mult = layer_decay ** (num_layers - encoder_offset)  # It comes after all the ViTTT blocks.
            else:
                # This comprises the final projections/LayerNorms in the per-task decoder heads.
                # Recall that RoPE does not have trainable parameters.
                lr_mult = 1.0

            # ==========================================
            # 3. Route to Optimizers with Decoupled Base LRs
            # ==========================================
            if use_muon:
                if param.dim() == 2 and "point_head" not in name and "pose_head" not in name:
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
            else:
                lr = adamw_lr * lr_mult
                group_key = (lr, wd)
                if group_key not in AdamW_params:
                    AdamW_params[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
                AdamW_params[group_key]["params"].append(param)

        # If we're not using Muon, we need to add a dummy parameter at the very end.
        if not use_muon:
            Muon_params[(1, 1)] = {"params": [torch.zeros(1, 1)], "lr": 1, "weight_decay": 1}  # dummy parameter

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
                pred = self.PP3DR_model(batch['images'], batch['rope_x'], batch['rope_y'])
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
            pred = self.PP3DR_model(batch['images'], batch['rope_x'], batch['rope_y'])
            loss, loss_dict = self.metric(pred, batch)
            self.state.val_losses[self.state.epoch - 1] += loss.detach()
            for k, v in loss_dict.items():
                sum_loss_dict[k] += v.item()

        # Log to Tensorboard once per epoch
        if self.accelerator.is_local_main_process:
            loss_dict = {f"val/{key}": value / len_dataloader for key, value in sum_loss_dict.items()}
            self.accelerator.log(loss_dict, step=self.global_step)
            self.global_step += 1

    def per_epoch(self):
        val_iter = iter(self.val_dataloader)
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

        # Pre-load train iterator for the next epoch.
        if self.state.epoch != self.epochs:
            self.train_iter = iter(self.train_dataloader)

        # Validation on Sintel
        self.PP3DR_model.eval()
        # with torch.profiler.record_function("validation"):
        with torch.no_grad():
            self.val_on_dataset("Sintel", val_iter, self.val_dataloader)

        self.state.epoch += 1
