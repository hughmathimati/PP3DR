from models.ViTTT import ViTTT
from models.Dinov3 import load_dinov3, obtain_features
from datasets.sintel_dataset import sintel_dataset
from datasets.nrgbd_dataset import nrgbd_dataset
from datasets.scannet_dataset import scannet_dataset
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


def set_seed(seed=42):
    # Python built-in
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_on_dataset(name, iterator):
    # I'm basically just expecting python to correctly reference the non-parameter variables.
    cur_batch = next(iterator)
    cur_gt = obtain_features(processor, dino, cur_batch.to("cuda:0", non_blocking=True), remove_registers=False).to("cuda:1", non_blocking=True)
    batch_id = 0 # DEBUG
    for next_batch in tqdm(iterator, desc=name):
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
            with torch.profiler.record_function("ViTTT_train_inference"):
                pred = ViTTT_model(cur_batch.to("cuda:1", non_blocking=True))
            # torch.no_grad() is included inside obtain_features().
            with torch.profiler.record_function("dino_train_inference"):
                next_gt = obtain_features(processor, dino, next_batch.to("cuda:0", non_blocking=True), remove_registers=False).to("cuda:1", non_blocking=True)
            loss = metric(pred, cur_gt)
        train_losses[epoch] += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Hopefully this automatically synchronizes rather than leading to a race condition...
        cur_batch, cur_gt = next_batch, next_gt
        batch_id += 1
        if batch_id == 50: break
    # Complete the last iteration
    with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)), torch.profiler.record_function("ViTTT_train_inference"):
        pred = ViTTT_model(cur_batch.to("cuda:1", non_blocking=True))
        loss = metric(pred, cur_gt)
    train_losses[epoch] += loss.detach()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def dino_producer(iterator, pipeline_queue):
    # ---------------------------------------------------------
    # THREAD A: THE DINO PRODUCER (GPU 0)
    # ---------------------------------------------------------
    stream_dino = torch.cuda.Stream(device=0)

    for batch in iterator:
        with torch.cuda.stream(stream_dino), torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            # Move batch to GPU 0
            b_g0 = batch.to("cuda:0", non_blocking=True)

            with torch.profiler.record_function("dino_train_inference"):
                gt_g0 = obtain_features(processor, dino, b_g0, remove_registers=False)

            # Move data across PCIe to GPU 1
            gt_g1 = gt_g0.to("cuda:1", non_blocking=True)
            b_g1 = batch.to("cuda:1", non_blocking=True)

        # Create a specific event marker for THIS EXACT BATCH
        event = torch.cuda.Event()
        event.record(stream_dino)

        # Send the GPU 1 tensors and the event marker to the ViTTT thread.
        # If the queue is full, this thread pauses, preventing OOMs.
        pipeline_queue.put((b_g1, gt_g1, event))

    # Send a sentinel value to tell the consumer the epoch is over
    pipeline_queue.put(None)


def async_train_on_dataset(name, iterator):
    # maxsize=2 acts as our double-buffer. It naturally throttles
    # the CPU, preventing the runaway OOM crash on GPU 1.
    pipeline_queue = queue.Queue(maxsize=2)

    # Start the background producer thread
    producer = threading.Thread(target=dino_producer, args = (iterator, pipeline_queue))
    producer.start()

    # ---------------------------------------------------------
    # THREAD B: THE ViTTT CONSUMER (GPU 1 - Main Thread)
    # ---------------------------------------------------------
    stream_vittt = torch.cuda.Stream(device=1)

    # Optional: If your dataloader doesn't support len(), wrap the queue extraction in a while loop
    for _ in tqdm(range(len(iterator)), desc=name):
        item = pipeline_queue.get()
        if item is None:
            break  # End of epoch

        b_g1, gt_g1, event = item

        with torch.cuda.stream(stream_vittt):
            # Wait ONLY for this specific batch's transfer to finish, nothing else
            stream_vittt.wait_event(event)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.profiler.record_function("ViTTT_train_inference"):
                    pred = ViTTT_model(b_g1)
                    loss = metric(pred, gt_g1)

            # We can use .detach() safely now because the queue size throttles us
            train_losses[epoch] += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Clean up the thread before moving to the next dataset/epoch
    producer.join()


def async_val(name, iterator):
    # maxsize=2 acts as our double-buffer. It naturally throttles
    # the CPU, preventing the runaway OOM crash on GPU 1.
    pipeline_queue = queue.Queue(maxsize=2)

    # Start the background producer thread
    producer = threading.Thread(target=dino_producer, args = (iterator, pipeline_queue))
    producer.start()

    # ---------------------------------------------------------
    # THREAD B: THE ViTTT CONSUMER (GPU 1 - Main Thread)
    # ---------------------------------------------------------
    stream_vittt = torch.cuda.Stream(device=1)

    # Optional: If your dataloader doesn't support len(), wrap the queue extraction in a while loop
    for _ in tqdm(range(len(iterator)), desc=name):
        item = pipeline_queue.get()
        if item is None:
            break  # End of epoch

        b_g1, gt_g1, event = item

        with torch.cuda.stream(stream_vittt), torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Wait ONLY for this specific batch's transfer to finish, nothing else
            stream_vittt.wait_event(event)
            # with torch.profiler.record_function("ViTTT_train_inference"):
            pred = ViTTT_model(b_g1)
            loss = metric(pred, gt_g1)

            # We can use .detach() safely now because the queue size throttles us
            val_losses[epoch] += loss.detach()

    # Clean up the thread before moving to the next dataset/epoch
    producer.join()


def checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, path = None):
    """
    Saves the training state *after* the training section (right before validation). In other words, `epoch` is the most
    recently-completed training epoch.

    Parameters
    ----------
    epoch
    model
    optimizer
    scheduler
    train_losses
    val_losses
    path
    """
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }, f"/vulcanscratch/hughma/ViT/checkpoints/Epoch {epoch}.pth" if path is None else path)
    print(f"Finished saving checkpoint for epoch {epoch}")


def initialize(epochs, checkpoint = None):
    ViTTT_model = ViTTT().to("cuda:1")

    # Muon can't optimize the convolution, which has a parameter of size (1280, 3, 16, 16); thus, I must use AdamW.
    optimizer = torch.optim.AdamW(ViTTT_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if checkpoint is None:
        # Keep everything on the GPU during training.
        train_losses, val_losses = torch.zeros((epochs), device="cuda:1"), torch.zeros((epochs), device="cuda:1")
        next_epoch = 0
    else:
        checkpoint = torch.load(checkpoint, weights_only = True)
        next_epoch = checkpoint["epoch"] + 1
        ViTTT_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_losses = checkpoint["train_losses"].to("cuda:1", non_blocking = True)
        val_losses = checkpoint["val_losses"].to("cuda:1", non_blocking = True)
    return next_epoch, ViTTT_model, optimizer, scheduler, train_losses, val_losses


def prepare_dataloaders():
    # The batch sizes are the largest I could empirically find without causing OOM.
    scannet_dataloader = torch.utils.data.DataLoader(scannet_dataset(), batch_size=8, shuffle=True, num_workers=4,
                                                     persistent_workers=True, pin_memory=True)
    dtu_dataloader = torch.utils.data.DataLoader(dtu_dataset(), batch_size=4, shuffle=True, num_workers=4,
                                                 persistent_workers=True, pin_memory=True)
    nrgbd_dataloader = torch.utils.data.DataLoader(nrgbd_dataset(), batch_size=32, shuffle=True, num_workers=4,
                                                   persistent_workers=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(sintel_dataset(), batch_size=32, shuffle=False, num_workers=4,
                                                 persistent_workers=True, pin_memory=True)
    return scannet_dataloader, dtu_dataloader, nrgbd_dataloader, val_dataloader


if __name__ == "__main__":
    epochs = 10
    checkpoint_every = 5
    checkpoint_thread = None
    jobs = [
        partial(initialize, epochs),
        load_dinov3,
        prepare_dataloaders,
    ]
    # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(job) for job in jobs]

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # torch.autograd.set_detect_anomaly(True) # DEBUG
        # torch.set_float32_matmul_precision('high')
        set_seed()
        metric = nn.MSELoss()

        # futures.as_completed returns futures in the order they complete, not their original order.
        for future in concurrent.futures.as_completed(futures):
            match futures.index(future):
                case 0:
                    next_epoch, ViTTT_model, optimizer, scheduler, train_losses, val_losses = future.result()
                case 1:
                    processor, dino = future.result()
                    dino = dino.to("cuda:0").eval()
                case 2:
                    scannet_dataloader, dtu_dataloader, nrgbd_dataloader, val_dataloader = future.result()

    # Immediately start loading the data.
    scannet_iter = iter(scannet_dataloader)
    dtu_iter = iter(dtu_dataloader)
    nrgbd_iter = iter(nrgbd_dataloader)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,  # the cpu activities
            torch.profiler.ProfilerActivity.CUDA,  # the gpu activities
        ],
        # schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    ) as prof:
        # Training loop
        for epoch in range(next_epoch, epochs):
            # Start pre-loading validation data as early as possible.
            val_iter = iter(val_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}:")
            ViTTT_model.train()

             # Wait for checkpointing to complete
            if checkpoint_thread is not None:
                checkpoint_thread.join()
                checkpoint_thread = None # Saves us from entering this if statement until the next checkpoint

            async_train_on_dataset("Scannetv2", scannet_iter)
            async_train_on_dataset("DTU", dtu_iter)
            async_train_on_dataset("NRGBD", nrgbd_iter)

            """
            Asynchronous checkpointing as soon as training completes, giving us the maximum amount of time before we must
            wait for it to finish (so we don't optimizer.step() while trying to save the model).
            I'm not saving the randomizer states because that would mean we can't even create the next train dataloaders
            until we're done saving, meaning we can't move on at all, meaning the save wouldn't actually be asynchronous.
            """
            if epoch % checkpoint_every == 0 and epoch > 0 and epoch < epochs - 1:
                checkpoint_thread = threading.Thread(target=checkpoint, args=(
                    epoch, ViTTT_model, optimizer, scheduler, train_losses, val_losses
                ))
                checkpoint_thread.start()

            # Start pre-loading the training data as early as possible.
            if epoch != epochs - 1:
                scannet_iter = iter(scannet_dataloader)
                dtu_iter = iter(dtu_dataloader)
                nrgbd_iter = iter(nrgbd_dataloader)

            # Validation on Sintel
            ViTTT_model.eval()
            # with torch.no_grad(), torch.profiler.record_function("val_inference"):
            with torch.no_grad():
                async_val("Validation (Sintel)", val_iter)

    # End of training loop; write losses to file
    print("Training done.")
    import pickle
    with open("train_losses.pkl", "wb") as f:
        pickle.dump(train_losses.numpy(force=True), f)
    print("Saved train_losses")
    with open("val_losses.pkl", "wb") as f:
        pickle.dump(val_losses.numpy(force=True), f)
    print("Saved val_losses")
    torch.save(ViTTT_model.state_dict(), "ViTTT.pth")
    print("Saved model")
    # prof.export_chrome_trace(f"trace.json")
    # print("Saved trace")