# overfit_experiment.py
import os
import time
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from util.crop import center_crop_arr
import util.misc as misc


def _unpack_batch_normalized(batch, device):
    """
    Returns x in ImageNet-normalized space (float32), y long.
    Matches your normalized-space training path.
    """
    x, y = batch
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255.0)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def _save_original_images(x_norm, outdir, max_n=16):
    """
    Save original (clean) images from normalized space.
    x_norm: ImageNet-normalized images [B,3,H,W]
    """
    os.makedirs(outdir, exist_ok=True)

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x_norm.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x_norm.device).view(1, 3, 1, 1)

    imgs = (x_norm * std + mean).clamp(0, 1).cpu()

    import imageio.v2 as imageio
    n = min(imgs.size(0), max_n)
    for i in range(n):
        im = (imgs[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(outdir, f"000_{i:02d}.png"), im)


@torch.no_grad()
def _save_debug_images(x_pred_norm, step, outdir, max_n=16):
    """
    x_pred_norm: ImageNet-normalized images in [B,3,H,W] (float)
    Saves PNGs after denorm to [0,1].
    """
    os.makedirs(outdir, exist_ok=True)
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x_pred_norm.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x_pred_norm.device).view(1, 3, 1, 1)
    imgs = (x_pred_norm * std + mean).clamp(0, 1).cpu()

    # Save individual images (no cv2 dependency needed here)
    import imageio.v2 as imageio
    n = min(imgs.size(0), max_n)
    for i in range(n):
        im = (imgs[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(outdir, f"{step:07d}_{i:02d}.png"), im)


def run_overfit(args, model, model_without_ddp, optimizer, device, log_writer=None):
    """
    Overfit experiment:
      - tiny subset (args.overfit_n images)
      - fixed timestep t = args.overfit_t
      - label_drop_prob=0
      - disables EMA update (optional)
    """
    assert args.img_size in (224, 256, 512), "Adjust if you use other sizes."

    # Force settings that make overfit easier / more diagnostic
    args.label_drop_prob = 0.0
    model_without_ddp.label_drop_prob = 0.0

    # Data: deterministic tiny subset
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.PILToTensor()
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)

    n = min(args.overfit_n, len(dataset))
    idx = list(range(n))
    tiny = Subset(dataset, idx)

    # DDP sampler: fixed subset but shuffled each epoch
    num_tasks = misc.get_world_size()
    rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        tiny, num_replicas=num_tasks, rank=rank, shuffle=True, drop_last=True
    )

    loader = DataLoader(
        tiny,
        batch_size=min(args.overfit_batch_size, n),
        sampler=sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
    )

    # Monkeypatch fixed-t forward for the overfit run
    # Keeps everything else identical.
    fixed_t = float(args.overfit_t)

    def forward_fixed_t(x, labels):
        labels_dropped = labels  # no label dropout
        t = torch.full((x.size(0), 1, 1, 1), fixed_t, device=x.device, dtype=x.dtype)

        e = torch.randn_like(x) * model_without_ddp.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(model_without_ddp.t_eps)

        # network predicts x in normalized space
        x_pred = model_without_ddp.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(model_without_ddp.t_eps)

        loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        return loss, x_pred

    # Training loop: run by steps (more convenient than epochs)
    model.train(True)
    steps = int(args.overfit_steps)
    print_freq = int(args.overfit_print_freq)

    # Optional: disable EMA updates for clarity
    do_ema = bool(args.overfit_use_ema)

    start = time.time()
    step = 0
    # Create an iterator that never ends
    while step < steps:
        sampler.set_epoch(step // max(1, len(loader)))  # reshuffle occasionally

        for batch in loader:
            x, y = _unpack_batch_normalized(batch, device)
            if step == 0 and args.overfit_save_imgs and misc.is_main_process():
                _save_original_images(
                    x_norm=x,
                    outdir=os.path.join(args.output_dir, "overfit_debug", "originals"),
                    max_n=16
                )


            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, x_pred = forward_fixed_t(x, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if do_ema:
                model_without_ddp.update_ema()

            if misc.is_main_process() and (step % print_freq == 0 or step == steps - 1):
                elapsed = time.time() - start
                print(f"[overfit] step {step:6d}/{steps}  loss {loss.item():.6f}  t={fixed_t}  time={elapsed:.1f}s")

                if log_writer is not None:
                    log_writer.add_scalar("overfit/loss", loss.item(), step)

                if args.overfit_save_imgs and (step % args.overfit_img_freq == 0):
                    _save_debug_images(
                        x_pred_norm=x_pred.detach(),
                        step=step,
                        outdir=os.path.join(args.output_dir, "overfit_debug"),
                        max_n=16
                    )

            step += 1
            if step >= steps:
                break

    if misc.is_main_process():
        print("Overfit done. Debug images (if enabled) are in:", os.path.join(args.output_dir, "overfit_debug"))
