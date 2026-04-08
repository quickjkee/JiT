import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr, create_dataloader
import util.misc as misc
import util.dist as dist

import copy
from engine_jit import train_one_epoch, evaluate, evaluate_linear_probing
from overfit_experiment import run_overfit
from denoiser import Denoiser

def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train') # Two families: ['JiT-B/16', ...]; ['DinoJiT-B/16', ...]
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--dino_trained_path', type=str)
    parser.add_argument('--dino_init_path', type=str)
    parser.add_argument('--in_context_len', default=32, type=int)
    parser.add_argument('--reg_len', default=0, type=int)
    parser.add_argument('--in_context_start', default=4, type=int)

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--yt_config_path', default='configs/imagenet_yt_config.yaml', type=str,
                        help='Path to the config of imagenet dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)

    # overfit experiment
    parser.add_argument('--overfit', action='store_true', help='Run tiny overfit experiment instead of full training')
    parser.add_argument('--overfit_n', type=int, default=16, help='Number of images in overfit subset')
    parser.add_argument('--overfit_steps', type=int, default=3000, help='Number of optimizer steps for overfit run')
    parser.add_argument('--overfit_t', type=float, default=0.8, help='Fixed timestep for overfit test')
    parser.add_argument('--overfit_batch_size', type=int, default=16, help='Batch size for overfit run')
    parser.add_argument('--overfit_print_freq', type=int, default=50, help='Print/log every N steps')
    parser.add_argument('--overfit_use_ema', action='store_true', help='Update EMA during overfit (off by default)')
    parser.add_argument('--overfit_save_imgs', action='store_true', help='Save reconstructions during overfit')
    parser.add_argument('--overfit_img_freq', type=int, default=200, help='Save images every N steps')

    return parser


def _verify_checksum(name: str, tensors, device, global_rank: int):
    """Allgather a scalar checksum from every rank and assert all ranks agree.

    Works across all GPUs on all hosts (uses the full NCCL process group).
    Reports mismatching ranks as (global_rank, host_idx, local_gpu) triples.
    """
    checksum = torch.tensor(0.0, device=device)
    for t in tensors:
        if isinstance(t, torch.Tensor):
            checksum += t.to(device=device, dtype=torch.float32).sum()

    all_checksums = dist.allgather(checksum.unsqueeze(0))  # [world_size]
    max_diff = (all_checksums - all_checksums[0]).abs().max().item()

    if global_rank == 0:
        num_ranks = all_checksums.shape[0]
        gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
        if max_diff == 0.0:
            print(f"{name} checksum PASSED: all {num_ranks} ranks agree.")
        else:
            lines = []
            for r, cs in enumerate(all_checksums.tolist()):
                host_idx = r // gpus_per_node
                local_gpu = r % gpus_per_node
                marker = " <-- MISMATCH" if abs(cs - all_checksums[0].item()) > 0 else ""
                lines.append(f"  rank {r:3d} (host {host_idx}, gpu {local_gpu}): {cs:.6e}{marker}")
            raise RuntimeError(
                f"{name} mismatch across ranks!\n"
                f"  Max absolute diff: {max_diff:.6e}\n"
                + "\n".join(lines)
            )
    dist.barrier()


def init_distributed_mode(timeout=30):
    try:
        dist.initialize()
        dist.barrier()
    except RuntimeError:
        print(f'{">" * 75}  NCCL Error  {"<" * 75}', flush=True)
        time.sleep(timeout)


def main(args):
    init_distributed_mode()
    misc.setup_for_distributed(dist.is_master())
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = dist.get_local_rank()
    torch.cuda.set_device(device)

    # Set seeds for reproducibility
    global_rank = dist.get_rank()
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    num_tasks = dist.get_world_size()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    if os.path.exists(args.data_path):
        transform_train = transforms.Compose([
                            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.PILToTensor()
                            ])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", sampler_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=True,
        )
    else:
        data_loader_train = create_dataloader(args.yt_config_path, args.batch_size)


    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * num_tasks
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    # Load checkpoint on rank 0 only, then broadcast to all ranks
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    checkpoint = {}
    if global_rank == 0 and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if global_rank == 0 and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])

    # Broadcast model params and buffers from rank 0 to all ranks
    dist.barrier()
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
    dist.barrier()

    _verify_checksum("Model params+buffers",
                     list(model.parameters()) + list(model.buffers()),
                     device, global_rank)

    model = DDP(model, device_ids=[device])
    model_without_ddp = model.module

    # Initialize EMA params on all ranks (from broadcast model params)
    model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
    model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))

    if global_rank == 0 and 'model_ema1' in checkpoint:
        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        print("Resumed EMA from checkpoint at", args.resume)

    # Broadcast EMA params from rank 0
    for param in model_without_ddp.ema_params1:
        dist.broadcast(param.data, src=0)
    for param in model_without_ddp.ema_params2:
        dist.broadcast(param.data, src=0)
    dist.barrier()

    _verify_checksum("EMA1 params", model_without_ddp.ema_params1, device, global_rank)
    _verify_checksum("EMA2 params", model_without_ddp.ema_params2, device, global_rank)

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    if global_rank == 0 and 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded optimizer & epoch state!")

    # Broadcast optimizer state and start_epoch from rank 0
    obj_list = [optimizer.state_dict()]
    dist.broadcast_object_list(obj_list, src=0)
    optimizer.load_state_dict(obj_list[0])

    start_epoch_tensor = torch.tensor(args.start_epoch, dtype=torch.long, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    args.start_epoch = start_epoch_tensor.item()

    dist.barrier()

    # Verify optimizer states are identical across all ranks after broadcast
    opt_tensors = [v for group in optimizer.state_dict()['state'].values()
                   for v in group.values()]
    _verify_checksum("Optimizer state", opt_tensors, device, global_rank)

    if global_rank == 0 and not checkpoint:
        print("Training from scratch")
    del checkpoint

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Toy overfit experiment
    if args.overfit:
        if dist.is_master():
            print("Running OVERFIT experiment (tiny subset) ...")

        run_overfit(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            device=device,
            log_writer=log_writer
        )
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if num_tasks > 1 and os.path.exists(args.data_path):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            if 'Dino' in args.model:
                evaluate_linear_probing(model_without_ddp.net, args, device=device)
            torch.cuda.empty_cache()

        if dist.is_master() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
