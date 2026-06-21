import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import re

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr, create_dataloader
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate, evaluate_linear_probing
from overfit_experiment import run_overfit
from denoiser import Denoiser


def train_only_last_jit_part(model, last_n_blocks=2, train_in_context_posemb=False):
    """
    Works for:
    - Denoiser wrapper with model.net = JiT
    - raw JiT model
    """

    # Freeze everything in the full model first
    for name, p in model.named_parameters():
        p.requires_grad = False

    # Your training code suggests Denoiser has .net
    jit = model.net if hasattr(model, "net") else model

    assert hasattr(jit, "blocks"), "Could not find jit.blocks"
    assert hasattr(jit, "final_layer"), "Could not find jit.final_layer"

    depth = len(jit.blocks)
    start_block = max(0, depth - last_n_blocks)

    # Unfreeze last N transformer blocks
    for i in range(start_block, depth):
        for p in jit.blocks[i].parameters():
            p.requires_grad = True

    # Unfreeze final prediction layer
    for p in jit.final_layer.parameters():
        p.requires_grad = True

    # Optional: also train in-context positional tokens
    if train_in_context_posemb and hasattr(jit, "in_context_posemb"):
        jit.in_context_posemb.requires_grad = True

    print(f"Training JiT blocks [{start_block}, ..., {depth - 1}] + final_layer")

    for name, p in model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {n_trainable / 1e6:.3f}M / {n_total / 1e6:.3f}M")


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
    parser.add_argument('--in_context_end', default=100, type=int)

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
    parser.add_argument('--p_drop_registers', default=0.05, type=float)

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
    parser.add_argument('--rg_scale', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--rg_alone_scale', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--cfg_rg_scale', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--interval_min_rg', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max_rg', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--interval_min_rg_alone', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max_rg_alone', default=1.0, type=float,
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
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

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


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

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

    # Train only last part of JiT
    train_only_last_jit_part(
        model.net,
        last_n_blocks=1,
        train_in_context_posemb=False,
    )

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

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
        if misc.is_main_process():
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
        if args.distributed and os.path.exists(args.data_path):
            data_loader_train.sampler.set_epoch(epoch)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, 
                         forward_fn_type='cfg')
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, 
                         forward_fn_type='rg')
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, 
                         forward_fn_type='cfg_rg')
            if 'Dino' in args.model:
                evaluate_linear_probing(model_without_ddp.net, args, device=device)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
