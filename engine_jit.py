import math
import sys
import os
import shutil
import numpy as np

import torch
import cv2
import util.dist as dist
import util.misc as misc
import util.lr_sched as lr_sched
import copy

from util.fid import calculate_fid


def unpack_batch(batch, device, args):
    if os.path.exists(args.data_path):
        x, y = batch
    else:
        x = batch['image']
        y = torch.tensor(batch['label'])
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
    x = x * 2.0 - 1.0
    y = y.to(device, non_blocking=True)
    return x, y


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, run=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    print(len(data_loader))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x, labels = unpack_batch(batch, device, args=args)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if data_iter_step % args.log_freq == 0:
            if run is not None:
                run.log({
                    'train_loss': loss_value_reduce,
                    'lr': lr
                })

        if data_iter_step >= len(data_loader):
            break


@torch.no_grad()
def evaluate(model_without_ddp, args, batch_size=32, run=None):

    model_without_ddp.eval()
    local_world_size = torch.cuda.device_count()
    local_rank = dist.get_local_rank()
    num_steps = args.num_images // (batch_size * local_world_size) + 1
    
    print(local_rank, local_world_size, num_steps)

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    dist.barrier()
    
    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = local_world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        # denormalize images 
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * local_world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    dist.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if args.img_size == 256 or args.img_size == 224:
        fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
    elif args.img_size == 512:
        fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
    else:
        raise NotImplementedError
    fid = calculate_fid(save_folder, fid_statistics_file, inception_path='fid_stats/pt_inception-2015-12-05-6726825d.pth')
    postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
    if run is not None:
        run.log({f'fid{postfix}': fid})
        
    print("FID: {:.4f}".format(fid))
    if dist.is_local_master():
        shutil.rmtree(save_folder)

    dist.barrier()
    model_without_ddp.train()
    return
