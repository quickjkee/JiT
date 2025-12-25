import math
import sys
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
import util.lr_sched as lr_sched
import copy

from util.fid import calculate_fid
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm



def unpack_batch(batch, device, case='JiT'):
    x, y = batch
    x.to(torch.float32)
    x = x / 255. 
    x = x * 2.0 - 1.0
    y = y.to(device, non_blocking=True).long()
    return x, y


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x, labels = unpack_batch(batch, device, case=args.model)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels, 
                         repa_coeff=args.repa_coeff)

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

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

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

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        fid = calculate_fid(save_folder, fid_statistics_file, inception_path='fid_stats/pt_inception-2015-12-05-6726825d.pth')
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        print("FID: {:.4f}".format(fid))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()


def evaluate_linear_probing(model, args, device):

    @torch.no_grad()
    def extract_features(model, loader, device, t = 1.0):
        model.eval()
        feats, labels = [], []

        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, non_blocking=True)
            x = x / 255.
            x = x * 2.0 - 1.0

            e = torch.randn_like(x) 
            z = t * x + (1 - t) * e
            t_ = torch.tensor([t]).repeat(x.size(0)).flatten().cuda()

            _, cls_ = model(z, t=t_, y=y) 
            cls = F.normalize(cls_, dim=1)

            feats.append(cls)
            labels.append(y)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels

    def make_subset(dataset, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
        return Subset(dataset, idx)

    def center_crop_arr(pil_image, image_size):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


    transform_train = transforms.Compose([
                          transforms.Lambda(lambda img: center_crop_arr(img, 256)),
                          transforms.PILToTensor()
                        ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    subset_train = make_subset(dataset_train, n=20000, seed=0)  # pick 5k/10k/20k
    subset_val = make_subset(dataset_train, n=5000, seed=1)  # pick 5k/10k/20k

    train_loader = DataLoader(
                    subset_train,
                    batch_size=512,
                    shuffle=False,
                    num_workers=8,
                )

    val_loader = DataLoader(
        subset_val,
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )

    t = [1.0, 0.8, 0.6, 0.4, 0.2]
    for t_ in t:
        Xtr, Ytr = extract_features(model, train_loader, device, t=t_)
        Xva, Yva = extract_features(model, val_loader, device, t=t_)
        
        Xtr = F.normalize(Xtr, dim=1)
        Xva = F.normalize(Xva, dim=1)
        num_classes = int(Ytr.max().item() + 1)
        print(f'Num classes {num_classes}, {t_}')
        
        clf = nn.Linear(Xtr.shape[1], num_classes).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(20):
            clf.train()
            # simple full-batch training; for big datasets, use a DataLoader over (Xtr, Ytr)
            logits = clf(Xtr)
            loss = criterion(logits, Ytr)
        
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            clf.eval()
            with torch.no_grad():
                val_acc = (clf(Xva).argmax(1) == Yva).float().mean().item()
            print(f"epoch {epoch:02d} loss {loss.item():.4f} val_acc {val_acc*100:.2f}%")

    torch.distributed.barrier()