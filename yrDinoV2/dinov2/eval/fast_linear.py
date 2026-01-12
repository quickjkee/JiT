import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

def evaluate_linear_probing(model, cfg, device):

    # utils fns
    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(model, loader, device, t=None):
        model.eval()
        model = model.to(device).float() 
        feats, labels = [], []

        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, non_blocking=True)
            
            xb = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model.forward_features(xb, class_idxs=y, t=t)
            cls = out["x_norm_clstoken"]
            cls = F.normalize(cls, dim=1)

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
    # --------------------------------------------------------------------------------

    transform_train = transforms.Compose([
        transforms.Resize(256),
         transforms.CenterCrop(224),
        # transforms.Lambda(lambda img: center_crop_arr(img, 256)),
        transforms.ToTensor()
    ])
    dataset_train = datasets.ImageFolder(cfg.train.dataset_path, transform=transform_train)

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

    ts = [0.0, 0.4, 0.8, 0.95]
    accuracies = {}
    for t in ts:
        Xtr, Ytr = extract_features(model, train_loader, device, t=t)
        Xva, Yva = extract_features(model, val_loader, device, t=t)
        
        Xtr = F.normalize(Xtr, dim=1)
        Xva = F.normalize(Xva, dim=1)
        num_classes = int(Ytr.max().item() + 1)
        print(f'Num classes {num_classes}, {t=}')
        
        clf = nn.Linear(Xtr.shape[1], num_classes).to(device)
        clf.train()
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(20):
            # simple full-batch training; for big datasets, use a DataLoader over (Xtr, Ytr)
            logits = clf(Xtr)
            loss = criterion(logits, Ytr)
        
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            with torch.no_grad():
                val_acc = (clf(Xva).argmax(1) == Yva).float().mean().item()
                accuracies[f"val_acc_{t}"] = val_acc
            print(f"epoch {epoch:02d} loss {loss.item():.4f} val_acc {val_acc*100:.2f}%")
    return accuracies

