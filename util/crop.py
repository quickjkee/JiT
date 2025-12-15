import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image


class TwoResWrapper(torch.utils.data.Dataset):
    """
    Returns (x224_for_dino, x256_target, y) from the SAME underlying image
    with a single shared random horizontal flip decision.
    """
    def __init__(self, base_ds, img_size=256, flip_p=0.5):
        self.base = base_ds
        self.img_size = img_size
        self.flip_p = flip_p

        self.to_tensor = transforms.PILToTensor()
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]  # PIL image

        # shared randomness (one flip decision for BOTH views)
        if random.random() < self.flip_p:
            img = TF.hflip(img)

        # make 256 view (PIL -> center crop -> tensor float in [0,1])
        img256 = center_crop_arr(img, self.img_size)                 # PIL
        x256 = self.to_tensor(img256).float().div(255.0)             # [3,256,256], float

        # make 224 view for DINO (derived from x256 => perfectly aligned)
        x224 = F.interpolate(
            x256.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False
        ).squeeze(0)                                                 # [3,224,224]
        x224 = self.norm(x224)                                       # DINO-style normalize

        return x224, x256, y


def crop_dinov2(x, resolution): 
    x = x / 255. 
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x) 
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic') 
    return x


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


def load_ds_train(img_size, ds, flip_p=0.5):
    """
    img_size: target resolution for diffusion supervision (typically 256)
    ds: a dataset like ImageFolder with transform=None returning (PIL, label)

    returns dataset that yields:
      x224_for_dino: [3,224,224] normalized
      x256_target:   [3,img_size,img_size] float in [0,1]
      y:             int label
    """
    return TwoResWrapper(ds, img_size=img_size, flip_p=flip_p)



