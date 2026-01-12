# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

from util.model_util import RMSNorm
from model_jit import JiTBlock, VisionRotaryEmbeddingFast
from collections import OrderedDict
from omegaconf import OmegaConf

from yrDinoV2.dinov2.configs import dinov2_default_config
from yrDinoV2.dinov2.models import build_model_from_cfg

logger = logging.getLogger("dinov2")




########################################################################################################################
#                                              MODEL DEFINITIONS                                                       #
########################################################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    #@torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class DinoJiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        dino_model,
        patch_size=16,
        num_classes=1000,
        input_size=256,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        depth=12,
        num_heads=16,
        in_context_len=32,
        bottleneck_dim=128,
        in_context_start=8,
    ):
        super().__init__()

        self.hidden_size = dino_model.embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.out_channels = 3

        self.dino_model = dino_model
        self.dino_model.requires_grad_(True)
        #for block in dino_model.blocks[-6:]:
        #    for p in block.parameters():
        #        p.requires_grad = True
        #self.dino_model.norm.requires_grad_(True)

        # time and class embed
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.hidden_size)

        # rope
        half_head_dim = self.hidden_size // num_heads // 2
        hw_seq_len = self.input_size // self.patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=1
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=1 + self.in_context_len
        )
        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, self.hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # decoder
        self.decoder_blocks = nn.ModuleList([
                JiTBlock(self.hidden_size, num_heads, mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                        proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
                for i in range(depth)
            ])
        self.final_layer = FinalLayer(self.hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) \
            and any(p.requires_grad for p in module.parameters(recurse=False)) \
            and not any(module is m for m in self.dino_model.modules()):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, drop_mid=False):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # Start embedding
        # -----------------------------------------
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        # -----------------------------------------

        # Encoder part
        # -----------------------------------------
        x = self.dino_model.forward_features(x, class_idxs=y, t=1 - t, noise=1)
        x_cls, x = x["x_norm_clstoken"], x["x_norm_patchtokens"]
        x = torch.cat([x_cls, x], dim=1)
        if drop_mid:
            return x_cls
        # -----------------------------------------

        # Decoder part
        # -----------------------------------------
        for i, block in enumerate(self.decoder_blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, 1 + self.in_context_len:]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        # -----------------------------------------

        return output


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu", weights_only=False)['teacher']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print('loaded')
    return model
    

def DinoJiT_B_16(dino_trained_path='checkpoints/eval/training_240000/teacher_checkpoint.pth',
                 dino_init_path='checkpoints/dinov2_vitb14_pretrain.pth',
                 **kwargs):
    default_cfg = OmegaConf.create(dinov2_default_config)
    config_file = 'configs/vitb14_noisy_pretrained_dinov2_low_lr.yaml'
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg) 
    cfg.MODEL.PRETRAINED = dino_init_path

    _, teacher_backbone, _ = build_model_from_cfg(cfg)
    dino_model = load_checkpoint(teacher_backbone, dino_trained_path)
    dinojit = DinoJiT(dino_model=dino_model, depth=8, num_heads=12,
                      in_context_len=32, in_context_start=4, patch_size=14, **kwargs)
    return dinojit


DinoJiT_models = {
    'DinoJiT-B/16': DinoJiT_B_16,
}