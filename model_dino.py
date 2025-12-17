# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import os, torch, torch.distributed as dist

from util.model_util import RMSNorm
from model_jit import JiTBlock, VisionRotaryEmbeddingFast
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

logger = logging.getLogger("dinov2")




########################################################################################################################
#                                              MODEL DEFINITIONS                                                       #
########################################################################################################################


def modulate(x, shift, scale):
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x

def gate(x, gate_):
    x = gate_.unsqueeze(1) * x
    return x


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


class BlockWithAdaLN(nn.Module):
    def __init__(self, blk, dim, cond_dim):
        super().__init__()

        # Reuse the exact submodules/attrs from blk
        self.norm1 = blk.norm1
        self.attn  = blk.attn
        self.ls1   = blk.ls1
        self.drop_path1 = blk.drop_path1

        self.norm2 = blk.norm2
        self.mlp   = blk.mlp
        self.ls2   = blk.ls2

        self.sample_drop_ratio = getattr(blk, "sample_drop_ratio", 0.0)

        # AdaLN params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c=None):
        B, N, D = x.shape

        if c is None:
            zeros = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            ones  = torch.ones(B, D, device=x.device, dtype=x.dtype)
            shift_msa = zeros; scale_msa = zeros; gate_msa = ones
            shift_mlp = zeros; scale_mlp = zeros; gate_mlp = ones
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=-1)

        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        attn_out = gate(attn_out, gate_msa)
        x = x + self.drop_path1(self.ls1(attn_out))

        mlp_out  = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        mlp_out = gate(mlp_out, gate_mlp)
        x = x + self.drop_path1(self.ls2(mlp_out)) 
        
        return x



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
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
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
        in_context_start=8
    ):
        super().__init__()

        self.dino_model = dino_model
        self.dino_model.requires_grad_(False)

        self.hidden_size = self.dino_model.embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.out_channels = 3

        # time and class embed
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.hidden_size)

        self.encoder_blocks = nn.ModuleList([BlockWithAdaLN(b, self.hidden_size, self.hidden_size) for b in self.dino_model.blocks])

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, self.hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        half_head_dim = self.hidden_size // num_heads // 2
        hw_seq_len = self.input_size // self.patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=in_context_len
        )

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
            if isinstance(module, nn.Linear) and any(p.requires_grad for p in module.parameters()):
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

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        x = F.interpolate(
            x, size=(224, 224), mode="bicubic", align_corners=False
        )
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)

        # class and time embeddings
        if t is not None and y is not None:
            t_emb = self.t_embedder(t)
            y_emb = self.y_embedder(y)
            c = t_emb + y_emb
        else:
            c = None
        x = self.dino_model.prepare_tokens_with_masks(x, None)


        for _, block in enumerate(self.encoder_blocks):
            x = block(x, c)

        x = self.dino_model.norm(x)
        cls = x[:, 0] 
        x = x[:, 1 + self.dino_model.num_register_tokens :]

        if t is None and y is None:
            return x, cls

        for i, block in enumerate(self.decoder_blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output


def DinoJiT_B_16(**kwargs):
    dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True, force_reload=False)
    return DinoJiT(dino_model=dinov2_vitb14, depth=12, num_heads=12,
                   in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


DinoJiT_models = {
    'DinoJiT-B/16': DinoJiT_B_16,
}