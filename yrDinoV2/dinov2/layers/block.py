# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Optional
import warnings
from functools import partial

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")



class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        diffusion_mode: bool = False
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        
        self.diffusion_mode = diffusion_mode
        if diffusion_mode:
            # AdaLN
            self.adaLN = nn.Sequential(
                nn.SiLU(inplace=False), 
                nn.Linear(dim, 6 * dim, bias=True)
            )
            
            
    def forward(self, x: Tensor, c: Optional[Tensor] = None, w: Optional[Tensor] = None) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        # AdaLN variant
        if self.diffusion_mode:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                (w * self.adaLN(c).unsqueeze(1)).chunk(6, dim=-1)
            )
            gamma1 = gamma1 + 1.0
            gamma2 = gamma2 + 1.0

            # ones = torch.ones_like(x, dtype=x.dtype, device=x.device)
            # gamma1, gamma2 = ones, ones
        else:
            ones = torch.ones_like(x, dtype=x.dtype, device=x.device)
            zeros = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = ones, ones, zeros, zeros, zeros, zeros
            
        def adaln_attn_residual_func(x: Tensor, scale: Tensor, shift: Tensor, gamma: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x).mul(scale + 1).add(shift)).mul(gamma))
                            
        def adaln_ffn_residual_func(x: Tensor, scale: Tensor, shift: Tensor, gamma: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x).mul(scale + 1).add(shift)).mul(gamma))
        
        adaln_attn_residual_func = partial(adaln_attn_residual_func, scale=scale1, shift=shift1, gamma=gamma1)
        adaln_ffn_residual_func = partial(adaln_ffn_residual_func, scale=scale2, shift=shift2, gamma=gamma2)
        
        attn_residual_func = adaln_attn_residual_func if self.diffusion_mode else attn_residual_func
        ffn_residual_func = adaln_ffn_residual_func if self.diffusion_mode else ffn_residual_func
        
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = Attention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
        
    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def cat_gammas_scales_shifts(gamma_list, scale_list, shift_list, branges=None):
    
    if branges is not None:
        gamma_cat = index_select_cat([x.flatten(1) for x in gamma_list], branges).view(1, -1, gamma_list[0].shape[-1])
        scale_cat = index_select_cat([x.flatten(1) for x in scale_list], branges).view(1, -1, scale_list[0].shape[-1])
        shift_cat = index_select_cat([x.flatten(1) for x in shift_list], branges).view(1, -1, shift_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(gamma.reshape([1, -1, *gamma.shape[2:]]) for gamma in gamma_list)
        gamma_cat = torch.cat(tensors_bs1, dim=1)
        
        tensors_bs1 = tuple(scale.reshape([1, -1, *scale.shape[2:]]) for scale in scale_list)
        scale_cat = torch.cat(tensors_bs1, dim=1)
        
        tensors_bs1 = tuple(shift.reshape([1, -1, *shift.shape[2:]]) for shift in shift_list)
        shift_cat = torch.cat(tensors_bs1, dim=1)
    
    return gamma_cat, scale_cat, shift_cat


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    gamma_list, scale_list, shift_list,
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    gamma_cat, scale_cat, shift_cat = cat_gammas_scales_shifts(gamma_list, scale_list, shift_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, gamma_cat, scale_cat, shift_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor], c: List[Tensor] = None, w: List[Tensor] = None) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        print('hui')
            
        # AdaLN variant
        if self.diffusion_mode:
            gamma1_list, gamma2_list = [], []
            scale1_list, scale2_list = [], []
            shift1_list, shift2_list = [], []
            
            # rename
            c_list = c
            w_list = w
            for x, c, w in zip(x_list, c_list, w_list):
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                    (w * self.adaLN(c).unsqueeze(1)).chunk(6, dim=-1)
                )
                gamma1 = gamma1 + 1.0
                gamma2 = gamma2 + 1.0
                 
                gamma1_list.append(gamma1.expand(*x.shape))
                gamma2_list.append(gamma2.expand(*x.shape))
                scale1_list.append(scale1.expand(*x.shape))
                scale2_list.append(scale2.expand(*x.shape))
                shift1_list.append(shift1.expand(*x.shape))
                shift2_list.append(shift2.expand(*x.shape))
                
            # ones = [torch.ones_like(x, dtype=x.dtype, device=x.device) for x in x_list]
            # zeros = [torch.zeros_like(x, dtype=x.dtype, device=x.device) for x in x_list]
            # gamma1_list, gamma2_list = ones, ones
        else:
            ones = [torch.ones_like(x, dtype=x.dtype, device=x.device) for x in x_list]
            zeros = [torch.zeros_like(x, dtype=x.dtype, device=x.device) for x in x_list]
            gamma1_list, gamma2_list, scale1_list, scale2_list, shift1_list, shift2_list = ones, ones, zeros, zeros, zeros, zeros
                
        
        if self.training and self.sample_drop_ratio > 0.0:
            
            def attn_residual_func(x: Tensor, gamma: Tensor, scale: Tensor, shift: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x).mul(scale + 1).add(shift), attn_bias=attn_bias).mul(gamma)
                                
            def ffn_residual_func(x: Tensor, gamma: Tensor, scale: Tensor, shift: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x).mul(scale + 1).add(shift)).mul(gamma)
            
            x_list = drop_add_residual_stochastic_depth_list(
                x_list, gamma1_list, scale1_list, shift1_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list, gamma2_list, scale2_list, shift2_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, gamma: Tensor, scale: Tensor, shift: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x).mul(scale + 1).add(shift), attn_bias=attn_bias).mul(gamma))
                                
            def ffn_residual_func(x: Tensor, gamma: Tensor, scale: Tensor, shift: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x).mul(scale + 1).add(shift)).mul(gamma))
            
            attn_bias, x = get_attn_bias_and_cat(x_list)
            gamma1, scale1, shift1 = cat_gammas_scales_shifts(gamma1_list, scale1_list, shift1_list)
            gamma2, scale2, shift2 = cat_gammas_scales_shifts(gamma2_list, scale2_list, shift2_list)
            
            x = x + attn_residual_func(x, gamma1, scale1, shift1, attn_bias=attn_bias)
            x = x + ffn_residual_func(x, gamma2, scale2, shift2)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, **kwargs):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, **kwargs)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, **kwargs)
        else:
            raise AssertionError
