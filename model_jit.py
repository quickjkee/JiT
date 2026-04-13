# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm, RMSNormSplit


def split_mod(mod):
    return mod.chunk(6, dim=-1)  # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

def apply_mod(x, shift, scale, shift_reg=None, scale_reg=None, split_point=0):
    if split_point <= 0:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    x_reg = x[:, :split_point]
    x_main = x[:, split_point:]

    x_reg = x_reg * (1 + scale_reg.unsqueeze(1)) + shift_reg.unsqueeze(1)
    x_main = x_main * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return torch.cat([x_reg, x_main], dim=1)

def apply_gate(x, gate, gate_reg=None, split_point=0):
    if split_point <= 0:
        return gate.unsqueeze(1) * x

    x_reg = x[:, :split_point]
    x_main = x[:, split_point:]

    x_reg = gate_reg.unsqueeze(1) * x_reg
    x_main = gate.unsqueeze(1) * x_main
    return torch.cat([x_reg, x_main], dim=1)


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


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, split_point, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNormSplit(head_dim, eps=1e-6, split_point=split_point, seq_dim=2) \
            if split_point > 0 else RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNormSplit(head_dim, eps=1e-6, split_point=split_point, seq_dim=2) \
            if split_point > 0 else RMSNorm(head_dim, eps=1e-6)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class SplitSwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        split_point: int = 0,
        shared_w12_reg: nn.Module = None,
        shared_w3_reg: nn.Module = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)

        self.split_point = split_point
        self.ffn_dropout = nn.Dropout(drop)

        if split_point > 0:
            self.w12_reg = shared_w12_reg if shared_w12_reg is not None else nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3_reg = shared_w3_reg if shared_w3_reg is not None else nn.Linear(hidden_dim, dim, bias=bias)

            self.w12_main = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3_main = nn.Linear(hidden_dim, dim, bias=bias)
        else:
            self.w12_reg = None
            self.w3_reg = None
            self.w12_main = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w3_main = nn.Linear(hidden_dim, dim, bias=bias)

    def forward_branch(self, x, w12, w3):
        x12 = w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.ffn_dropout(hidden)
        return w3(hidden)

    def forward(self, x):
        if self.split_point > 0:
            x_reg = x[:, :self.split_point]
            x_main = x[:, self.split_point:]

            y_reg = self.forward_branch(x_reg, self.w12_reg, self.w3_reg)
            y_main = self.forward_branch(x_main, self.w12_main, self.w3_main)

            return torch.cat([y_reg, y_main], dim=1)

        return self.forward_branch(x, self.w12_main, self.w3_main)


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

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        split_point=0,
        shared_mlp_w12_reg=None,
        shared_mlp_w3_reg=None,
        shared_adaLN_reg=None,
    ):
        super().__init__()

        self.split_point = split_point
        self.norm1 = RMSNormSplit(hidden_size, eps=1e-6, split_point=split_point) if split_point > 0 else RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            split_point=split_point
        )
        self.norm2 = RMSNormSplit(hidden_size, eps=1e-6, split_point=split_point) if split_point > 0 else RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = (
            SplitSwiGLUFFN(
                hidden_size,
                mlp_hidden_dim,
                drop=proj_drop,
                split_point=split_point,
                shared_w12_reg=shared_mlp_w12_reg,
                shared_w3_reg=shared_mlp_w3_reg,
            )
            if split_point > 0 else
            SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        )

        self.adaLN_act = nn.SiLU()
        self.adaLN_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.adaLN_reg = shared_adaLN_reg if split_point > 0 and shared_adaLN_reg is not None \
            else (nn.Linear(hidden_size, 6 * hidden_size, bias=True) if split_point > 0 else None)

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        h = self.adaLN_act(c)
        mod = self.adaLN_proj(h)
        reg_mod = self.adaLN_reg(h) if self.split_point > 0 else None

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_mod(mod)
        if reg_mod is not None:
            shift_msa_r, scale_msa_r, gate_msa_r, shift_mlp_r, scale_mlp_r, gate_mlp_r = split_mod(reg_mod)
        else:
            shift_msa_r = scale_msa_r = gate_msa_r = shift_mlp_r = scale_mlp_r = gate_mlp_r = None

        h = apply_mod(
            self.norm1(x),
            shift_msa, scale_msa,
            shift_msa_r, scale_msa_r,
            self.split_point
        )
        h = self.attn(h, rope=feat_rope)
        h = apply_gate(h, gate_msa, gate_msa_r, self.split_point)
        x = x + h

        h = apply_mod(
            self.norm2(x),
            shift_mlp, scale_mlp,
            shift_mlp_r, scale_mlp_r,
            self.split_point
        )
        h = self.mlp(h)
        h = apply_gate(h, gate_mlp, gate_mlp_r, self.split_point)
        x = x + h

        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        split_points = [
            self.in_context_len if i >= self.in_context_start else 0
            for i in range(depth)
        ]

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        reg_hidden_dim = int(mlp_hidden_dim * 2 / 3)

        if self.in_context_len > 0:
            self.shared_mlp_w12_reg = nn.Linear(hidden_size, 2 * reg_hidden_dim, bias=True)
            self.shared_mlp_w3_reg = nn.Linear(reg_hidden_dim, hidden_size, bias=True)
            self.shared_adaLN_reg = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        else:
            self.shared_mlp_w12_reg = None
            self.shared_mlp_w3_reg = None
            self.shared_adaLN_reg = None

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                split_point=split_points[i],
                shared_mlp_w12_reg=self.shared_mlp_w12_reg if split_points[i] > 0 else None,
                shared_mlp_w3_reg=self.shared_mlp_w3_reg if split_points[i] > 0 else None,
                shared_adaLN_reg=self.shared_adaLN_reg if split_points[i] > 0 else None,
            )
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_proj.weight, 0)
            nn.init.constant_(block.adaLN_proj.bias, 0)
            if hasattr(block, "adaLN_lora_B"):
                nn.init.constant_(block.adaLN_lora_B.weight, 0)

        if self.shared_adaLN_reg is not None:
            nn.init.constant_(self.shared_adaLN_reg.weight, 0)
            nn.init.constant_(self.shared_adaLN_reg.bias, 0)

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
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=16, **kwargs) # in_context_len=32, in_context_start=4,

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=32, **kwargs) # in_context_len=32, in_context_start=4,

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=16, **kwargs) # in_context_len=32, in_context_start=8,

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=32, **kwargs) # in_context_len=32, in_context_start=8,

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=16, **kwargs) # in_context_len=32, in_context_start=10,

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=32, **kwargs) # in_context_len=32, in_context_start=10,


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}