from ast import arg
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def print_trainable(model):
    total = 0
    trainable = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            status = "TRAIN"
        else:
            status = "FROZEN"

        print(f"{status:6} | {name:60} | {n:>10}")

    frozen = total - trainable
    print("-" * 90)
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable ratio:  {100 * trainable / total:.2f}%")


def diffusion_loss(v, v_pred):
    loss = (v - v_pred) ** 2
    loss = loss.mean(dim=(1, 2, 3)).mean()
    return loss


def repa_loss(model_regs, dino_patches, topk=16, lambda_div=0.01):
    """
    model_regs:   [B, R, D]
    dino_patches: [B, L, D]
    """
    model_regs = F.normalize(model_regs, dim=-1)
    dino_patches = F.normalize(dino_patches, dim=-1)

    sim = torch.matmul(model_regs, dino_patches.transpose(-1, -2))   # [B, R, L]

    topk_sim = sim.topk(k=topk, dim=-1).values                        # [B, R, K]
    loss_align = -topk_sim.mean()

    reg_sim = torch.matmul(model_regs, model_regs.transpose(-1, -2))  # [B, R, R]
    R = model_regs.size(1)
    eye = torch.eye(R, device=model_regs.device, dtype=model_regs.dtype).unsqueeze(0)
    loss_div = ((reg_sim * (1.0 - eye)) ** 2).mean()

    loss = loss_align + lambda_div * loss_div
    return loss


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        dinov2_vitg14 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg",
            trust_repo=True, force_reload=False
        )
        dinov2_vitg14.eval().requires_grad_(False).cuda()
        object.__setattr__(self, "_dinov2_vitg14", dinov2_vitg14)

        self.net = JiT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                in_context_len=args.in_context_len,
                in_context_start=args.in_context_start,
                dino_embed_dim=dinov2_vitg14.embed_dim
            )
        print_trainable(self.net)


        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.args = args

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        with torch.inference_mode():
            x_dino = F.interpolate(
                x, size=(224, 224), mode="bicubic", align_corners=False
            )
            x_dino = (x_dino + 1.0) * 0.5          # [-1,1] → [0,1]
            x_dino = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x_dino)
            with torch.autocast(device_type="cuda", enabled=False):
                x_registers = self._dinov2_vitg14.forward_features(x_dino.float())['x_norm_patchtokens']

        labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred, registers_pred = self.net(z, t.flatten(), labels_dropped, drop_registers_layer=5)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        loss = diffusion_loss(v, v_pred)
        loss_repa = repa_loss(x_registers, registers_pred)
        loss = loss + 0.1 * loss_repa

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)

        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)