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


def repa_loss(dino_feats, x_mid, t=None):
    dino_feats = F.normalize(dino_feats, dim=-1) # [B,T,D]
    x_mid = F.normalize(x_mid, dim=-1) # [B,T,D]
    cos_sim = (dino_feats * x_mid).sum(dim=-1)    # [B,T]
    loss_repa = -cos_sim.mean(dim=(1)).mean()
    return loss_repa


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        self.net = JiT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                in_context_len=args.in_context_len,
                in_context_start=args.in_context_start,
                do_in_context=args.do_in_context
            )
        print_trainable(self.net)


        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.args = args

        self.label_drop_prob = args.label_drop_prob
        self.p_drop_registers = args.p_drop_registers
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
        self.cfg_rg_scale = args.cfg_rg_scale
        self.rg_scale = args.rg_scale
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.rg_interval = (args.interval_min_rg, args.interval_max_rg)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        p_without_regs = self.p_drop_registers
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # one branch per iteration
        train_without_regs = self.training and (torch.rand((), device=x.device) < p_without_regs)
        use_registers = False if train_without_regs else True

        x_pred = self.net(
            z,
            t.flatten(),
            labels_dropped,
            use_registers=use_registers,
        )

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        loss = diffusion_loss(v, v_pred)

        return loss

    @torch.no_grad()
    def generate(self, labels, use_registers=False, do_cfg_rg=False):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        forward = self._forward_sample_cfg_rg if do_cfg_rg else self._forward_sample_cfg

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
            z = stepper(z, t, t_next, labels, use_registers, forward)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, use_registers, forward)
        return z

    @torch.no_grad()
    def _forward_sample_cfg(self, z, t, labels, use_registers):
        # conditional
        x_cond = self.net(z, t.flatten(), labels, use_registers)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), use_registers)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _forward_sample_cfg_rg(self, z, t, labels, use_registers):
        # conditional, WITH registers       (A)
        x_cond_reg = self.net(z, t.flatten(), labels, use_registers=True)
        v_cond_reg = (x_cond_reg - z) / (1.0 - t).clamp_min(self.t_eps)

        # conditional, NO registers         (B)  -- same class, drives the register axis
        x_cond_noreg = self.net(z, t.flatten(), labels, use_registers=False)
        v_cond_noreg = (x_cond_noreg - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional, WITH registers     (C)  -- CFG keeps registers on both sides
        x_uncond_reg = self.net(z, t.flatten(), torch.full_like(labels, 1000), use_registers=True)
        v_uncond_reg = (x_uncond_reg - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        low_rg, high_rg = self.rg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        interval_mask_reg = (t < high_rg) & ((low_rg == 0) | (t > low_rg))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_rg_scale, 1.0)   # class weight  w_c  -> (A - C)
        reg_scale_interval = torch.where(interval_mask_reg, self.rg_scale, 0.0)   # register weight w_r -> (A - B)

        return v_uncond_reg + cfg_scale_interval * (v_cond_reg - v_uncond_reg) + reg_scale_interval * (v_cond_reg - v_cond_noreg)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, use_registers, forward):
        v_pred =forward(z, t, labels, use_registers)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, use_registers, forward):
        v_pred_t = forward(z, t, labels, use_registers)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = forward(z_next_euler, t_next, labels, use_registers)

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