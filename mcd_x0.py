from ast import arg
import re
import torch
import copy
import torch.nn as nn
from math import exp
from model_jit import JiT_models
import torchvision.utils as vutils
import torch.distributed as dist


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

def mse_loss(pred, target):
    loss = (pred - target) ** 2
    loss = (loss.mean(dim=(1, 2, 3))).mean()
    return loss

def huber_loss(pred, target, w=None, delta=1.0):
    """
    pred, target: [B, C, H, W]
    w: optional [B] or [B,1,1,1] sample weights
    delta: Huber threshold
    """
    diff = pred - target
    abs_diff = diff.abs()

    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    linear = abs_diff - quadratic

    loss = 0.5 * quadratic.pow(2) + delta * linear
    loss = loss.mean(dim=(1, 2, 3))  # per-sample

    if w is not None:
        loss = loss * w.view(-1)

    return loss.mean()
    

class MCD_x0(nn.Module):
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

        # distillation hyper params
        self.timesteps = torch.linspace(0.0, 1.0, self.steps+1)
        self.timesteps_end = self.timesteps[1:]
        self.timesteps_start = self.timesteps[:-1]
        intervals = torch.chunk(self.timesteps, args.num_boundaries)
        self.boundaries = torch.tensor(
                        [interval[0] for interval in intervals] + [intervals[-1][-1]]
                    )

    def create_teacher(self):
        self.net_teacher = copy.deepcopy(self.net).eval()
        self.net_teacher.requires_grad_(False)

    def sample_discrete_t_start(self, n, device):
        timesteps_start = self.timesteps_start.to(device)
        idx = torch.randint(
            low=0,
            high=timesteps_start.numel(),
            size=(n,),
            device=device
        )
        return timesteps_start[idx], idx

    def forward(self, x, labels, ema_model=None):
        # Timesteps
        t_start, idx = self.sample_discrete_t_start(x.size(0), device=x.device)
        idx_next = torch.clamp(idx + 1, max=self.timesteps_end.numel() - 1)
        t_next = self.timesteps_end.to(x.device)[idx]
        t_next_next = self.timesteps_end.to(x.device)[idx_next].view(-1, *([1] * (x.ndim - 1)))
        t_start, t_next = t_start.view(-1, *([1] * (x.ndim - 1))), t_next.view(-1, *([1] * (x.ndim - 1)))

        boundaries = self.boundaries.to(x.device)
        idx = torch.searchsorted(boundaries, t_next.flatten(), right=False).clamp(max=boundaries.numel()-1)
        t_boundary = boundaries[idx].view_as(t_next)

        # Teacher
        e = torch.randn_like(x) * self.noise_scale
        z_start = t_start * x + (1 - t_start) * e
        z_next, v_pred_start = self._heun_step(z_start, t_start, t_next, labels, 
                                               model=self.net_teacher, return_v=True)
        x0_pred_start = z_start + v_pred_start * (1 - t_start)

        # Prediction
        delta_pred = self.net(x0_pred_start, t_start.flatten(), labels)

        # Target
        with torch.no_grad():
            _, v_pred_next = self._heun_step(z_next, t_next, t_next_next, labels, 
                                             model=self.net_teacher, return_v=True)
            x0_pred_next = z_next + v_pred_next * (1 - t_next)
            target_fn = self.net if ema_model is None else ema_model.net
            delta_target = target_fn(x0_pred_next, t_next.flatten(), labels)
            scale = (t_boundary - t_next) / (1 - t_next).clamp_min(self.t_eps)
            x0_boundary_target = x0_pred_next + scale * delta_target
            boundary_mask = (t_next - t_boundary).abs() < 1e-6
            x0_boundary_target = torch.where(boundary_mask, x0_pred_next, x0_boundary_target)

        scale = (t_boundary - t_start) / (1 - t_start).clamp_min(self.t_eps)
        delta_target = (x0_boundary_target - x0_pred_start) / scale.clamp_min(1e-4)
        
        t_ = t_start.view(-1).clamp(self.t_eps, 1 - self.t_eps)
        snr = (t_ * t_) / ((1 - t_) * (1 - t_))
        w_t = torch.minimum(snr, torch.tensor(10.0, device=t_.device, dtype=t_.dtype))  # gamma=10
        dt = (t_boundary - t_start).view(-1).abs()
        w_seg = 1.0 / (dt + 1e-3)
        w = w_t * w_seg
        loss = huber_loss(delta_pred, delta_target, w=w, t=t_start.flatten())
        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = self.boundaries.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1).to(device)
        timsteps_next = self.timesteps_end.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1).to(device)
        size = len(timesteps)

        # ode
        for i in range(size - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if i == 0:
                _, v = self._heun_step(z, t, timsteps_next[i], labels, 
                                       model=self.net_teacher, return_v=True)
                z = z + v * (1 - t)     
            z_ = self.net(z, t.flatten(), labels)
            scale = (t_next - t) / (1 - t).clamp_min(self.t_eps)
            z = z + scale * z_

        return z.to(t.dtype)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, model):
        if model is None:
            model = self.net

        # conditional
        x_cond = model(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = model(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, model=None):
        v_pred = self._forward_sample(z, t, labels, model)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, model=None, return_v=False):
        v_pred_t = self._forward_sample(z, t, labels, model)
        dt = t_next - t
        z_next_euler = z + dt * v_pred_t
        z_next = z_next_euler.clone()
        v_pred = v_pred_t.clone()

        heun_mask = (t_next.view(-1) != 1.0)  # [B]
        if heun_mask.any():
            v_pred_t_next = self._forward_sample(
                z_next_euler[heun_mask],
                t_next.view(-1)[heun_mask].view(-1, 1, 1, 1),
                labels[heun_mask],
                model,
            )
            v_pred_heun = 0.5 * (v_pred_t[heun_mask] + v_pred_t_next)
            v_pred[heun_mask] = v_pred_heun
            z_next[heun_mask] = z[heun_mask] + dt[heun_mask] * v_pred_heun
    
        return (z_next, v_pred) if return_v else z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)