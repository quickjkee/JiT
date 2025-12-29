from ast import arg
import torch
import copy
import torch.nn as nn
from math import exp
from model_jit import JiT_models


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
    loss = loss.mean(dim=(1, 2, 3)).mean()
    return loss

class MCD(nn.Module):
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


    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_discrete_t_start(self, n, device):
        z = torch.randn(n, device=device) * 0.8 - 0.8
        t = torch.sigmoid(z)
        idx = torch.bucketize(t, self.timesteps_start)
        idx = (idx - 1).clamp(0, self.timesteps_start.numel() - 1)
        return self.timesteps_start[idx], idx

    def forward(self, x, labels):
        t_start, idx = self.sample_discrete_t_start(x.size(0), device=x.device)
        t_next = self.timesteps_end[idx]
        t_start, t_next = t_start.view(-1, *([1] * (x.ndim - 1))), t_next.view(-1, *([1] * (x.ndim - 1)))

        e = torch.randn_like(x) * self.noise_scale
        z_start = t_start * x + (1 - t_start) * e
        z_next = self._heun_step(z_start, 
                                 t_start, 
                                 t_next, 
                                 labels, model=self.net_teacher)

        x0_start = self.net(z_start, t_start.flatten(), labels)
        v_start = (x0_start - z_start) / (1.0 - t_start).clamp_min(self.t_eps)
        f_start = z_start + (t_next - t_start) * v_start

        with torch.no_grad():
            idx = torch.searchsorted(self.boundaries.view(-1, *([1] * (x.ndim - 1))), t_next, right=False).clamp(max=self.boundaries.numel()-1)
            t_boundary = self.boundaries.view(-1, *([1] * (x.ndim - 1)))[idx]

            x0_next = self.net(z_next, t_next.flatten(), labels)
            v_next = (x0_next - z_next) / (1.0 - t_next).clamp_min(self.t_eps)
            f_next = z_next + (t_boundary - t_next) * v_next

        loss = mse_loss(f_start, f_next)

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = self.boundaries.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        size = len(timesteps)

        # ode
        for i in range(size - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x0 = self.net(z, t.flatten(), labels)
            v = (x0 - z) / (1.0 - t).clamp_min(self.t_eps)
            z = z + (t_next - t) * v

        return z

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
    def _heun_step(self, z, t, t_next, labels, model=None):
        v_pred_t = self._forward_sample(z, t, labels, model)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, model)

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
