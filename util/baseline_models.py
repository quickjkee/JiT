"""Thin inference adapters for official ImageNet model repositories.

One adapter per process: upstream repositories use overlapping module names.
"""
import contextlib
import json
import tempfile
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REPOS = {'repa': 'REPA', 'pixelflow': 'PixelFlow', 'pixnerd': 'PixNerd', 'rae': 'RAE'}


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_weights(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    return state.get('ema', state)


def load_rae_dit(config):
    from utils.model_utils import get_obj_from_str

    # Avoid CPU rotary cos/sin initialization, which segfaulted on the remote runtime.
    # Both released checkpoints include these buffers; strict assignment restores them.
    with torch.device('meta'):
        model = get_obj_from_str(config.target)(**config.get('params', {}))
    model.load_state_dict(load_weights(config.ckpt), strict=True, assign=True)
    return model


def build_generator(args, device):
    assets = Path(args.assets_dir).resolve()
    repo = Path(args.repo).resolve() if args.repo else assets / REPOS[args.model]
    sys.path.insert(0, str(assets / 'python_deps'))
    sys.path.insert(0, str(repo))
    if not repo.is_dir():
        raise FileNotFoundError(f'{repo}: run prepare_baselines.py first')
    if args.model == 'repa':
        from diffusers import AutoencoderKL
        from models.sit import SiT_models
        from samplers import euler_maruyama_sampler
        from utils import load_legacy_checkpoints
        model = SiT_models['SiT-XL/2'](input_size=64, use_cfg=True, z_dims=[768],
                                     encoder_depth=8, fused_attn=True, qk_norm=False)
        path = args.ckpt or assets / 'weights/repa/sit_xl_2_repa_512.pt'
        state = load_weights(path)
        if any(k.startswith('decoder_blocks.') for k in state):
            state = load_legacy_checkpoints(state, encoder_depth=8)
        model.load_state_dict(state, strict=True)
        model = model.eval().to(device)
        vae_path = assets / 'weights/vae'
        vae = AutoencoderKL.from_pretrained(str(vae_path)).eval().to(device)

        def generate(labels):
            z = torch.randn(len(labels), 4, 64, 64, device=device)
            x = euler_maruyama_sampler(model, z, labels, num_steps=args.steps,
                cfg_scale=args.cfg, guidance_low=args.interval_min,
                guidance_high=args.interval_max, path_type='linear').float()
            return (vae.decode(x / 0.18215).sample + 1) / 2
        return generate

    if args.model == 'pixelflow':
        # Match the official sample_ddp.py precision settings.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from pixelflow.utils.config import instantiate_from_config
        from pixelflow.pipeline_pixelflow import PixelFlowPipeline
        from pixelflow.scheduling_pixelflow import PixelFlowScheduler
        folder = Path(args.ckpt) if args.ckpt else assets / 'weights/pixelflow'
        config = OmegaConf.load(folder / 'config.yaml')
        model = instantiate_from_config(config.model)
        model.load_state_dict(load_weights(folder / 'model.pt'), strict=True)
        model = model.eval().to(device)
        scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                       num_stages=config.scheduler.num_stages, gamma=-1/3)
        pipeline = PixelFlowPipeline(scheduler, model)
        print(f'PixelFlow: depth={config.model.params.depth}, '
              f'width={config.model.params.num_attention_heads * config.model.params.attention_head_dim}, '
              f'patch_size={model.patch_size}, solver={args.pixelflow_solver}, '
              f'cfg_max={args.cfg}', flush=True)

        def generate(labels):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                x = pipeline(prompt=labels.tolist(), height=256, width=256,
                    num_inference_steps=[args.steps] * config.scheduler.num_stages,
                    guidance_scale=args.cfg, device=device, shift=1.0,
                    use_ode_dopri5=args.pixelflow_solver == 'dopri5')
            return torch.from_numpy(x).permute(0, 3, 1, 2)
        return generate

    if args.model == 'pixnerd':
        from src.models.transformer.pixnerd_c2i import PixNerDiT
        from src.diffusion.flow_matching.sampling import EulerSampler
        from src.diffusion.flow_matching.scheduling import LinearScheduler
        from src.diffusion.base.guidance import simple_guidance_fn
        config = OmegaConf.load(repo / 'configs_c2i/pix256std1_repa_pixnerd_xl.yaml')
        model = PixNerDiT(**config.model.denoiser.init_args)
        path = args.ckpt or assets / 'weights/pixnerd/res512_ft200k_epoch=325-step=1800000_emainit.ckpt'
        state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        # Released 512 checkpoint includes EMA weights; prefer those explicitly.
        prefix = 'ema_denoiser.' if any(k.startswith('ema_denoiser.') for k in state) else 'denoiser.'
        print(f'PixNerd checkpoint prefix: {prefix}', flush=True)
        model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        model = model.eval().to(device)
        sampler = EulerSampler(scheduler=LinearScheduler(), num_steps=args.steps,
            guidance=args.cfg, guidance_fn=simple_guidance_fn,
            guidance_interval_min=args.interval_min, guidance_interval_max=args.interval_max)

        def generate(labels):
            previous_precision = torch.get_float32_matmul_precision()
            try:
                # Match upstream patch_bugs.py without changing FD encoder precision.
                torch.set_float32_matmul_precision('medium')
                z = torch.randn(len(labels), 3, 512, 512, device=device)
                return (sampler(model, z, labels, torch.full_like(labels, 1000)) + 1) / 2
            finally:
                torch.set_float32_matmul_precision(previous_precision)
        return generate

    sys.path.insert(0, str(repo / 'src'))
    from utils.model_utils import instantiate_from_config
    from stage2.transport import create_transport, Sampler
    config_path = args.config or repo / 'configs/stage2/sampling/ImageNet512/DiTDH-XL_DINOv2-B_decXL_AG.yaml'
    config = OmegaConf.load(config_path)
    if config.sampler.mode != 'ODE' or config.guidance.method != 'autoguidance':
        raise ValueError('RAE adapter requires an ODE/autoguidance sampling config')
    if args.ckpt:
        config.stage_2.ckpt = str(Path(args.ckpt).resolve())
    if not args.config:
        encoder_path = str(assets / 'weights/rae_encoder')
        config.stage_1.params.encoder_config_path = encoder_path
        config.stage_1.params.encoder_params.dinov2_path = encoder_path
    prefix = f'[rank={os.environ.get("RANK", "0")}] RAE'
    with working_directory(repo):
        # Transformers 5 validates types before RAE can replace its string placeholder.
        # Resolve only that placeholder to the exact value RAE assigns immediately after load.
        decoder_config = Path(config.stage_1.params.decoder_config_path) / 'config.json'
        decoder_data = json.loads(decoder_config.read_text())
        with tempfile.TemporaryDirectory(prefix='rae_decoder_') as temporary:
            if decoder_data.get('patch_size') == 'SHOULD BE RELOADED':
                decoder_data['patch_size'] = config.stage_1.params.get('decoder_patch_size', 16)
                Path(temporary, 'config.json').write_text(json.dumps(decoder_data))
                config.stage_1.params.decoder_config_path = temporary
            print(f'{prefix}: constructing stage 1 on CPU', flush=True)
            rae = instantiate_from_config(config.stage_1).eval()
            print(f'{prefix}: moving stage 1 to {device}', flush=True)
            rae = rae.to(device)
        print(f'{prefix}: verifying decoder checkpoint', flush=True)
        rae.decoder.load_state_dict(load_weights(config.stage_1.params.pretrained_decoder_path), strict=True)
        print(f'{prefix}: loading stage 2 from {config.stage_2.ckpt} (meta initialization)', flush=True)
        model = load_rae_dit(config.stage_2).eval()
        print(f'{prefix}: moving stage 2 to {device}', flush=True)
        model = model.to(device)
        print(f'{prefix}: loading autoguidance model (meta initialization)', flush=True)
        guide = load_rae_dit(config.guidance.guidance_model).eval().to(device)
        print(f'{prefix}: all models loaded', flush=True)
    transport = create_transport(**config.transport.params,
        time_dist_shift=(config.misc.time_dist_shift_dim / config.misc.time_dist_shift_base) ** 0.5)
    params = OmegaConf.to_container(config.sampler.params)
    params['num_steps'] = args.steps
    sample = Sampler(transport).sample_ode(**params)

    def generate(labels):
        z = torch.randn(len(labels), *config.misc.latent_size, device=device)
        y = torch.cat([labels, torch.full_like(labels, 1000)])
        x = sample(torch.cat([z, z]), model.forward_with_autoguidance,
            y=y, cfg_scale=args.cfg, cfg_interval=(args.interval_min, args.interval_max),
            additional_model_forward=guide.forward)[-1].chunk(2)[0]
        return rae.decode(x)
    return generate
