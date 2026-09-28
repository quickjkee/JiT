"""Generate official baseline samples and score them using JiT's FD_r pipeline."""
import argparse
import faulthandler
from importlib.metadata import version
import sys
import json
from math import isfinite
import os
from pathlib import Path
import subprocess

from PIL import Image
import torch
import torch.distributed as dist

from util.baseline_models import ROOT, REPOS, build_generator
from util.fd_repr import calculate_fdr, report_inputs


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', choices=REPOS, required=True)
    parser.add_argument('--repo')
    parser.add_argument('--assets-dir', default=str(ROOT / 'external'))
    parser.add_argument('--ckpt')
    parser.add_argument('--config')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-images', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--pixelflow-solver', choices=['euler', 'dopri5'])
    parser.add_argument('--cfg', type=float, required=True)
    parser.add_argument('--interval-min', type=float, default=0.)
    parser.add_argument('--interval-max', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-fdr', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--fdr-models', nargs='+')
    parser.add_argument('--fdr-stats-dir', default=str(ROOT / 'fid_stats/fd_repr'))
    parser.add_argument('--fdr-weights-dir', default=str(ROOT / 'fd_encoders'))
    parser.add_argument('--fdr-bsz', type=int, default=32)
    args = parser.parse_args()
    if args.num_images < 2 or args.batch_size < 1 or args.steps < 2:
        parser.error('num-images >= 2, batch-size >= 1 and steps >= 2 are required')
    if args.model == 'pixelflow':
        args.pixelflow_solver = args.pixelflow_solver or 'dopri5'
    elif args.pixelflow_solver is not None:
        parser.error('--pixelflow-solver is supported only for PixelFlow')
    if args.config and args.model != 'rae':
        parser.error('--config is supported only for RAE')
    if args.cfg < 1 or not 0 <= args.interval_min < args.interval_max <= 1:
        parser.error('Require cfg >= 1 and 0 <= interval-min < interval-max <= 1')
    if args.model == 'pixelflow' and (args.interval_min, args.interval_max) != (0., 1.):
        parser.error('PixelFlow applies CFG throughout; guidance intervals are unsupported')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        # Workers only synchronize filesystem writes; no GPU tensors are communicated.
        dist.init_process_group('gloo')
    print(f'[rank={rank}] Python {sys.version.split()[0]}, torch {torch.__version__}, '
          f'CUDA {torch.version.cuda}, transformers {version("transformers")}, '
          f'device={device}, barrier_backend={dist.get_backend() if world > 1 else "none"}', flush=True)
    torch.manual_seed(args.seed * world + rank)
    out = Path(args.output_dir).resolve()
    samples = out / 'samples'
    if not args.skip_fdr and not report_inputs(args.fdr_stats_dir, args.fdr_weights_dir,
                                              args.fdr_models, verbose=rank == 0):
        raise FileNotFoundError('Prepare FD statistics and encoders before evaluation')
    if not args.score_only:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            samples.mkdir()  # Refuse to mix new samples with an earlier run.
            assets = Path(args.assets_dir).resolve()
            repo = Path(args.repo).resolve() if args.repo else assets / REPOS[args.model]
            revision_file = repo / 'SOURCE_REVISION'
            commit = (revision_file.read_text().strip() if revision_file.exists() else
                      subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
            from prepare_baselines import WEIGHTS, WEIGHT_REVISIONS
            artifacts = [dict(repo=hub, revision=WEIGHT_REVISIONS[hub], filename=filename,
                              local_path=str(assets / dest / filename))
                         for hub, filename, dest in WEIGHTS[args.model]]
            (out / 'run.json').write_text(json.dumps(dict(vars(args), source_commit=commit,
                default_artifacts=artifacts,
                world_size=world, resolution=256 if args.model == 'pixelflow' else 512,
                labels='index modulo 1000', torch_version=torch.__version__), indent=2))
        if world > 1:
            dist.barrier()
        print(f'[rank={rank}] Loading {args.model}', flush=True)
        generate = build_generator(args, device)
        print(f'[rank={rank}] Generator ready', flush=True)
        indices = list(range(rank, args.num_images, world))
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                ids = indices[start:start + args.batch_size]
                labels = torch.tensor([i % 1000 for i in ids], device=device)
                x = generate(labels).detach().float()
                size = 256 if args.model == 'pixelflow' else 512
                if x.shape != (len(ids), 3, size, size) or not torch.isfinite(x).all():
                    raise RuntimeError(f'Invalid samples: shape={x.shape}, finite={torch.isfinite(x).all()}')
                print(f'rank={rank} samples={start + len(ids)}/{len(indices)} '
                      f'range=[{x.min().item():.3f},{x.max().item():.3f}]', flush=True)
                images = x.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for idx, img in zip(ids, images):
                    Image.fromarray(img).save(samples / f'{idx:06d}.png')
        del generate
        torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()  # Other workers can exit before rank zero scores images.
    if rank == 0:
        files = list(samples.glob('*.png'))
        expected = {f'{i:06d}.png' for i in range(args.num_images)}
        if {p.name for p in files} != expected:
            raise RuntimeError('Sample file set does not match the requested count')
        if not args.skip_fdr:
            result = calculate_fdr(str(samples), args.fdr_stats_dir, models=args.fdr_models,
                device=str(device), batch_size=args.fdr_bsz, num_workers=4,
                weights_dir=args.fdr_weights_dir,
                inception_path=str(ROOT / 'fid_stats/pt_inception-2015-12-05-6726825d.pth'))
            if not all(isfinite(v) for v in result['fd'].values()):
                raise RuntimeError('Non-finite FD result; no metrics file written')
            (out / 'metrics.json').write_text(json.dumps(result, indent=2))
            print(f"FDr^{len(result['fdr'])}: {result['fdr6']:.4f}", flush=True)


if __name__ == '__main__':
    main()
