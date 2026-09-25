"""
FD_r^6: Frechet distances in six representation spaces, normalised and averaged.

From "Representation Frechet Loss for Visual Generation" (https://github.com/Jiawei-Yang/FD-loss,
arXiv 2604.28190). Each space contributes the Frechet distance between the generated set and the
reference statistics, divided by the distance that same space assigns to two disjoint halves of
real ImageNet data (the published "validation FD"), which puts the six on a common scale:

    FD_r^6 = mean_m  FD_m(generated, reference) / valFD_m

The spaces are Inception (the usual FID), ConvNeXt-V2, DINOv2, CLIP, MAE and SigLIP. The valFD
constants and the reference statistics are the ones released with the paper; download them with
`python prepare_fd_stats.py` before using this module.

The Inception term reuses the FID this repo already computes, so passing `fid_value` avoids
running Inception twice.
"""
import os
import pathlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from util.fid import calculate_frechet_distance

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png', 'ppm', 'tif', 'tiff', 'webp'}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> timm identifier, input size the reference statistics were computed at, the published
# validation FD used as the normaliser, and the reference statistics file
REPR_MODELS = OrderedDict([
    ('inception', dict(timm=None, target=None, valfd=1.68,
                       stats='guided_diffusion_stats.npz')),
    ('convnext', dict(timm='convnextv2_base.fcmae_ft_in22k_in1k', target=224, valfd=56.87,
                      stats='convnext_in256_t224_stats.npz')),
    ('dinov2', dict(timm='vit_large_patch14_dinov2.lvd142m', target=256, valfd=14.19,
                    stats='vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz')),
    ('clip', dict(timm='vit_large_patch14_clip_224.openai', target=256, valfd=5.60,
                  stats='vit_large_patch14_clip_224_openai_in256_t256_stats.npz')),
    ('mae', dict(timm='vit_large_patch16_224.mae', target=224, valfd=0.04,
                 stats='vit_large_patch16_224_mae_in256_t224_stats.npz')),
    ('siglip', dict(timm='vit_so400m_patch16_siglip_256.v2_webli', target=224, valfd=0.60,
                    stats='vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz')),
])

DEFAULT_MODELS = list(REPR_MODELS)

# where prepare_fd_encoders.py puts the encoder weights, so that scoring needs no network
DEFAULT_WEIGHTS_DIR = 'fd_encoders'


class _ImagePathDataset(torch.utils.data.Dataset):
    """Loads images lazily as float tensors in [0, 1]; the generated PNGs are already 256x256."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
        return x.permute(2, 0, 1).float().div_(255)


def list_images(folder, num_images=None):
    folder = pathlib.Path(folder)
    files = sorted(f for ext in IMAGE_EXTENSIONS for f in folder.glob('*.{}'.format(ext)))
    if not files:
        raise RuntimeError('no images found in {}'.format(folder))
    return files[:num_images] if num_images else files


def encoder_file(name):
    """local weights filename for one representation space"""
    return REPR_MODELS[name]['timm'].replace('/', '_').replace('.', '_') + '.pt'


def build_encoder(name, device, weights_dir=DEFAULT_WEIGHTS_DIR):
    """
    The encoder for one representation space, built the way the reference statistics were.

    Mirrors FD-loss: the model is created with dynamic_img_size / dynamic_img_pad so that it
    accepts the reference input size (256 is not a multiple of DINOv2's patch size of 14, and
    the padding is what keeps the token grid the same as theirs), and the normalisation comes
    from the model's own pretrained config, which differs from ImageNet for CLIP and SigLIP.

    Weights come from `weights_dir` when they are there, so the metric runs with no network
    access; see prepare_fd_encoders.py. Otherwise timm downloads them.
    """
    import timm

    spec = REPR_MODELS[name]
    local = os.path.join(weights_dir, encoder_file(name)) if weights_dir else None
    offline = local is not None and os.path.exists(local)

    kwargs = dict(pretrained=not offline, num_classes=0)
    try:
        model = timm.create_model(spec['timm'], dynamic_img_size=True, dynamic_img_pad=True,
                                  **kwargs)
    except TypeError:                      # CNNs take any input size already
        model = timm.create_model(spec['timm'], **kwargs)
    if offline:
        model.load_state_dict(torch.load(local, map_location='cpu', weights_only=True))
    model = model.eval().to(device)

    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg.get('mean', IMAGENET_MEAN), device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get('std', IMAGENET_STD), device=device).view(1, 3, 1, 1)
    return model, mean, std


def preprocess(x, target, mean, std):
    """[0, 1] images -> encoder input: bicubic resize to the reference size, then normalise"""
    x = F.interpolate(x, size=(target, target), mode='bicubic', align_corners=False,
                      antialias=True)
    return (x - mean) / std


def _pool(model, x):
    """cls / attention-pooled feature for ViTs, spatially averaged feature for CNNs"""
    out = model.forward_features(x)
    if out.ndim == 4:                                   # [B, C, H, W]
        return out.mean(dim=(2, 3))
    if getattr(model, 'attn_pool', None) is not None:   # e.g. SigLIP
        return model.forward_head(out, pre_logits=True)
    return out[:, 0]                                    # class token


@torch.no_grad()
def representation_statistics(folder, name, device, batch_size=64, num_workers=8, num_images=None,
                              weights_dir=DEFAULT_WEIGHTS_DIR):
    """mu, sigma of one representation space over the images in `folder`"""
    spec = REPR_MODELS[name]
    model, mean, std = build_encoder(name, device, weights_dir)

    loader = torch.utils.data.DataLoader(
        _ImagePathDataset(list_images(folder, num_images)), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=num_workers)

    feats = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        feats.append(_pool(model, preprocess(x, spec['target'], mean, std)).float().cpu())
    del model
    torch.cuda.empty_cache()

    feats = torch.cat(feats).numpy()
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def report_inputs(stats_dir, weights_dir=DEFAULT_WEIGHTS_DIR, models=None, verbose=True):
    """
    Log what is in the statistics and encoder directories, and whether FD_r can run from them.

    Returns True when every space in `models` has its reference statistics; missing encoders are
    only a warning, since timm would download them if the machine has network access.
    """
    models = list(models) if models else list(DEFAULT_MODELS)
    missing_stats, missing_weights = [], []
    lines = ['FD_r inputs:']

    lines.append('  statistics dir: {}{}'.format(
        os.path.abspath(stats_dir), '' if os.path.isdir(stats_dir) else '   (does not exist)'))
    for name in models:
        f = REPR_MODELS[name]['stats']
        path = os.path.join(stats_dir, f)
        if os.path.exists(path):
            lines.append('    [ok]      {:<10} {:<60} {:>7.1f} MB'.format(
                name, f, os.path.getsize(path) / 1e6))
        else:
            missing_stats.append(name)
            lines.append('    [MISSING] {:<10} {}'.format(name, f))

    lines.append('  encoder dir:    {}{}'.format(
        os.path.abspath(weights_dir) if weights_dir else '(none)',
        '' if weights_dir and os.path.isdir(weights_dir) else '   (does not exist)'))
    for name in models:
        if name == 'inception':
            lines.append('    [ok]      {:<10} {}'.format(name, 'fid_stats/pt_inception-*.pth (in repo)'))
            continue
        f = encoder_file(name)
        path = os.path.join(weights_dir, f) if weights_dir else None
        if path and os.path.exists(path):
            lines.append('    [ok]      {:<10} {:<60} {:>7.2f} GB'.format(
                name, f, os.path.getsize(path) / 1e9))
        else:
            missing_weights.append(name)
            lines.append('    [download] {:<9} {:<60} not found, timm would fetch it'.format(name, f))

    if missing_stats:
        lines.append('  -> cannot run: no reference statistics for {}. Run prepare_fd_stats.py '
                     'and point --fdr_stats_dir at the result.'.format(', '.join(missing_stats)))
    if missing_weights:
        lines.append('  -> no local weights for {}; on a machine without network access run '
                     'prepare_fd_encoders.py and point --fdr_weights_dir at the result.'
                     .format(', '.join(missing_weights)))
    if verbose:
        print('\n'.join(lines), flush=True)
    return not missing_stats


def _reference(stats_dir, name):
    path = os.path.join(stats_dir, REPR_MODELS[name]['stats'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            'missing reference statistics for {}: {}\nRun `python prepare_fd_stats.py` first.'
            .format(name, path))
    with np.load(path) as f:
        return f['mu'][:], f['sigma'][:]


def calculate_fdr(folder, stats_dir, models=None, fid_value=None, device=None,
                  batch_size=64, num_workers=8, num_images=None, verbose=True,
                  inception_path='fid_stats/pt_inception-2015-12-05-6726825d.pth',
                  weights_dir=DEFAULT_WEIGHTS_DIR):
    """
    Frechet distance per representation space, its normalised version, and their mean.

    folder     directory of generated images
    stats_dir  directory holding the reference statistics (see prepare_fd_stats.py)
    fid_value  the Inception FID if it has already been computed for this folder; when it is
               None the Inception term is computed here against `guided_diffusion_stats.npz`

    Returns {'fd': {name: raw}, 'fdr': {name: raw / valFD}, 'fdr6': mean over the spaces}.
    """
    models = list(models) if models else list(DEFAULT_MODELS)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # log what the two input directories hold before spending anything on features
    if not report_inputs(stats_dir, weights_dir, models, verbose=verbose):
        raise FileNotFoundError('missing FD_r reference statistics in {}'.format(os.path.abspath(stats_dir)))

    fd, fdr = OrderedDict(), OrderedDict()
    for name in models:
        if name not in REPR_MODELS:
            raise KeyError('unknown representation space {!r}; known: {}'
                           .format(name, ', '.join(REPR_MODELS)))
        if name == 'inception' and fid_value is not None:
            raw = float(fid_value)
        else:
            if name == 'inception':
                from util.fid import calculate_fid
                raw = calculate_fid(folder, os.path.join(stats_dir, REPR_MODELS[name]['stats']),
                                    device=str(device), batch_size=batch_size,
                                    num_workers=num_workers, inception_path=inception_path)
            else:
                mu, sigma = representation_statistics(folder, name, device, batch_size,
                                                      num_workers, num_images, weights_dir)
                ref_mu, ref_sigma = _reference(stats_dir, name)
                raw = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)
        fd[name] = float(raw)
        fdr[name] = float(raw) / REPR_MODELS[name]['valfd']
        if verbose:
            print('FD[{}] = {:.4f}   FDr[{}] = {:.4f}'.format(name, fd[name], name, fdr[name]))

    out = dict(fd=fd, fdr=fdr, fdr6=float(np.mean(list(fdr.values()))))
    if verbose:
        print('FDr^{} = {:.4f}'.format(len(fdr), out['fdr6']))
    return out
