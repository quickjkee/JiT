"""Fetch pinned official inference source and the selected ImageNet checkpoints."""
import argparse
from pathlib import Path
import subprocess
import sys
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'repa': ('sihyun-yu/REPA', 'REPA', '67f714503e3892f993844aab088ffc5791c92613'),
    'pixelflow': ('ShoufaChen/PixelFlow', 'PixelFlow', '8805204d1be8df22382280959b3063974ff3d77d'),
    'pixnerd': ('MCG-NJU/PixNerd', 'PixNerd', '6da060bd4b11a4a0ac2443a31b79701090e23c23'),
    'rae': ('bytetriper/RAE', 'RAE', 'a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041'),
}
WEIGHTS = {
    'repa': [('kyungmnlee/DMF', 'sit_xl_2_repa_512.pt', 'weights/repa')]
        + [('stabilityai/sd-vae-ft-ema', f, 'weights/vae') for f in
           ['config.json', 'diffusion_pytorch_model.safetensors']],
    'pixelflow': [('ShoufaChen/PixelFlow-Class2Image', f, 'weights/pixelflow')
                  for f in ['config.yaml', 'model.pt']],
    'pixnerd': [('MCG-NJU/PixNerd-XL-P16-C2I',
                'res512_ft200k_epoch=325-step=1800000_emainit.ckpt', 'weights/pixnerd')],
    'rae': [('nyu-visionx/RAE-collections', f, 'RAE/models') for f in [
        'DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-XL_ep400/stage2_model.pt',
        'DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-S_ep20/stage2_model.pt',
        'decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt',
        'stats/dinov2/wReg_base/imagenet1k_512/stat.pt']]
        + [('facebook/dinov2-with-registers-base', f, 'weights/rae_encoder') for f in
           ['config.json', 'preprocessor_config.json', 'model.safetensors']],
}


WEIGHT_REVISIONS = {
    'MCG-NJU/PixNerd-XL-P16-C2I': 'edd8adfd6b2f433a02543094f331c9207f40b9db',
    'ShoufaChen/PixelFlow-Class2Image': 'bc57cd2968e60ca48141bf4c18666d7df75aa4b8',
    'facebook/dinov2-with-registers-base': 'a1d738ccfa7ae170945f210395d99dde8adb1805',
    'kyungmnlee/DMF': '4b040eb1ed1e3f9e1d01dd89d499f720296fba98',
    'nyu-visionx/RAE-collections': '1be4f03273523431f099a934da4cf1940dc6039f',
    'stabilityai/sd-vae-ft-ema': 'f04b2c4b98319346dad8c65879f680b1997b204a',
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--models', nargs='+', choices=SOURCES, default=list(SOURCES))
    p.add_argument('--install-deps', action='store_true',
                   help='Install inference extras under external/python_deps without changing the environment')
    args = p.parse_args()
    external = ROOT / 'external'
    external.mkdir(exist_ok=True)
    for model in args.models:
        remote, folder, revision = SOURCES[model]
        repo = external / folder
        if not repo.exists():
            subprocess.run(['git', 'clone', 'https://github.com/' + remote + '.git', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{repo} is at {actual}; expected {revision}. Existing checkout left untouched.')
        for hub, filename, dest in WEIGHTS[model]:
            print(hf_hub_download(hub, filename, revision=WEIGHT_REVISIONS[hub], local_dir=external / dest), flush=True)
    if args.install_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--target', str(external / 'python_deps'),
                        'diffusers==0.32.2', 'torchdiffeq==0.2.4'], check=True)


if __name__ == '__main__':
    main()
