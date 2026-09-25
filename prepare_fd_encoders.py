"""
Download the encoders FD_r^6 needs, so that scoring runs with no network access.

Run this once on a machine that has internet, then copy the output directory (and
`fid_stats/fd_repr`, written by prepare_fd_stats.py) to the cluster. util/fd_repr.py picks the
weights up from there and never contacts the hub.

    python prepare_fd_encoders.py                 # -> fd_encoders/*.pt  (about 5 GB)
    python prepare_fd_encoders.py --models dinov2 clip

The Inception encoder is not included: it already lives in fid_stats/.
"""
import argparse
import os

import torch

from util.fd_repr import DEFAULT_WEIGHTS_DIR, REPR_MODELS, encoder_file


def main():
    parser = argparse.ArgumentParser('encoders for FD_r^6')
    parser.add_argument('--out_dir', default=DEFAULT_WEIGHTS_DIR, type=str)
    parser.add_argument('--models', default=None, type=str, nargs='+',
                        help='subset of: {}'.format(' '.join(n for n in REPR_MODELS if n != 'inception')))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    import timm

    names = args.models or [n for n in REPR_MODELS if n != 'inception']
    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        if name == 'inception':
            print('inception: already in fid_stats/, skipping')
            continue
        spec = REPR_MODELS[name]
        out = os.path.join(args.out_dir, encoder_file(name))
        if os.path.exists(out) and not args.overwrite:
            print('{}: {} exists, skipping'.format(name, out))
            continue
        print('{}: downloading {}'.format(name, spec['timm']))
        model = timm.create_model(spec['timm'], pretrained=True, num_classes=0)
        torch.save(model.state_dict(), out)
        print('   wrote {} ({:.1f} GB)'.format(out, os.path.getsize(out) / 1e9))
        del model

    print('\ncopy {} and fid_stats/fd_repr to the machine that will score runs'.format(args.out_dir))


if __name__ == '__main__':
    main()
