"""
Fetch the reference statistics needed for FD_r^6 (see util/fd_repr.py).

The statistics are the ones released with "Representation Frechet Loss for Visual Generation"
(https://github.com/Jiawei-Yang/FD-loss): one bundle on the Hugging Face hub holding the mean and
covariance of ImageNet in each representation space. This script downloads the bundle and unpacks
it into one .npz per space.

    python prepare_fd_stats.py                    # -> fid_stats/fd_repr/*.npz
    python prepare_fd_stats.py --out_dir <dir>
"""
import argparse
import os
import pickle

import numpy as np

from util.fd_repr import REPR_MODELS


def main():
    parser = argparse.ArgumentParser('reference statistics for FD_r^6')
    parser.add_argument('--out_dir', default='fid_stats/fd_repr', type=str)
    parser.add_argument('--repo', default='jjiaweiyang/FD-Loss', type=str,
                        help='Hugging Face repo holding the released statistics')
    parser.add_argument('--bundle', default='data/fid_stats/paper_ref_stats.pkl', type=str)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=args.repo, filename=args.bundle)
    with open(path, 'rb') as f:
        bundle = pickle.load(f)          # {npz filename: {array name: array}}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, arrays in bundle.items():
        np.savez(os.path.join(args.out_dir, name), **arrays)
        print('wrote {} ({})'.format(os.path.join(args.out_dir, name), ', '.join(sorted(arrays))))

    missing = [n for n, s in REPR_MODELS.items()
               if not os.path.exists(os.path.join(args.out_dir, s['stats']))]
    if missing:
        print('\nstill missing statistics for: {}'.format(', '.join(missing)))
        print('those spaces cannot be evaluated; pass --fdr_models with the rest')
    else:
        print('\nall {} representation spaces are ready'.format(len(REPR_MODELS)))


if __name__ == '__main__':
    main()
