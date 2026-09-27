"""Create relocatable, verified inference .tar bundles for Nirvana inputs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from prepare_baselines import ROOT, SOURCES, WEIGHTS, WEIGHT_REVISIONS

BUNDLES = {'repa': 'sit_repa_512', 'pixelflow': 'pixelflow_256',
           'pixnerd': 'pixnerd_512', 'rae': 'rae_512'}


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def package(model, destination):
    external = ROOT / 'external'
    repo_url, folder, revision = SOURCES[model]
    repo = external / folder
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'])
    if actual != revision or dirty:
        raise RuntimeError(f'{repo}: expected clean source revision {revision}')
    files = {}
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
    for name in filter(None, tracked):
        files[f'{folder}/{name}'] = repo / name
    for _, filename, local_dir in WEIGHTS[model]:
        files[f'{local_dir}/{filename}'] = external / local_dir / filename
    for path in (external / 'python_deps').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files[str(path.relative_to(external))] = path
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (external / 'python_deps/diffusers/__init__.py').is_file():
        raise FileNotFoundError('Run prepare_baselines.py --install-deps first')
    name = BUNDLES[model]
    target = destination / f'{name}.tar'
    temporary = target.with_suffix('.tar.partial')
    if target.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to overwrite {target} or its partial archive')
    manifest = dict(model=model, source=f'https://github.com/{repo_url}', source_revision=revision,
                    weights=[dict(repo=hub, filename=f, revision=WEIGHT_REVISIONS[hub])
                             for hub, f, _ in WEIGHTS[model]], files={})
    with tarfile.open(temporary, 'x') as archive:
        for relative, path in sorted(files.items()):
            with path.open('rb') as stream:
                manifest['files'][relative] = dict(size=path.stat().st_size, sha256=sha256(stream))
            archive.add(path, arcname=f'{name}/{relative}', recursive=False)
        add_bytes(archive, f'{name}/{folder}/SOURCE_REVISION', (revision + '\n').encode())
        add_bytes(archive, f'{name}/manifest.json', json.dumps(manifest, indent=2).encode())
    # Verify every packaged source/weight/dependency against its original file digest.
    with tarfile.open(temporary, 'r') as archive:
        for relative, expected in manifest['files'].items():
            with archive.extractfile(f'{name}/{relative}') as stream:
                if sha256(stream) != expected['sha256']:
                    raise RuntimeError(f'Archive content mismatch: {relative}')
    os.replace(temporary, target)
    with target.open('rb') as stream:
        checksum = sha256(stream)
    target.with_suffix('.tar.sha256').write_text(f'{checksum}  {target.name}\n')
    print(f'Verified {target}: {target.stat().st_size / 1e9:.3f} GB, {len(files)} files', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path.home() / 'nirvana_data')
    parser.add_argument('--models', nargs='+', choices=BUNDLES, default=list(BUNDLES))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        package(model, args.out_dir)


if __name__ == '__main__':
    main()
