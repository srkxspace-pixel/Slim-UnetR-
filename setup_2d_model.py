"""Download pinned public 2D weights (and optional real validation data). No inference uploads."""
import argparse
import hashlib
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model_store/brisc_2d'
HEAD_REVISION = '04c94c44998bd793e97dd8c1a539fbfbd21c958b'
BASE_REVISION = '739f5d4692954e4a185eac280dec1ba5a7d52f1d'
DATA_REVISION = '6aafaf9cfe043e74a1d2a45222339096f220046b'
FILES = [
    ('base_model.bin', f'https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640/resolve/{BASE_REVISION}/pytorch_model.bin', 'a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df', 'sha256'),
    ('seg_best.pt', f'https://huggingface.co/imbenjita/BrainTumorClassifier/resolve/{HEAD_REVISION}/seg_best.pt', '20947b5b737b47de3631fdb9930f6d81c79b92703c8a65f554b948e1ddfd07ee', 'sha256'),
    ('config.json', f'https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640/resolve/{BASE_REVISION}/config.json', '3cf1f37a73b0e34d1fd6fda13760f9037a91c46e', 'git'),
    ('UPSTREAM_MODEL_CARD.md', f'https://huggingface.co/imbenjita/BrainTumorClassifier/resolve/{HEAD_REVISION}/README.md', 'f0f10da8f275c1e174cb0e911b222e13f9e9c744', 'git'),
    ('upstream_reference.py', f'https://huggingface.co/imbenjita/BrainTumorClassifier/resolve/{HEAD_REVISION}/brisc_segformer_convnext.py', 'cda696fea3e2096173358ac5399d1963d9ddf121', 'git'),
]
DATA = ('brisc.zip', f'https://huggingface.co/datasets/amirreza-fateh/brisc/resolve/{DATA_REVISION}/brisc.zip', '85511543c2ad9abff44e68ee5abab353a5089d5b6f241960c2ea7462aaf4262b', 'sha256')


def digest(path, kind='sha256'):
    result = hashlib.sha1() if kind == 'git' else hashlib.sha256()
    if kind == 'git':
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(directory, spec):
    name, url, expected, kind = spec
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        if digest(target, kind) != expected:
            raise RuntimeError(f'{target} has an unexpected hash. Restore or move it before retrying.')
        print(f'Verified existing {name}', flush=True)
        return
    temporary = target.with_name(target.name + '.partial')
    for attempt in range(3):
        try:
            print(f'Downloading {name} (attempt {attempt + 1}/3)', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'SlimUNETR-local-demo/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    total += len(block)
                    if total > 400 * 1024 * 1024:
                        raise RuntimeError('Download exceeded the expected size boundary.')
                    output.write(block)
            if digest(temporary, kind) != expected:
                raise RuntimeError(f'Checksum failed for {name}; file was not installed.')
            temporary.replace(target)
            print(f'Installed and verified {name}', flush=True)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-data', action='store_true', help='Also download the public BRISC archive for real-image verification')
    args = parser.parse_args()
    for spec in FILES:
        download(MODEL, spec)
    if args.test_data:
        download(ROOT / 'outputs/verification_2d/source', DATA)
    print('2D model ready for offline loading. No training performed.', flush=True)
