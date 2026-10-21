"""Install verified public pretrained weights and check the bundled real MRI example."""
import argparse
from pathlib import Path
import shutil

from setup_2d_model import download, FILES, MODEL
from src.demo_case import demo_case

ROOT = Path(__file__).resolve().parent
SLIM_MODEL = (
    'brats21_pytorch_model.bin',
    'https://github.com/deepang-ai/Slim-UNETR/releases/download/v1.0.0/brats21_pytorch_model.bin',
    '8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4',
    'sha256',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-2d', action='store_true', help='Also install the optional BRISC JPG/PNG model (about 352 MB)')
    args = parser.parse_args()
    demo_case()
    print('Bundled real MRI and reference hashes verified.', flush=True)
    download(ROOT / 'model_store/official_brats2021', SLIM_MODEL)
    if args.with_2d:
        for spec in FILES:
            download(MODEL, spec)
        shutil.copyfile(ROOT / 'docs/models/NVIDIA_LICENSE.txt', MODEL / 'NVIDIA_LICENSE.txt')
    print('Setup complete. Start: python local_ui.py --open-browser', flush=True)


if __name__ == '__main__':
    main()
