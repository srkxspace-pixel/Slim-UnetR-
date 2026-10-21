"""Portable locations and integrity checks for the attributed public 3D demonstration case."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/BRATS_409'
EXAMPLE_HASHES = {
    'image.nii.gz': '74a4d182237e887ba5ec58b5fc4762c3b7913388d3122d1285525aacf9eeead2',
    'reference.nii.gz': 'a4c0573207472dc2a9cd334d36aee31dd555a28de5ad301bbcdaeac89ecf021e',
}


def demo_case(verify=True):
    for filename, expected in EXAMPLE_HASHES.items():
        path = EXAMPLE / filename
        if not path.is_file():
            raise ValueError(f'Bundled demo file is missing: {path}. Restore examples/BRATS_409 from the repository.')
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundled demo file changed: {filename}. Restore the repository copy before demonstrating.')
    return {'image': EXAMPLE / 'image.nii.gz', 'reference': EXAMPLE / 'reference.nii.gz'}
