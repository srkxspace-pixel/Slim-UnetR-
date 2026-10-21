"""Run fresh Slim UNETR inference on the bundled real MRI and its reference, without a browser."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

from src.demo_case import demo_case

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing results are never overwritten')
    args = parser.parse_args()
    files = demo_case()
    destination = args.output_dir or ROOT / 'outputs' / ('demo_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=False)
    from src.local_inference import Engine, render_slice
    engine = Engine(device=args.device)
    result = engine.predict(destination, files, {key: path.name for key, path in files.items()},
                            'four_dimensional', 'msd', lambda stage, detail, *args: print(f'{stage}: {detail}', flush=True))
    for panel in ('mri', 'reference', 'prediction'):
        png = render_slice(destination, 'axial', result['default_slice'], 'flair', panel, 'all', .55)
        (destination / f'{panel}.png').write_bytes(png)
    print(f'Completed fresh inference on {engine.info["gpu"] or "CPU"}. Results: {destination}', flush=True)
    for region, metrics in result['metrics'].items():
        print(f'{region}: Dice {metrics["dice"] * 100:.2f}%, HD95 {metrics["hd95_mm"]} mm', flush=True)
    print('Local case comparison; not a reproduction of the paper benchmark.', flush=True)


if __name__ == '__main__':
    main()
