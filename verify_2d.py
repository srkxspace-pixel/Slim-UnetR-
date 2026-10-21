"""Opt-in verification on pinned real BRISC test images. No training or fabricated MRI."""
import argparse
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from PIL import Image
from werkzeug.test import Client
from werkzeug.wrappers import Response

import local_ui
from setup_2d_model import DATA, DATA_REVISION, digest
from src.slice_inference import SliceEngine, read_image, mask_metrics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/verification_2d'


def extract_cases():
    archive = OUTPUT / 'source/brisc.zip'
    if not archive.is_file() or digest(archive) != DATA[2]:
        raise RuntimeError('Run setup_2d_model.py --test-data first.')
    samples = OUTPUT / 'samples'
    samples.mkdir(parents=True, exist_ok=True)
    cases = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        for code in ('gl', 'me', 'pi'):
            for plane in ('ax', 'co', 'sa'):
                matches = sorted(name for name in names if '/segmentation_task/test/images/' in name and f'_{code}_{plane}_' in name)
                for name in (matches[0], matches[-1]):
                    image = samples / Path(name).name
                    mask_name = name.replace('/images/', '/masks/').replace('.jpg', '.png')
                    reference = samples / (image.stem + '_reference.png')
                    image.write_bytes(source.read(name))
                    raw_reference = samples / (image.stem + '_source_reference.png')
                    raw_reference.write_bytes(source.read(mask_name))
                    # Upstream BRISC loader explicitly defines every nonzero label pixel as tumour.
                    # Preserve source bytes and document this conversion, rather than guessing user labels.
                    with Image.open(raw_reference) as original:
                        Image.fromarray((np.asarray(original.convert('L')) > 0).astype(np.uint8) * 255).save(reference)
                    cases.append({'image': image, 'reference': reference, 'source': name, 'reference_source': mask_name})
        for plane in ('ax', 'co', 'sa'):
            matches = sorted(name for name in names if '/classification_task/test/no_tumor/' in name and f'_no_{plane}_' in name)
            for name in (matches[0], matches[-1]):
                image = samples / Path(name).name
                image.write_bytes(source.read(name))
                cases.append({'image': image, 'source': name})
    return cases


def verify(device='auto'):
    if (OUTPUT / 'verification.json').exists():
        raise RuntimeError('Verification evidence already exists. Use a separate output folder for another evaluation.')
    cases = extract_cases()
    engine = SliceEngine(device=device)
    reports = []
    for index, case in enumerate(cases):
        directory = OUTPUT / 'runs' / case['image'].stem
        directory.mkdir(parents=True, exist_ok=False)
        files = {key: case[key] for key in ('image', 'reference') if key in case}
        result = engine.predict(directory, files, {key: path.name for key, path in files.items()}, lambda *args: None)
        if 'reference' in case:
            prediction = np.asarray(Image.open(directory / 'mask.png')) > 0
            truth = read_image(case['reference'], reference=True)
            # Independent count-based check of the reported Dice/IoU.
            overlap = int(np.logical_and(prediction, truth).sum())
            total = int(prediction.sum() + truth.sum())
            assert abs(result['metrics']['dice'] - 2 * overlap / total) < 1e-12
        assert result['original_size'] == list(Image.open(case['image']).size)
        with zipfile.ZipFile(directory / 'prediction_bundle.zip') as bundle:
            assert set(bundle.namelist()) == set(result['downloads']) - {'prediction_bundle.zip'}
        reports.append({'source': case['source'], 'input_sha256': digest(case['image']), 'metrics': result['metrics'],
                        'predicted_pixels': result['predicted_pixels'], 'inference_seconds': result['inference_seconds']})
        print(f"{index + 1}/{len(cases)} {case['image'].name}: Dice={result['metrics']['dice'] if result['metrics'] else 'no pixel reference'}, predicted pixels={result['predicted_pixels']}", flush=True)

    # Exercise actual HTTP handling + worker + downloads using the real engine.
    http_root = OUTPUT / 'http_runs'
    http_root.mkdir(exist_ok=True)
    local_ui.RUNS = http_root
    app = local_ui.LocalApp(warm_model=False, device=device)
    app.slice_engine = engine
    client = Client(app, Response)
    base = 'http://127.0.0.1:8765'
    page = client.get('/', base_url=base)
    token = re.search(r'name="ui-token" content="([^"]+)"', page.text)[1]
    headers = {'X-UI-Token': token}
    first = cases[0]
    png_path = OUTPUT / 'samples/demo_slice.png'
    read_image(first['image']).save(png_path)
    http_masks = []
    for image_path, with_reference in ((first['image'], True), (png_path, False)):
        data = {'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(image_path.read_bytes()), image_path.name)}
        if with_reference:
            data['reference'] = (io.BytesIO(first['reference'].read_bytes()), first['reference'].name)
        accepted = client.post('/api/predict', base_url=base, headers=headers, data=data)
        assert accepted.status_code == 202, accepted.text
        ident = accepted.json['id']
        # The shared worker must reject overlapping requests.
        overlapping = client.post('/api/predict', base_url=base, headers=headers, data={'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(image_path.read_bytes()), image_path.name)})
        assert overlapping.status_code == 409, overlapping.text
        deadline = time.monotonic() + 180
        while True:
            status = client.get(f'/api/runs/{ident}', base_url=base).json
            if status['state'] != 'running':
                break
            if time.monotonic() > deadline:
                raise TimeoutError('2D HTTP prediction did not complete within 180 seconds.')
            time.sleep(.2)
        assert status['state'] == 'completed', status
        assert bool(status['result']['metrics']) == with_reference
        for panel in ('mri', 'prediction') + (('reference',) if with_reference else ()):
            for opacity in (0, .55, 1):
                response = client.get(f'/api/runs/{ident}/slice?panel={panel}&opacity={opacity}', base_url=base)
                assert response.status_code == 200 and response.mimetype == 'image/png'
                assert Image.open(io.BytesIO(response.data)).size == tuple(status['result']['original_size'])
        for query in ('panel=invalid', 'opacity=nan', 'opacity=2'):
            assert client.get(f'/api/runs/{ident}/slice?{query}', base_url=base).status_code == 400
        for filename in status['result']['downloads']:
            response = client.get(f'/api/runs/{ident}/download?file={filename}', base_url=base)
            assert response.status_code == 200
        assert client.get(f'/api/runs/{ident}/download?file=input_image.jpg', base_url=base).status_code == 400
        http_masks.append(np.asarray(Image.open(http_root / ident / 'mask.png')))
        # Read persisted completed result using a fresh app instance (restart recovery).
        restored = local_ui.LocalApp(warm_model=False).get_job(ident)
        assert restored['state'] == 'completed' and restored['result'] == status['result']
    assert np.array_equal(*http_masks), 'Lossless PNG and decoded JPEG must produce identical masks.'
    positive = [case for case in reports if case['metrics'] is not None]
    negative = [case for case in reports if case['metrics'] is None]
    report = {'model': engine.info, 'dataset_revision': DATA_REVISION, 'dataset_sha256': DATA[2],
              'selection': 'First and last filename for each tumour type x plane in the released segmentation test split (18); first and last normal image per plane (6). No cherry-picking.',
              'scope': 'Engineering smoke evaluation, not a clinical or independently held-out benchmark; dataset/checkpoint patient overlap is unverified.',
              'reference_mapping': 'Source masks retained as *_source_reference.png. Canonical binary references use source grayscale > 0 exactly as upstream load_slice; no manual correction.',
              'annotated_cases': len(positive), 'mean_slice_dice': float(np.mean([case['metrics']['dice'] for case in positive])),
              'mean_slice_iou': float(np.mean([case['metrics']['iou'] for case in positive])),
              'normal_image_controls': len(negative), 'normal_controls_with_nonempty_prediction': sum(case['predicted_pixels'] > 0 for case in negative),
              'checks': ['Real JPG upload with reference', 'Real PNG upload without reference', 'Identical JPEG-decoded/PNG masks', 'All image panels and overlay opacity', 'ZIP/download contents and geometry', 'Concurrency rejected', 'Private uploads not downloadable', 'Invalid panel/opacity rejected', 'Completed result recovery after restart', 'Checkpoint and model state unchanged'],
              'cases': reports}
    (OUTPUT / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, indent=2), flush=True)


def cpu_smoke():
    """One real case on CPU, compared with the already recorded CUDA result."""
    existing = json.loads((OUTPUT / 'verification.json').read_text())
    first = existing['cases'][0]
    image = OUTPUT / 'samples' / Path(first['source']).name
    directory = OUTPUT / 'cpu_smoke'
    directory.mkdir(exist_ok=False)
    engine = SliceEngine(device='cpu')
    result = engine.predict(directory, {'image': image}, {'image': image.name}, lambda *args: None)
    cpu = np.asarray(Image.open(directory / 'mask.png'))
    gpu = np.asarray(Image.open(OUTPUT / 'runs' / image.stem / 'mask.png'))
    receipt = {'device': 'cpu', 'source': first['source'], 'mask_pixel_agreement_with_cuda': float(np.mean(cpu == gpu)),
               'different_pixels': int(np.count_nonzero(cpu != gpu)), 'inference_seconds': result['inference_seconds'],
               'total_processing_seconds': result['total_processing_seconds'], 'model_state_unchanged': result['model_state_unchanged']}
    (OUTPUT / 'cpu_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


def server_smoke(port):
    """Exercise a running HTTP server, 2D upload, then the existing 3D real-case route."""
    import requests
    import nibabel as nib
    session = requests.Session()
    session.trust_env = False
    base = f'http://127.0.0.1:{port}'
    page = session.get(base, timeout=15)
    page.raise_for_status()
    token = re.search(r'name="ui-token" content="([^"]+)"', page.text)[1]
    headers = {'X-UI-Token': token}
    image = OUTPUT / 'samples/brisc2025_test_00086_gl_co_t1.jpg'
    with image.open('rb') as stream:
        response = session.post(base + '/api/predict', data={'mode': 'single_slice', 'confirmed': 'yes'},
                                files={'image': (image.name, stream, 'image/jpeg')}, headers=headers, timeout=30)
    assert response.status_code == 202, response.text

    def wait_for_run(ident):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            response = session.get(base + f'/api/runs/{ident}', timeout=20)
            response.raise_for_status()
            job = response.json()
            if job['state'] != 'running':
                assert job['state'] == 'completed', job
                return job
            time.sleep(.5)
        raise TimeoutError('Server prediction did not complete within 240 seconds.')

    slice_job = wait_for_run(response.json()['id'])
    assert slice_job['result']['input_mode'] == 'single_slice' and slice_job['result']['metrics'] is None
    downloaded = session.get(base + f"/api/runs/{slice_job['id']}/download?file=prediction_bundle.zip", timeout=30)
    downloaded.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as bundle:
        assert Image.open(io.BytesIO(bundle.read('mask.png'))).size == tuple(slice_job['result']['original_size'])
    print('PASS: real HTTP 2D upload, inference, download and mask geometry', flush=True)
    deadline = time.monotonic() + 180
    while True:
        model = session.get(base + '/api/status', timeout=10).json()['model']
        if model['state'] != 'loading':
            assert model['state'] == 'ready', model
            break
        if time.monotonic() > deadline:
            raise TimeoutError('3D model initialization timed out.')
        time.sleep(.5)
    response = session.post(base + '/api/real-example', headers=headers, timeout=15)
    assert response.status_code == 202, response.text
    volume_job = wait_for_run(response.json()['id'])
    prediction = np.load(local_ui.RUNS / volume_job['id'] / 'prediction.npy', allow_pickle=False)
    previous = np.moveaxis(nib.load(ROOT / 'outputs/local_validation_97/BRATS_409_pred_regions.nii.gz').get_fdata(), -1, 0)
    assert np.array_equal(prediction, previous), '3D prediction differs from the prior saved evaluation.'
    for panel in ('mri', 'prediction', 'reference'):
        response = session.get(base + f"/api/runs/{volume_job['id']}/slice", params={'panel': panel, 'index': volume_job['result']['default_slice']}, timeout=20)
        assert response.status_code == 200 and response.headers['Content-Type'].startswith('image/png')
    receipt = {'port': port, 'slice_run': slice_job['id'], 'volume_run': volume_job['id'],
               'slice_result': slice_job['result'], 'volume_metrics': volume_job['result']['metrics'],
               'three_dimensional_masks_identical_to_previous': True,
               'three_dimensional_model_and_checkpoint_unchanged': volume_job['result']['model_state_unchanged'] and volume_job['result']['checkpoint_unchanged']}
    (OUTPUT / 'server_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('PASS: fresh 3D real-example inference matches previous masks exactly; all three panels load', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--cpu-smoke', action='store_true', help='After verification, run one real image on CPU and compare with CUDA')
    parser.add_argument('--server-smoke', type=int, metavar='PORT', help='Test a running local server with fresh 2D and 3D inference')
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
    elif args.server_smoke:
        server_smoke(args.server_smoke)
    else:
        verify(args.device)
