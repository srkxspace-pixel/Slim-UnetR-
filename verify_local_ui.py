"""Opt-in integration check: uses only existing real BRATS_409 MRI data.

Run with the local UI server running. No training, generated MRI or changed weights.
"""
import argparse
import http.cookiejar
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--skip-inference', action='store_true')
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    page = opener.open(base, timeout=15).read().decode()
    token = re.search(r'name="ui-token" content="([^"]+)"', page)[1]

    def request(path, data=None, headers=None, expected=200):
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            response = opener.open(req, timeout=60)
        except urllib.error.HTTPError as error:
            response = error
        payload = response.read()
        assert response.code == expected, (path, response.code, payload[:500])
        return payload

    assert json.loads(request('/api/status'))['upload_limit_mb'] == 384
    request('/api/status', headers={'Origin': 'https://example.com'}, expected=403)
    request('/api/status', headers={'Host': 'example.com'}, expected=400)
    request('/api/predict', data=b'', expected=403)
    assert b'Brain MRI segmentation' in page.encode()
    print('PASS: HTTP page, host/origin/token protection', flush=True)
    if args.skip_inference:
        return
    import nibabel as nib
    import numpy as np
    from src.demo_case import demo_case
    files = demo_case()
    image_path, reference_path = files['image'], files['reference']
    assert image_path.is_file() and reference_path.is_file(), 'Existing real example is required.'
    initialization_deadline = time.monotonic() + 180
    while json.loads(request('/api/status'))['model']['state'] == 'loading':
        if time.monotonic() > initialization_deadline:
            raise TimeoutError('Model initialization exceeded three minutes.')
        time.sleep(1)
    assert json.loads(request('/api/status'))['model']['state'] == 'ready'

    def upload(files, mode, scheme=''):
        boundary = 'slim-' + uuid.uuid4().hex
        body = io.BytesIO()
        for key, value in {'mode': mode, 'confirmed': 'yes', 'label_scheme': scheme}.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for role, path in files.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{role}"; filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            body.write(path.read_bytes())
            body.write(b'\r\n')
        body.write(f'--{boundary}--\r\n'.encode())
        ident = json.loads(request('/api/predict', body.getvalue(), {'X-UI-Token': token, 'Content-Type': f'multipart/form-data; boundary={boundary}'}, 202))['id']
        print('Uploaded real case:', mode, ident, flush=True)
        request('/api/real-example', data=b'', headers={'X-UI-Token': token}, expected=409)
        last_stage = None
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = json.loads(request(f'/api/runs/{ident}'))
            if job['stage'] != last_stage:
                print(job['stage'], job['detail'], flush=True)
                last_stage = job['stage']
            if job['state'] != 'running':
                assert job['state'] == 'completed', job
                return ident, job['result']
            time.sleep(1)
        raise TimeoutError('Prediction exceeded ten minutes.')

    first, report = upload({'image': image_path, 'reference': reference_path}, 'four_dimensional', 'msd')
    assert report['checkpoint_unchanged'] and report['model_state_unchanged']
    assert report['training'] is False and report['synthetic_inputs'] is False
    assert report['metrics'] and report['reference_provided']
    for axis in ('axial', 'coronal', 'sagittal'):
        for panel in ('mri', 'reference', 'prediction'):
            png = request(f'/api/runs/{first}/slice?axis={axis}&index=80&modality=t1ce&panel={panel}&region=ET&opacity=.6')
            assert png.startswith(b'\x89PNG\r\n\x1a\n')
    request(f'/api/runs/{first}/slice?index=99999', expected=400)
    request(f'/api/runs/{first}/download?file=../../config.yml', expected=400)
    request(f'/api/runs/{first}/download?file=input_image.nii.gz', expected=400)
    bundle = zipfile.ZipFile(io.BytesIO(request(f'/api/runs/{first}/download?file=prediction_bundle.zip')))
    assert {'report.json', 'prediction_TC.nii.gz', 'prediction_WT.nii.gz', 'prediction_ET.nii.gz'} <= set(bundle.namelist())
    for region in ('TC', 'WT', 'ET'):
        image = nib.load(ROOT / 'outputs/ui_runs' / first / f'prediction_{region}.nii.gz')
        assert image.shape == tuple(report['processed_shape'])
        assert np.allclose(image.affine, report['processed_affine'])
        assert image.header.get_xyzt_units()[0] == 'mm'
    print('PASS: real 4D upload, reference metrics, all planes, downloads and invalid-view rejection', flush=True)

    # Split the four acquired channels without changing any voxel intensities.
    split_dir = ROOT / 'outputs/ui_verification/real_BRATS_409_modalities'
    split_dir.mkdir(parents=True, exist_ok=True)
    source = nib.load(image_path)
    files = {}
    for i, role in enumerate(('flair', 't1', 't1ce', 't2')):
        path = split_dir / f'BRATS_409_{role}.nii.gz'
        channel = nib.Nifti1Image(np.asarray(source.dataobj[..., i]), source.affine, source.header.copy())
        channel.header.set_xyzt_units('mm')
        nib.save(channel, path)
        assert np.array_equal(np.asarray(nib.load(path).dataobj), np.asarray(source.dataobj[..., i]))
        files[role] = path
    second, no_reference = upload(files, 'four_files')
    assert no_reference['metrics'] is None and not no_reference['reference_provided']
    first_masks = np.load(ROOT / 'outputs/ui_runs' / first / 'prediction.npy')
    second_masks = np.load(ROOT / 'outputs/ui_runs' / second / 'prediction.npy')
    assert np.array_equal(first_masks, second_masks), 'Four-file and 4D predictions differ.'
    previous = ROOT / 'outputs/local_validation_97/BRATS_409_pred_regions.nii.gz'
    matches_previous = None
    if previous.is_file():
        matches_previous = np.array_equal(first_masks, np.moveaxis(np.asarray(nib.load(previous).dataobj), -1, 0))
        assert matches_previous, 'The UI result differs from the saved local evaluation.'
    assert no_reference['checkpoint_unchanged'] and no_reference['model_state_unchanged']
    evidence = {'real_case': str(image_path), 'four_dimensional_run': first, 'four_files_run': second,
                'predictions_identical': True, 'no_training': True, 'no_synthetic_inputs': True,
                'matches_previous_local_evaluation': matches_previous,
                'metrics': report['metrics'], 'four_file_metrics': None,
                'verified': ['HTTP upload in both formats', 'optional real reference', 'no reference / no metrics',
                             'all three planes and panels', 'downloaded ZIP and NIfTI geometry',
                             'CSRF/host/origin checks', 'concurrent inference rejected', 'invalid view and download rejected'],
                'checkpoint_sha256': report['checkpoint_sha256']}
    destination = ROOT / 'outputs/ui_verification/verification.json'
    destination.write_text(json.dumps(evidence, indent=2))
    print('PASS: actual four-file upload produces identical predictions; no fabricated metrics', flush=True)
    print('Evidence:', destination, flush=True)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
