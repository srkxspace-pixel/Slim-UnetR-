"""Recalculate saved metrics and compare local files against a freshly downloaded pinned release.

No inference, optimization, prediction editing or training. Paper table values are visually
transcribed from the author's table images; this is not a full-paper protocol reproduction.
"""
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
UPSTREAM = AUDIT / 'upstream'
sys.path.insert(0, str(ROOT))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def normalized(path):
    return '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8-sig').splitlines()).strip()


def run():
    report = json.loads((ROOT / 'outputs/local_validation_97/run_report.json').read_text())
    with (ROOT / 'outputs/local_validation_97/per_case_metrics.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(report['cases']) == 97
    assert [row['case'] for row in rows] == report['split']['case_ids']
    source_checks = {}
    for path in ('src/SlimUNETR/SlimUNETR.py', 'src/SlimUNETR/Encoder.py', 'src/SlimUNETR/Decoder.py',
                 'src/SlimUNETR/Slim_UNETR_Block.py', 'src/loader.py', 'inference.py', 'config.yml',
                 'figures/table1.jpg', 'figures/table2.jpg', 'figures/table3.jpg'):
        local, remote = ROOT / path, UPSTREAM / path.replace('/', '__')
        check = {'local_sha256': sha(local), 'upstream_sha256': sha(remote), 'bytes_identical': sha(local) == sha(remote)}
        if local.suffix != '.jpg':
            check['normalized_text_identical'] = normalized(local) == normalized(remote)
            if not check['normalized_text_identical']:
                diff = '\n'.join(difflib.unified_diff(normalized(remote).splitlines(), normalized(local).splitlines(),
                                                       fromfile='upstream/' + path, tofile='local/' + path, lineterm=''))
                (AUDIT / (path.replace('/', '__') + '.diff')).write_text(diff, encoding='utf-8')
        source_checks[path] = check
    local_checkpoint = ROOT / 'model_store/official_brats2021/brats21_pytorch_model.bin'
    checkpoint = {'local_sha256': sha(local_checkpoint), 'fresh_release_sha256': sha(UPSTREAM / 'brats21_pytorch_model.bin')}
    assert checkpoint['local_sha256'] == checkpoint['fresh_release_sha256'] == report['checkpoint_sha256']
    checkpoint['fresh_download_identical'] = True
    checkpoint['publisher_signed_checksum_available'] = False
    paper_brats = {'WT': 93.77, 'ET': 89.36, 'TC': 94.19, 'overall': 92.44,
                   'hd95_mm': {'WT': 7.34, 'ET': 4.50, 'TC': 4.24, 'overall': 5.36}}
    paper_msd = {'WT': 89.38, 'ET': 65.70, 'TC': 85.73, 'overall': 80.27}
    local_metrics = {}
    for region in ('TC', 'WT', 'ET'):
        dice = [float(row['dice_' + region]) for row in rows if row['dice_' + region]]
        hd = [float(row['hd95_mm_' + region]) for row in rows if row['hd95_mm_' + region]]
        mean = statistics.mean(dice)
        hd_mean = statistics.mean(hd)
        assert abs(mean - report['summary']['dice'][region]['mean']) < 1e-12
        for row, case in zip(rows, report['cases']):
            value = float(row['dice_' + region]) if row['dice_' + region] else None
            assert value == case['dice'][region]
            assert float(row['hd95_mm_' + region]) == float(case['hd95_mm'][region])
        expected_hd = float(report['summary']['hd95_mm'][region]['mean'])
        assert hd_mean == expected_hd or abs(hd_mean - expected_hd) < 1e-10
        local_metrics[region] = {'dice_percent': mean * 100, 'defined_dice_cases': len(dice),
                                 'hd95_mm': 'Infinity' if math.isinf(hd_mean) else hd_mean,
                                 'finite_only_hd95_mm': statistics.mean([value for value in hd if math.isfinite(value)]),
                                 'infinite_hd95_cases': sum(math.isinf(value) for value in hd),
                                 'difference_from_paper_msd_percentage_points': mean * 100 - paper_msd[region]}
    overall = statistics.mean(item['dice_percent'] for item in local_metrics.values())
    assert abs(overall / 100 - report['summary']['overall_dice']) < 1e-12
    local_metrics['overall'] = {'dice_percent': overall, 'difference_from_paper_msd_percentage_points': overall - paper_msd['overall']}
    import torch
    import yaml
    from src.SlimUNETR.SlimUNETR import SlimUNETR
    config = yaml.safe_load((ROOT / 'config.yml').read_text())['brain2019']['slim_unetr']
    model = SlimUNETR(**config)
    weights = torch.load(local_checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    checkpoint['strict_cpu_weight_loading_passed'] = True
    checkpoint['parameters'] = sum(value.numel() for value in model.parameters())
    checkpoint['unchanged_after_loading'] = sha(local_checkpoint) == checkpoint['local_sha256']
    dataset = json.loads(Path(yaml.safe_load((ROOT / 'config.yml').read_text())['brain2019']['data_root'], 'dataset.json').read_text())
    two_d = json.loads((ROOT / 'outputs/verification_2d/verification.json').read_text())
    receipt = {'audit_date_local': '2026-10-22', 'paper_doi': '10.1109/TMI.2023.3326188',
               'upstream_release_commit': '6ae2f4e7498102fc17c823f28ca876a47fdbf1e7',
               'verdict': 'Official core implementation and checkpoint confirmed; published benchmark reproduction NOT established.',
               'paper_numbers_source': 'Authors release figures/table1.jpg, table2.jpg, table3.jpg; visually read; compared bytewise with local copies. Abstract independently confirms BraTS mean Dice 92.44%.',
               'full_paper_methods_retrieved': False, 'source_checks': source_checks, 'checkpoint': checkpoint,
               'paper_brats2021': paper_brats, 'paper_msd_task01': paper_msd, 'local_97_case_recomputed_metrics': local_metrics,
               'local_dataset_labels': dataset['labels'], 'local_dataset_modality': dataset['modality'],
               'notable_protocol_differences': [
                   'Local 97 MSD Task01 cases use released BraTS2021 weights, not a verified Task01 checkpoint.',
                   'Exact paper evaluation cases and released-checkpoint training membership are unverified.',
                   'Upstream MSD ET conversion uses label 2; local ET conversion uses label 3, matching local dataset manifest.',
                   'Original inference.py calls optimizer warm_up; local frozen runner does not update weights. Paper-time use of this code is unconfirmed.',
                   'Local HD95 empty-mask and aggregation policy is explicit; agreement with paper scoring policy is unconfirmed.',
                   'JPG/PNG uses a separate BRISC SegFormer model; its slice metrics do not reproduce Slim UNETR.'
               ],
               'two_d_scope': {'model': two_d['model']['name'], 'annotated_slice_count': two_d['annotated_cases'],
                               'mean_dice_percent': two_d['mean_slice_dice'] * 100, 'comparable_to_slim_unetr_paper': False},
               'no_inference_or_training_performed_in_this_audit': True,
               'limitations': 'This audit recalculates saved metric records and checks files/strict loading. It does not recompute every voxel-level metric or establish independence from checkpoint training. Full paper/supplement not accessible through inspected publisher/author links.'}
    (AUDIT / 'verification.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'checkpoint': checkpoint, 'source_matches': {key: value.get('normalized_text_identical', value['bytes_identical']) for key, value in source_checks.items()},
                      'local_metrics': local_metrics, 'dataset_labels': dataset['labels']}, indent=2), flush=True)


if __name__ == '__main__':
    run()
