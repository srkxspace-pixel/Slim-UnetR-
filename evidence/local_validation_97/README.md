# Local evaluation of official Slim UNETR weights on 97 MSD cases

These are our local evaluation results, not reproduced paper results. The authors' official BraTS 2021 checkpoint was applied to all 97 cases in the existing local MSD Task01 validation partition. Patient overlap with the checkpoint's training population is unverified.

## Results

| Region | Mean Dice | Dice cases | Primary mean HD95 in mm | Finite-only mean HD95 in mm | Finite / infinite HD95 cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tumour core (TC) | 81.3034% | 97 | 23.4461 | 23.4461 | 97 / 0 |
| Whole tumour (WT) | 85.3720% | 97 | 34.8553 | 34.8553 | 97 / 0 |
| Enhancing tumour (ET) | 75.0573% | 95 | Infinity | 20.1872 | 95 / 2 |
| Overall mean of the three regional means | 80.5776% | All 97 cases processed | Infinity | 26.1629 | See region counts above |

All 97 requested cases completed. No cases were skipped or replaced. Overall values are the equal-weight mean of TC, WT, and ET case means; Dice is not pooled across voxels or lesion sizes.

## Empty-region policy

Dice follows the existing MONAI convention `ignore_empty=True`: a region with no ground-truth foreground is undefined and excluded from that region's Dice mean. Both BRATS_026 and BRATS_024 have empty ET ground truth and nonempty ET predictions. Consequently, ET Dice has 95 defined cases; TC and WT each have 97.

HD95 is the maximum of the two directed 95th-percentile Euclidean surface distances, measured using the preprocessed image's physical voxel spacing in millimetres. If both masks are empty, HD95 is undefined and excluded; this did not occur in this run. If exactly one is empty, HD95 is infinite and retained in the primary mean. Thus the two ET false-positive cases make primary ET and overall HD95 infinite. No arbitrary finite penalty was substituted. This explicit empty-mask policy avoids silently dropping failures when percentile calculations return NaN for infinite surface distances.

The finite-only result is supplemental: it excludes those two ET region measurements, retaining 97 TC, 97 WT, and 95 ET measurements. Its overall value is the mean of those three regional finite-only means. It must not be presented as finite HD95 across every region in all 97 cases, or as the paper's scoring policy.

## Data and checkpoint

- Dataset: `D:/SRM Projects/datasets/Task01_BrainTumour`.
- Local split: existing `dataset.json` training-list order, first 387 cases excluded from this evaluation and final 97 cases evaluated, as selected by the existing 0.8 train ratio. This is not the authors' verified checkpoint split.
- Weights: `model_store/official_brats2021/brats21_pytorch_model.bin`.
- Official source: https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0
- Checkpoint SHA256: `8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4`.
- The model loaded all weights strictly, ran with `eval()`, gradients disabled, and `torch.inference_mode()`. No optimizer was created. Model state and checkpoint SHA256 were unchanged after all 97 cases.
- Existing MRI volumes were used; no synthetic inputs, training, fine-tuning, random data augmentation, or prediction postprocessing was performed. The original training and evaluation scripts were not executed.

## Evaluation settings

Modality order is FLAIR, T1w, t1gd, T2w. MSD labels are converted to overlapping TC, WT, ET masks using the existing loader. Images and labels use RAS orientation and 1 mm spacing, followed by nonzero per-channel intensity normalization for images. Sliding-window inference uses 128-cubed windows, overlap 0.5, one window at a time, and a sigmoid probability strictly greater than 0.5 for binary predictions. Prediction windows run on the NVIDIA GeForce GTX 1650 with full-volume accumulation on CPU.

SciPy 1.14.1 was missing and was installed as the CPU HD95 evaluation dependency. Existing model, NumPy, PyTorch, and MONAI versions were retained; runtime versions are recorded in `run_report.json`. The evaluation dependency is pinned in `requirements-evaluation.txt`.

## Files and verification

- `run_report.json`: summary, metric definitions, all per-case results, case IDs, source hashes, checkpoint provenance, and before/after model-state hashes.
- `per_case_metrics.csv`: all 97 case-level TC/WT/ET Dice and HD95 values; blank Dice means undefined, and `Infinity` is explicit.
- `case_results.jsonl`: per-case records written during execution.
- `verification.json`: saved-volume, score, aggregate, and hash verification results.
- All 97 `*_pred_regions.nii.gz` files contain binary TC, WT, and ET masks in the final dimension, in preprocessed RAS / 1 mm space. They are not integer-labelled tumour maps.
- PNGs show the first three cases only, as MRI, ground truth, and prediction comparisons. The displayed slice is chosen by the largest ground-truth WT area; reported Dice covers each complete volume.

Before the full evaluation, HD95 was checked on a previously saved real case using an independent KD-tree nearest-surface calculation. Final verification reopens all 97 saved predictions and compares Dice with the original labels, and checks HD95 independently on two cases with large distances. No synthetic test images are involved.

## Reproduce this local evaluation

From the project root in PowerShell, use a fresh output directory so previous results are preserved:

```powershell
& ./.venv/Scripts/python.exe -u inference_official.py --num-cases 97 --save-figures 3 --output-dir outputs/local_validation_97_repeat
```

This command reuses the existing checkpoint and dataset and performs inference only. Running more cases or these local scores does not resolve the unknown patient overlap or reproduce the paper's BraTS or MSD benchmark numbers.
