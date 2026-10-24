# Official Slim UNETR inference run

Completed on 2026-10-18 using the authors' BraTS 2021 checkpoint and the existing local MSD Task01 BrainTumour dataset. No training, fine-tuning, synthetic input volumes, or random data augmentation was performed. No original project source or configuration was modified; a separate inference runner was added.

## Checkpoint provenance

- Authors' repository: https://github.com/deepang-ai/Slim-UNETR
- Release: https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0
- Release date from GitHub API: 2025-01-19
- Asset: https://github.com/deepang-ai/Slim-UNETR/releases/download/v1.0.0/brats21_pytorch_model.bin
- Local file: `model_store/official_brats2021/brats21_pytorch_model.bin`
- File size: 7,313,439 bytes, matching the release asset metadata.
- Local SHA256: `8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4`
- The publisher did not supply a checksum in the release API. The recorded checksum identifies the downloaded file; it is not a publisher-signed checksum.
- The release also provides hepatic-vessel weights; it does not list an MSD Task01 brain checkpoint.
- All four local model files (`SlimUNETR.py`, `Encoder.py`, `Decoder.py`, and `Slim_UNETR_Block.py`) matched the release tag's source after normalizing line endings and trailing whitespace.

## Run

From the project directory in PowerShell:

```powershell
& ./.venv/Scripts/python.exe -u inference_official.py --num-cases 3
```

The existing environment was used: Python 3.11.9, PyTorch 2.5.1+cu121, MONAI 1.2.0, NumPy 1.26.4, NVIDIA GeForce GTX 1650. No packages were installed. The strictly loaded local model has 1,783,425 parameters.

The first three cases in the existing validation partition were selected, beginning at manifest index 387 (zero-based). Preprocessing uses the project's deterministic validation operations: channel handling, TC/WT/ET conversion, RAS orientation, 1 mm spacing, and nonzero per-channel normalization. Inference uses 128-cubed sliding windows, overlap 0.5, batch size one, sigmoid, and threshold 0.5. The original `inference.py` was not executed because its warm-up updates weights.

## Observed results

These are whole-volume Dice scores, rather than scores for the single displayed slice.

| Case | Tumour core | Whole tumour | Enhancing tumour | Inference seconds |
| --- | ---: | ---: | ---: | ---: |
| BRATS_409 | 0.963294 | 0.965141 | 0.949706 | 6.156 |
| BRATS_258 | 0.900414 | 0.871396 | 0.918677 | 2.038 |
| BRATS_346 | 0.926550 | 0.833860 | 0.833548 | 1.903 |

Timing covers sliding-window prediction and thresholding; it excludes loading, preprocessing, plotting, and saving. The first case includes first-call GPU overhead. These measurements do not reproduce the paper's throughput protocol.

Each case has a PNG comparison and a prediction NIfTI volume. NIfTI predictions store binary TC, WT, and ET masks along their final axis, in preprocessed RAS / 1 mm space. These three cases retained the original canonical image grid. The PNG shows the axial slice with the greatest ground-truth whole-tumour area. Green is WT-only, yellow is TC excluding ET, and red is ET; independently predicted regions can overlap imperfectly, so the PNG uses ET then TC then WT priority.

`run_report.json` records source image and label hashes, checkpoint provenance, metrics, runtime, output locations, and state verification. The model state and checkpoint SHA256 were unchanged after inference. The downloaded checkpoint remains reusable; no optimizer was created or checkpoint saved by the runner.

## Limits

This is a BraTS 2021 checkpoint applied to MSD Task01 images. It is not an MSD-trained checkpoint, a reproduction of the paper's BraTS 2021 score, or evaluation of the complete validation dataset. Patient overlap between the released model's training data and these MSD cases has not been established; these scores must not be presented as independent held-out generalization results. The local medical data was read locally and was not uploaded.
