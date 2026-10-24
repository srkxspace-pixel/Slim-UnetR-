# Project handoff — Slim UNETR local MRI segmentation demo

Read this first. It says what the project is, who made which part, how to run it with no retraining, and where the saved results are.

## 1. What this project is (and is not)

A **local, offline-capable brain-tumour MRI segmentation application** built around the **authors' released Slim UNETR architecture and pretrained BraTS2021 checkpoint**. You upload a 4-modality 3D MRI, the pretrained model predicts tumour core (TC), whole tumour (WT) and enhancing tumour (ET), and you inspect, score and export the result in a browser.

**No training or fine-tuning was used to produce any weights or results in this bundle.** The checkpoint is the authors' release, SHA256-verified. Nothing needs retraining.

### Who made what

| Part | Origin |
| --- | --- |
| `src/SlimUNETR/*.py`, `main.py`, `train.sh`, `weight_test.py`, `visualize.py`, `src/optimizer.py`, `src/utils.py`, `figures/`, the pretrained checkpoint, the paper | **Upstream authors** (Pang et al., deepang-ai, MIT) — unmodified or nearly so |
| `config.yml`, `src/loader.py` (MSD enhancing-tumour label 2→3), `inference.py` | Upstream files **with small local edits**; see `evidence/paper_cross_verification_20261022/*.diff`. `config.yml` is read on every inference run — do not delete or retune it |
| `local_ui.py`, `ui/`, `src/local_inference.py`, `src/demo_case.py`, `src/slice_inference.py`, `inference_official.py`, `demo.py`, `setup_demo.py`, `setup_2d_model.py`, `setup_windows.cmd`, `verify_*.py`, `tests/` (23 tests), `.github/workflows/`, `docs/`, `evidence/`, optional 2D (BRISC SegFormer) extension | **Written for this project** by its original developer (see git history) |
| 97-case evaluation of the official checkpoint, cross-verification against the authors' release | **Done for this project** on the original laptop; results saved in `evidence/` |

Attribution is part of the project and must stay: [NOTICE.md](NOTICE.md), [LICENSE](LICENSE), [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

**How to describe it truthfully.** If you installed and ran it yourself: *"I set up and ran a locally packaged demo of the authors' pretrained Slim UNETR model on a real MRI case and reviewed the saved validation evidence."* Do not claim you designed the network, trained the weights, wrote the application, or produced the paper's or the saved evaluation results unless you did. Assessors who ask will be told by the licence files and git history anyway.

## 2. Set up (no retraining, no dataset download)

Needs: Windows 10/11 **64-bit Intel/AMD** (not ARM), **Python 3.11 64-bit** from python.org with the **py launcher** ticked (check: `py -3.11 --version`; do not use the Microsoft Store Python), ~4 GB free disk, and internet for about 300 MB of packages (first run only, 10–25 minutes).

**From the handoff zip** (weights already included in `model_store/`):

1. Right-click the zip → Properties → tick **Unblock** if shown → **Extract All** to a short path such as `C:\SlimUNETR` (avoid OneDrive/Desktop). The files end up in `C:\SlimUNETR\Slim-UNETR\`.
2. Open that folder in a terminal (Shift+right-click → Open in Terminal) and run:

   ```powershell
   .\setup_windows.cmd
   ```

   Double-clicking also works; the window now stays open at the end. It creates `.venv`, installs pinned PyTorch 2.6.0 CPU and packages, verifies the checkpoint hash, runs the 23 tests and runs real inference on the bundled MRI. pip's "new version available" notice and many DeprecationWarnings are expected.
3. Run **`run_local_ui.cmd`** and open <http://127.0.0.1:8765>. Stop with Ctrl+C.

**From GitHub:** `git clone https://github.com/srkxspace-pixel/Slim-UnetR-.git`, then the same `setup_windows.cmd`; it also downloads the ~7 MB checkpoint and verifies its SHA256. This route has the code, docs and `evidence/` but not the full `outputs/` folders or the weights until setup downloads them. Manual commands and Linux/macOS notes are in [README.md](README.md).

### Troubleshooting

- *`py` not recognised / "Python 3.11 not found"*: re-run the Python installer → Modify → tick **py launcher**.
- *pip cannot download (proxy/firewall/certificate)*: hosts needed are `download.pytorch.org`, `download-r2.pytorch.org`, `pypi.org`, `files.pythonhosted.org`. Behind a proxy, `$env:HTTPS_PROXY="http://host:port"` first. Weights are already in the zip, so only pip needs internet.
- *`WinError 206` / path too long*: move the folder to a shorter path.
- *"Windows protected your PC"*: More info → Run anyway, or unblock the zip before extracting.
- *`curl /api/...` shows 403*: that is the intended protection. Health check is the page `/` returning 200.
- *Typing `cmd /c setup_windows.cmd` fails*: use `.\setup_windows.cmd` (leading `.\`).
- Anything else: [docs/TESTING.md](docs/TESTING.md), [LOCAL_UI.md](LOCAL_UI.md).

## 3. What you should see

Terminal (`demo.py`, also the last step of setup) on the bundled real case `examples/BRATS_409`:

| Region | Dice | HD95 (printed unrounded, ≈) |
| --- | ---: | ---: |
| Tumour core | 96.33% | 1.41 mm |
| Whole tumour | 96.51% | 1.73 mm |
| Enhancing tumour | 94.97% | 1.41 mm |

These describe **one case**, not overall model accuracy. The same numbers were reproduced by a clean install of this bundle on CPU (23 tests passed; UI integration checks passed).

Browser: click **Run real local example · BRATS_409** — a fresh prediction runs on the bundled files; then browse planes, slices, modalities, regions and opacity, and download the ZIP/report. **Rehearse this click-through once yourself** in Chrome or Edge: file-picker uploads were checked on the original laptop (screenshots in `evidence/ui_verification/`, taken on a GTX 1650), and automated checks cover the HTTP side. On CPU you will see device "CPU" and different timings, with the same Dice/HD95. Presenter script: [docs/DEMO.md](docs/DEMO.md).

## 4. Saved results from the original work

Produced on the original laptop (NVIDIA GTX 1650, PyTorch 2.5.1+cu121) and stored so nothing has to be re-run. `evidence/` is the curated subset (also in the repo). The zip additionally carries the raw folders under `outputs/` (masks, per-case files). `outputs/` is the original working folder: `local_documentation_archive/` and `local_validation_97/review/` contain historical notes whose `D:\` paths and wording are out of date — follow this file and README.md.

| Evidence | What it shows | Where |
| --- | --- | --- |
| 97-case evaluation of the official checkpoint on MSD Task01 | Mean Dice TC 81.30%, WT 85.37%, ET 75.06% (ET over 95 of 97 cases; ET HD95 infinite by the stated empty-mask policy). MSD cases of **unknown overlap** with the checkpoint's BraTS2021 training data: **not** a held-out accuracy and **not** a reproduction of the paper's 92.44% | `evidence/local_validation_97/` |
| Official checkpoint on 3 MSD cases | Comparison PNGs, run report | `evidence/official_brats2021_on_msd/` |
| Cross-verification against the authors' release | Checkpoint SHA256 match, source-file audit, diffs (the saved `config.yml.diff` predates the final `data_root` edit) | `evidence/paper_cross_verification_20261022/` |
| Browser UI verification | Screenshots and verification results | `evidence/ui_verification/` |
| Optional 2D slice verification | Dice/IoU on real BRISC slices (a different model) | `evidence/verification_2d/` |
| Bundled-case demo run | PNGs and `report.json` (CPU run in the original laptop's environment) | `evidence/bundled_case_demo/` |
| Clean-clone test record | 23 tests, CPU run, UI integration | [docs/VERIFICATION.md](docs/VERIFICATION.md) |

## 5. Not done / limits

- **Not a reproduction of the paper.** See [docs/EVALUATION.md](docs/EVALUATION.md).
- **Training was not run or revalidated.** `main.py`, `train.sh` and `config.yml`'s training sections are upstream and need the full dataset and a multi-GPU setup. They are not needed for anything above.
- Re-running the 97-case evaluation needs the full MSD Task01 dataset (not included; the saved results are).
- The pinned PyTorch 2.6.0 CPU setup was verified on Windows. Saved evidence is from the older GPU environment. Linux/macOS and the 2.6.0 CUDA build are untested. Check the GitHub Actions tab for the hosted CI result.
- **Optional 2D model** (separate archive `Slim-UNETR-optional-2d-addon.zip`, ~630 MB; the model is ~352 MB and the rest is test images): extract it over the same `Slim-UNETR` folder so `model_store\brisc_2d` appears, then `.venv\Scripts\python.exe -m pip install -r requirements-2d.txt`. It is a different model with research-use terms; its scores are not Slim UNETR results.
- Research demonstration only; not a clinical tool.
