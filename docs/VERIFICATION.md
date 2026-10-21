# Clean-clone verification — 22 October 2026

The publication candidate was committed and cloned into an isolated directory. A newly created Python environment was used, without the original environment's installed packages. The clone contained the versioned example but no downloaded weights, full dataset, or historical result folders.

## Tested environment

- Windows AMD64, Python 3.11.9.
- Fresh CPU installation: PyTorch **2.6.0+cpu**, NumPy **1.26.4**, MONAI **1.2.0**, SciPy **1.14.1**, nibabel **5.4.2**, Pillow **12.3.0**.
- The README's CPU installation and pinned application requirements installed successfully. `pip check` passed.
- Source baseline: commit `011954a` (subsequent publication changes add verification documentation only).

## Passed checks

1. **Fresh setup:** the clone downloaded the official checkpoint itself; checksum verification passed. Bundled MRI/reference hashes passed.
2. **Unit/regression tests:** all **23 tests** passed in the clean clone, including the acquired-image fallback when the optional BRISC dataset is absent.
3. **Actual terminal inference:** `demo.py --device cpu` processed the real bundled case, produced PNGs, NIfTI masks, report and ZIP, and verified unchanged model/checkpoint state.
4. **Cross-device regression:** fresh CPU prediction masks were exactly equal to the earlier saved GPU masks for this case (zero differing voxels).
5. **Running-server integration:** the clean clone's CPU server accepted a real 4D upload with reference and a real four-file upload without reference; predictions were identical. All three viewing planes/panels, optional metrics, ZIP/NIfTI geometry, concurrent-request rejection and host/origin/token/download protections passed.
6. **Optional 2D compatibility:** after adding only `requirements-2d.txt`, the same fresh CPU environment loaded verified cached BRISC weights and segmented a real BRISC JPG/reference pair. Dice **86.55%**, IoU **76.29%**, with unchanged model/checkpoint state. This verifies compatibility for that case, not general quality. The optional downloader's existing-file hash path was exercised; the 352 MB model artifacts had been downloaded/verified previously.
7. **Publication hygiene:** staged files excluded virtual environments, caches, downloaded model stores, full datasets, uploads and generated results. Documentation links and JavaScript syntax were checked. A credential-pattern scan of the publication files found no matching keys; this is a targeted check, not a comprehensive security audit.

## Bundled 3D case results

| Region | Dice | HD95 |
| --- | ---: | ---: |
| Tumour core | 96.33% | 1.41 mm |
| Whole tumour | 96.51% | 1.73 mm |
| Enhancing tumour | 94.97% | 1.41 mm |

One fresh terminal run took **6.46 seconds for inference** and **17.48 seconds for processing**, excluding model initialization/imports. Timings vary with hardware and load; they are not the paper's throughput benchmark. The masks and metrics describe this example, not overall model accuracy.

The local uncommitted machine-level evidence is intentionally retained under ignored `outputs/clone_smoke/outputs/`, including `fresh_cpu_demo/report.json`, `ui_verification/verification.json`, and `optional_2d_smoke/report.json`. A fresh clone can regenerate its own evidence using the documented commands.

## Remaining verification limits

- A second physical laptop was not available. The clean-clone/fresh-environment test checks portability on this Windows laptop; hardware-specific behavior can still differ.
- Linux/macOS and the new PyTorch 2.6.0 CUDA build were not locally tested.
- Automated visual/file-picker browser verification was blocked by the available browser surface. Rehearse manual upload in Chrome/Edge on the presentation laptop. HTTP/model tests do not replace that walkthrough.
- Training and exact paper-benchmark reproduction are outside this demo verification. See [evaluation scope](EVALUATION.md).

The Windows CPU GitHub Actions workflow runs the pinned install, unit tests, model setup and fresh real-case inference on future pushes. Its hosted result must be checked separately; local success alone does not establish a hosted CI pass.
