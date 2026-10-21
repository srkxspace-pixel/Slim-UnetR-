# Testing and reproducibility

Use Python 3.11 and the pinned requirements from the README. Run commands from the repository root.

## Fast regression checks

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
& .\.venv\Scripts\python.exe -m pip check
```

Tests exercise localhost cookie/token/host/origin checks, uploads, interrupted runs, downloads, the bundled MRI contract, JPG/PNG decoding, EXIF orientation, grayscale/bit depth, reference alignment, padding, opacity and empty-mask metric policy. Tests do not run a neural network or manufacture successful segmentation results. Without the optional BRISC archive, structural image tests extract acquired T1-contrast pixels and the true ET annotation from the bundled case; these fixtures never establish 2D model quality.

## Fresh model inference

```powershell
& .\.venv\Scripts\python.exe -u .\setup_demo.py
& .\.venv\Scripts\python.exe -u .\demo.py --device cpu
```

Inspect the newly created directory, PNGs, masks and report. Checkpoint setup verifies SHA256. Inference uses strict weight loading and verifies model/checkpoint state around prediction.

## Running-server integration

Terminal 1:

```powershell
& .\.venv\Scripts\python.exe -u .\local_ui.py --device cpu
```

Terminal 2, in the same project root:

```powershell
& .\.venv\Scripts\python.exe -u .\verify_local_ui.py
```

For a different port pass the same `--port` to both scripts. This uploads the bundled MRI/reference, then its four unchanged acquired channels separately, checks identical masks, optional metrics, planes/panels, NIfTI geometry, ZIP content, concurrency and request protection. It consumes two stored UI runs and writes local evidence under `outputs/ui_verification/`. Do not repeatedly rerun without a reason or commit uploaded scans.

## Optional 2D quality/integration check

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-2d.txt
& .\.venv\Scripts\python.exe -u .\setup_2d_model.py --test-data
& .\.venv\Scripts\python.exe -u .\verify_2d.py
& .\.venv\Scripts\python.exe -u .\verify_2d.py --cpu-smoke
```

This downloads about 352 MB of model artifacts and a 261 MB pinned public BRISC archive. The verifier extracts actual test cases and retains failures. Its first run creates `outputs/verification_2d/samples/`, including `brisc2025_test_00086_gl_co_t1.jpg` and its matching `*_reference.png` for practice. Source references are retained, with the author's nonzero-foreground rule documented. No MRI images are generated. The verifier refuses to overwrite its completed main evaluation; preserve earlier evidence before repeating.

The historical `verify_2d.py --server-smoke PORT` compares against the original local 97-case archive and is not required for a fresh clone. Use the portable `verify_local_ui.py` for 3D server acceptance.

## Evidence limits

Original Windows development used PyTorch 2.5.1+cu121/GTX 1650. Publication preparation separately checks a fresh Windows Python 3.11 / PyTorch 2.6.0 CPU environment. Refer to recorded verification for actual outcomes; commands/unit tests alone do not establish clinical quality or paper reproduction.

Automated browser visual/file-picker verification was unavailable in the original development session. Model/HTTP checks do not replace rehearsing the form and downloads in Chrome/Edge on the actual presentation laptop. Linux/macOS and new GPU models have not been physically tested here.
