# Local MRI demo

The maintained setup and operating instructions are in the [README](README.md).

- [Live presentation walkthrough](docs/DEMO.md)
- [Bundled 3D MRI and reference](examples/BRATS_409/README.md)
- [Testing and reproducibility](docs/TESTING.md)
- [Evaluation scope](docs/EVALUATION.md)
- [Optional 2D provenance](docs/models/BRISC_PROVENANCE.md)

Start an already configured checkout:

```powershell
& .\.venv\Scripts\python.exe -u .\local_ui.py --open-browser
```

Open http://127.0.0.1:8765 and keep the terminal running. For a fresh laptop follow the README from cloning through model setup first.
