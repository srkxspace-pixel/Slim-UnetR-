# Evaluation scope and paper comparison

The 3D demo uses the original Slim UNETR core implementation and authors' released BraTS2021 checkpoint. A fresh official download was compared with the local file; both had SHA256 `8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4`. All four core model source files matched release commit `6ae2f4e7498102fc17c823f28ca876a47fdbf1e7` after whitespace normalization; strict weight loading passed. The instantiated model has 1,783,425 parameters.

## Demonstration versus reproduction

The [paper abstract](https://pubmed.ncbi.nlm.nih.gov/37862274/) reports 92.44% mean Dice on BraTS2021. Our saved local 97-case evaluation applied the BraTS checkpoint to **MSD Task01**, with unknown patient overlap. It does not reproduce that benchmark.

| Dice (%) | Authors' MSD Table II | Historical local MSD evaluation |
| --- | ---: | ---: |
| Whole tumour | 89.38 | 85.37 |
| Enhancing tumour | 65.70 | 75.06 |
| Tumour core | 85.73 | 81.30 |
| Mean of regional means | 80.27 | 80.58 |

Paper values are from the [authors' table](https://github.com/deepang-ai/Slim-UNETR/blob/6ae2f4e7498102fc17c823f28ca876a47fdbf1e7/figures/table2.jpg); local aggregates were independently recalculated from 97 saved records. A similar mean does not establish matching performance: the experiment, split, checkpoint provenance and scoring conventions must also match. ET Dice had 95 defined local cases because two references had no ET foreground. Primary ET HD95 was infinite under the explicit one-empty-mask policy; finite-only values must not hide those failures.

The released MSD loader selects label 2 for ET; this application uses **label 3**, matching the local MSD dataset's explicit enhancing-tumour definition. The original `inference.py` performs optimizer updates during warm-up; the live app and `demo.py` perform frozen inference without those updates. The original script's paper-time role is unknown. The full paper methods/supplement and historical checkpoint-to-case membership remain unverified. Obtain those details before claiming an exact reproduction. The published throughput and baseline speedups have not been reproduced on the laptop.

## Bundled example

Earlier verified runs of the real BRATS_409 case produced approximately TC Dice **96.33%**, WT **96.51%**, ET **94.97%**. These are case-specific reference comparisons, not overall model accuracy. Current commands always generate a new prediction. CPU/build variation may change low-level numerical results. The downloaded ZIP records current hashes, settings and actual metrics.

## Optional JPG/PNG path

BRISC SegFormer-B5 predicts a single 2D binary tumour mask. A deterministic smoke selection of 18 annotated BRISC slices gave mean Dice 75.25%, including one zero-overlap failure. Two of six images labelled no tumour had nonempty predictions. The sample size is small and patient independence is unverified. Its scores are not comparable to the original Slim UNETR paper.

## Reproducible checks

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
& .\.venv\Scripts\python.exe -u .\demo.py --device cpu
```

To exercise the running server, open another terminal and run `python verify_local_ui.py`. It uploads the bundled case with its true reference, then uploads the four unchanged acquired channels separately without reference, checks identical predictions, all planes, metrics, downloads and request boundaries. It writes local evidence under `outputs/`; those files are intentionally ignored by Git. Extra 2D evaluation requires its separately downloaded public data and weights.
