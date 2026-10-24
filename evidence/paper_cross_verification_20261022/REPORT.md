# Slim UNETR paper cross-verification — 22 October 2026

## Verdict

The local 3D model's core implementation and released BraTS checkpoint match the authors' public release. The current saved outputs do **not** establish reproduction of the original paper's benchmark. The JPG/PNG feature uses a different 2D model and cannot be claimed as a Slim UNETR paper result.

## Sources and scope

Original paper: Yan Pang et al., *Slim UNETR: Scale Hybrid Transformers to Efficient 3D Medical Image Segmentation Under Limited Computational Resources*, DOI [10.1109/TMI.2023.3326188](https://doi.org/10.1109/TMI.2023.3326188), IEEE TMI 43(3), 994–1005. The [paper abstract](https://pubmed.ncbi.nlm.nih.gov/37862274/) states 92.44% Dice on BraTS2021 and links the old authors' repository. The [author publication page](https://pangyan.me/publications.html) identifies the same paper; the current [release](https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0) supplies the weights used here.

This audit downloaded the release source and original table images at commit `6ae2f4e7498102fc17c823f28ca876a47fdbf1e7`, a fresh release checkpoint, its metadata and repository tree. It independently recalculated local metrics from the CSV and cross-checked every value against the saved JSON. It did not rerun the 97-case inference, recompute every voxel-level score, train or modify any predictions. Earlier voxel-level verification remains separately recorded in `outputs/local_validation_97/verification.json`.

The full paper methods/supplement could not be retrieved through the inspected publisher/author links. Per-region targets below were read directly from the authors' released table images, which matched the local copies byte-for-byte. This limitation prevents a claim that every paper-time protocol detail was checked.

## 1. Implementation and weights

- All four core model files matched the pinned upstream source after normalizing whitespace: `SlimUNETR.py`, `Encoder.py`, `Decoder.py`, `Slim_UNETR_Block.py`.
- The existing checkpoint and fresh official download have identical SHA256: `8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4`.
- Strict CPU loading passed with no missing or unexpected model weights. The checkpoint file remained unchanged. The release API supplies no signed/publisher checksum; the hash comparison establishes equality to the freshly downloaded asset.
- The instantiated model has **1,783,425 parameters**. Table III reports 1.79 million: close in scale but not exactly the same number under ordinary two-decimal-million rounding. This is not evidence that the core release source is wrong.

## 2. Numerical comparison

The headline **92.44% mean Dice** is the paper's **BraTS2021** result. We have not evaluated its verified BraTS2021 benchmark population. Our 97 cases are **MSD Task01**, so comparing 80.58% directly with 92.44% does not measure reproduction on the same dataset.

For context, the authors' **Table II MSD Task01** result and our local MSD scores are:

| Dice (%) | Paper Table II | Local 97-case evaluation | Local minus paper (percentage points) |
| --- | ---: | ---: | ---: |
| Whole tumour (WT) | 89.38 | 85.37 | −4.01 |
| Enhancing tumour (ET) | 65.70 | 75.06 | +9.36 |
| Tumour core (TC) | 85.73 | 81.30 | −4.43 |
| Mean of regional means | 80.27 | 80.58 | +0.31 |

These are descriptive comparisons between different experimental setups, not a controlled performance comparison. The close overall mean conceals regional differences; it does not prove matching results or an improvement over the paper. Local ET Dice uses 95 defined cases because two have empty ET references; TC and WT use 97. The paper's corresponding empty-region/aggregation policy is unconfirmed.

Local HD95 means: TC **23.45 mm**, WT **34.86 mm**, ET **infinite** under our explicit one-empty-mask policy. The two ET empty-reference failures are BRATS_026 and BRATS_024. Finite-only overall HD95 **26.16 mm** excludes those two region measurements and must not replace the primary result. Paper Table I's BraTS average HD95 **5.36 mm** belongs to a different population; the supplied Table II contains Dice, not an MSD HD95 target.

## 3. Why exact reproduction is not established

1. **Checkpoint/population mismatch.** Our run applies released **BraTS2021-trained weights** to a local MSD holdout. The inspected release supplies BraTS and hepatic-vessel checkpoints, with no Task01 checkpoint listed. It does not establish the checkpoint that produced Table II.
2. **Unknown split and independence.** The local holdout is the last 97 entries of the 484-case dataset manifest. The pinned release contains no case-split manifest or prediction archive. The BraTS loader uses unsorted `os.listdir` and an 80/20 cut, which cannot reconstruct historical case membership by itself. Patient overlap with checkpoint training is unverified.
3. **Material label-mapping difference.** The released MSD ET converter uses **label 2**; our converter uses **label 3**. The local dataset manifest explicitly defines label 2 as *non-enhancing tumour* and label 3 as *enhancing tumour*. Our mapping agrees with that manifest. We should retain the correct mapping rather than change label meanings to chase a published score. This code discrepancy does not establish which mapping produced the paper's reported numbers or explain the whole score difference.
4. **Different evaluation path.** The released `inference.py` invokes `warm_up`, backpropagation and optimizer updates. Our runner keeps weights frozen. Whether this released warm-up procedure was used to obtain the paper's table is unknown. Additionally, the installed MONAI `AsDiscrete` uses `>= 0.5`, while our runner uses `> 0.5`; any effect from exact ties was not measured. Local `inference.py` also has minor Windows logging/import edits; it is not the app's prediction runner.
5. **Unverified efficiency protocol.** Table III reports 12.42 GFLOPs and 279.78 images/minute for brain inputs. Neither those measurements nor relative baseline speedups have been reproduced under the paper's hardware, input and timing conditions. Whole-volume laptop latency is not interchangeable with that throughput figure.

If the target used official BraTS challenge validation, [BraTS documentation](https://www.med.upenn.edu/cbica/brats2021/) explains that validation annotations were hidden and scoring used Synapse. We need the authors' exact evaluation set and its legitimate labels/scoring route, rather than assuming any 219 scans or any 20% split is equivalent.

## 4. JPG/PNG results

JPG/PNG mode uses **BRISC SegFormer-B5**, not Slim UNETR. Its single binary tumour mask and slice Dice/IoU are different outputs from the 3D model's TC/WT/ET masks. The 18-slice mean Dice **75.25%** and the practice sample Dice **86.55%** cannot be used as evidence for the original Slim UNETR paper. See the separate 2D provenance and verification reports.

## What is needed next

To claim paper reproduction, obtain the exact table-to-checkpoint correspondence, historical training/evaluation case IDs, matching labels or official scoring access, and the complete evaluation/aggregation protocol, including any weight adaptation. Use the corresponding Task01 checkpoint for its table. No current evidence justifies promising those scores by rerunning the present app or starting an arbitrary new training run.

Accurate presentation wording:

> We implemented local inference using the authors' released Slim UNETR architecture and BraTS2021 checkpoint, evaluated it on 97 local MSD cases, and added a separate pretrained 2D model for JPG/PNG demonstrations. These are local evaluation results; the original paper's benchmark has not been reproduced.

## Evidence files

- `verification.json`: recalculated metrics, hashes, source comparisons and strict-load result.
- `audit.py`: reproducible local audit against the downloaded pinned upstream files.
- `upstream/release_metadata.json` and `tree_metadata.json`: release assets and file inventory.
- `src__loader.py.diff`, `inference.py.diff`, `config.yml.diff`: exact source/config differences.
- `upstream/figures__table1.jpg`, `figures__table2.jpg`, `figures__table3.jpg`: source tables retained for inspection.
