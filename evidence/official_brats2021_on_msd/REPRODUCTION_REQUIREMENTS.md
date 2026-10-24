# Slim UNETR checkpoint reproduction requirements

Audit date: 2026-10-18. Scope: identify what is needed to reproduce reported evaluation results using released weights, without training or synthetic inputs. No additional inference, training, or dataset downloads were performed for this audit.

## Finding

The existing files support a verified inference demonstration. They do not establish the paper's evaluation population or an independent held-out benchmark for the downloaded BraTS 2021 checkpoint. Exact reproduction is blocked by missing data and unresolved checkpoint-to-split provenance, rather than a need to start training.

The full methods section was not retrievable through the publisher or author-linked pages during this audit. Counts below are attributed to the repository or local manifest, and numerical targets are transcribed from the paper table images shipped in the repository. No unknown split is inferred from a case count.

## Published targets and available assets

| Experiment | Target from supplied paper table | Official release asset | Local dataset | Missing for reproduction |
| --- | --- | --- | --- | --- |
| BraTS 2021, Table I | Mean Dice 92.44%; WT 93.77%, ET 89.36%, TC 94.19%; mean HD95 5.36 mm | `brats21_pytorch_model.bin`, already downloaded and strictly loaded | Not present in the configured datasets directory | Exact evaluated BraTS 2021 cases, labels or authorized scoring route, exact training/evaluation membership, and confirmation that this released checkpoint produced this row |
| MSD Task01 Brain Tumour, Table II | Mean Dice 80.27%; WT 89.38%, ET 65.70%, TC 85.73% | No Task01-trained asset listed in the inspected release | 484 labelled cases and 266 test images present | Corresponding Task01 checkpoint and exact evaluated case list; if evaluation used the 266 challenge test images, their labels or an authorized scoring service |
| MSD Task08 Hepatic Vessel, Table II | Mean Dice 69.73%; vessel 65.44%, tumour 74.02% | `hepatic_vessel_pytorch_model.bin`, available but not downloaded | Not present | Task08 images/labels, released hepatic checkpoint, and exact evaluated case list/checkpoint correspondence |

The official repository describes BraTS 2021 as 1,251 training and 219 validation cases. This dataset description does not resolve whether the paper's reported 92.44% came from that challenge validation set or an internal labelled holdout. The local loader's separate 80/20 rule must not be treated as proof of either interpretation.

## Exact split problem

`src/loader.py` obtains BraTS case order with `os.listdir(root)` without sorting or a saved case manifest. `get_dataloader` assigns the first `int(N * train_ratio)` cases to training and the rest to validation, with `train_ratio=0.8` in configuration. A random seed does not fix directory enumeration order. Downloading the same number of files and choosing a fresh last 20% does not reconstruct the checkpoint's historical held-out patients.

The inspected official release tree at commit `6ae2f4e7498102fc17c823f28ca876a47fdbf1e7` contains source and figures but no patient split manifest or evaluation prediction archive. The inspected release assets contain the BraTS 2021 and hepatic-vessel weight files plus source archives. No MSD Task01 checkpoint is listed there.

The current repository issue asking for the split has no comments in the public API response inspected during this audit:

- https://github.com/deepang-ai/Slim-UNETR/issues/1

An older repository issue specifically asks whether the 92.44% score was evaluated on the 219 challenge validation cases and requests a Synapse result. The comments inspected contained follow-up questions, not an author answer:

- https://github.com/aigzhusmart/Slim-UNETR/issues/2

These issues are evidence of an unresolved question, not evidence that a particular split or evaluation claim is false.

## Ground truth access

The official BraTS 2021 challenge page states that its training data include annotations, its validation annotations were withheld, and final testing was performed on hidden data. It identifies Synapse as the segmentation evaluation platform. Thus, if the target result used challenge validation or hidden testing, image downloads alone are insufficient for local Dice/HD95 calculation. We need the relevant ground truth lawfully available for research or confirmation of a currently usable official scoring route; current scoring availability was not established.

Source: https://www.med.upenn.edu/cbica/brats2021/

If the result instead used an internal holdout from the labelled training set, we need that exact list and confirmation it was excluded from checkpoint training. An arbitrary replacement split cannot support the same claim.

## What the existing files establish

- Dataset root: `D:/SRM Projects/datasets/Task01_BrainTumour`.
- Manifest reference: BraTS 2017; release field `2.0 04/05/2018`. These are not BraTS 2021 identities.
- All 484 referenced labelled images and all 484 corresponding labels exist; all 266 referenced test images exist.
- There are 484 unique labelled image names and 266 unique test names, with no name overlap between those two lists. Filename separation does not prove absence of duplicate patients or overlap with BraTS 2021 training.
- No `labelsTs` directory is present. The test manifest lists images only, not ground-truth paths.
- Local manifest order with the existing 0.8 ratio yields 387/97 cases. This is a local partition, not a verified checkpoint training/test split.
- Three of those 97 cases were actually processed and scored: BRATS_409, BRATS_258, BRATS_346. The other 94 have not been evaluated by our runner.
- The four local model source files matched the official release source after normalizing line endings and trailing whitespace in the preceding run.
- The downloaded official checkpoint SHA256 remains `8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4`; the previous run loaded all weights strictly and verified unchanged model state/checkpoint after inference.
- Saved predictions from the three cases were reopened, their grid alignment checked against canonical ground truth, and their Dice independently confirmed with MONAI. See `run_report.json`.
- Label meanings and modality order are explicit in the local manifest: FLAIR, T1w, t1gd, T2w; labels 1 edema, 2 non-enhancing tumour, 3 enhancing tumour. BraTS 2021 uses different numeric labels, so its conversion must use the BraTS 2021 mapping.

We can evaluate the remaining local 97-case partition, or all labelled MSD cases, with frozen BraTS weights and real inputs. Such an evaluation would be a checkpoint application experiment with unverified patient overlap, not reproduction of the paper's MSD score. We can predict the 266 test images but cannot compute their reference-based metrics with the supplied local files.

## Evaluation fidelity requirements

After obtaining the exact dataset and split, freeze the published checkpoint and match modality order, label conversion, orientation/spacing/intensity normalization, ROI, overlap, activation/threshold, any test-time augmentation or postprocessing, and case/class aggregation. Confirm empty-region handling for Dice and HD95 and physical spacing for distances. Export per-case metrics before calculating the table mean.

The code here specifies brain ROI 128 cubed, overlap 0.5, RAS orientation, 1 mm spacing, nonzero channel-wise normalization, sigmoid, threshold 0.5, and Dice/HD95 metrics. These are source-code facts, not a verified statement of every paper-time setting.

The original `inference.py` performs backward/optimizer updates during `warm_up` and then saves state. Reproducing that script unchanged would violate the no-training requirement. We must use frozen evaluation and ask whether the released asset is the final state used for the published numbers, including whether any evaluation-time weight adaptation was involved.

Table III reports 12.42 GFLOPs, 1.79 M parameters, and 279.78 images/min for brain inputs. The actual local parameter count was 1,783,425; it is not numerically identical to 1.79 M under two-decimal rounding. Counting conventions and paper-time implementation need confirmation before claiming an exact match. Local timing on GTX 1650 and full volumes cannot reproduce the paper throughput number without its hardware/input/timing protocol. Real-volume latency and memory can still be measured locally without synthetic inputs. Relative speed/size comparisons with baselines require the corresponding baseline implementations and identical measurement conditions.

## Minimum information to obtain

1. The full paper methods/supplement and confirmation of which table/row the released BraTS checkpoint reproduces.
2. Exact checkpoint training and evaluated case IDs, dataset version, and preferably a split manifest or checksums.
3. Scoring labels for that evaluated population, or an available official scoring route if labels remain hidden.
4. The evaluation command/settings, aggregation rules, any postprocessing, and confirmation that no weight update is required for the published score.
5. For the MSD Task01 result specifically, the Task01-trained checkpoint; for the hepatic result, the matching Task08 dataset and split.

There is no basis to promise exact paper accuracy solely by downloading more data or running our current local 97-case partition.

## Primary references

- Publisher abstract: https://doi.org/10.1109/TMI.2023.3326188
- Authors' repository: https://github.com/deepang-ai/Slim-UNETR
- Official checkpoint release: https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0
- Official BraTS challenge data and scoring description: https://www.med.upenn.edu/cbica/brats2021/
- Supplied result images: `figures/table1.jpg`, `figures/table2.jpg`, `figures/table3.jpg` in this project.
