"""Run an official BraTS 2021 checkpoint on existing MSD Task01 images only."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import monai
import nibabel as nib
import numpy as np
import scipy
import torch
import yaml

from src.loader import ConvertToMultiChannelBasedOnBratsClassesd
from src.SlimUNETR.SlimUNETR import SlimUNETR


RELEASE = "https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0"
CMAP = ListedColormap(
    [(0, 0, 0, 0), (0.1, 0.8, 0.2, 0.55), (1, 0.85, 0, 0.65), (1, 0.1, 0.1, 0.75)]
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def label_map(regions):
    result = np.zeros(regions.shape[1:], dtype=np.uint8)
    result[regions[1]] = 1
    result[regions[0]] = 2
    result[regions[2]] = 3
    return result


def hd95_regions(pred, gt, spacing):
    """Symmetric HD95 in mm; retain undefined and infinite empty-mask cases."""
    distances, statuses = [], []
    for prediction, target in zip(pred, gt):
        has_pred, has_gt = bool(prediction.any()), bool(target.any())
        if not has_pred and not has_gt:
            distances.append(None)
            statuses.append("both_empty_undefined")
        elif not has_pred or not has_gt:
            distances.append("Infinity")
            statuses.append("one_empty_infinite")
        else:
            value = monai.metrics.compute_hausdorff_distance(
                torch.from_numpy(prediction[None, None]),
                torch.from_numpy(target[None, None]),
                include_background=True, percentile=95, directed=False,
                distance_metric="euclidean", spacing=spacing,
            ).item()
            if not np.isfinite(value):
                raise ValueError("Non-finite HD95 for two nonempty masks")
            distances.append(float(value))
            statuses.append("finite")
    return distances, statuses


def aggregate_cases(cases, regions):
    summary = {"case_count": len(cases), "dice": {}, "hd95_mm": {}}
    for region in regions:
        dice = [case["dice"][region] for case in cases if case["dice"][region] is not None]
        distances = [case["hd95_mm"][region] for case in cases]
        finite = [value for value in distances if isinstance(value, (float, int))]
        infinite = distances.count("Infinity")
        summary["dice"][region] = {
            "mean": float(np.mean(dice)) if dice else None,
            "valid_cases": len(dice), "empty_ground_truth_excluded": len(cases) - len(dice),
        }
        summary["hd95_mm"][region] = {
            "mean": "Infinity" if infinite else float(np.mean(finite)) if finite else None,
            "finite_only_mean": float(np.mean(finite)) if finite else None,
            "finite_cases": len(finite), "infinite_cases": infinite,
            "both_empty_undefined_excluded": distances.count(None),
        }
    summary["overall_dice"] = float(np.mean([summary["dice"][r]["mean"] for r in regions]))
    regional_hd = [summary["hd95_mm"][r]["mean"] for r in regions]
    summary["overall_hd95_mm"] = (
        "Infinity" if "Infinity" in regional_hd
        else float(np.mean([v for v in regional_hd if v is not None]))
    )
    finite_means = [summary["hd95_mm"][r]["finite_only_mean"] for r in regions]
    summary["overall_hd95_finite_only_mm"] = float(np.mean([v for v in finite_means if v is not None]))
    summary["overall_definition"] = "Equal-weight arithmetic mean of TC, WT and ET case-mean metrics; not pooled voxel Dice."
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model_store/official_brats2021/brats21_pytorch_model.bin"))
    parser.add_argument('--data-root', type=Path, help='MSD Task01 directory containing dataset.json; overrides config.yml')
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--save-figures", type=int, default=3, help="Number of first cases to plot; all predicted volumes are saved")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/official_brats2021_on_msd"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_cases < 1:
        parser.error("--num-cases must be positive")
    if args.save_figures < 0:
        parser.error("--save-figures cannot be negative")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    torch.set_num_threads(4)
    with open("config.yml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["brain2019"]
    root = Path(args.data_root or os.environ.get('SLIM_UNETR_DATA_ROOT') or config['data_root']).resolve()
    with open(root / "dataset.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_modalities = {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}
    expected_labels = {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    if manifest["modality"] != expected_modalities or manifest["labels"] != expected_labels:
        raise ValueError("Dataset modalities or label meanings differ from the supported MSD Task01 manifest")
    records = manifest["training"]
    split = int(len(records) * config["trainer"]["train_ratio"])
    selected = records[split:split + args.num_cases]
    if len(selected) != args.num_cases:
        raise ValueError("Not enough cases in the local validation partition")
    cases = []
    for record in selected:
        resolved = {key: (root / record[key]).resolve() for key in ("image", "label")}
        for path in resolved.values():
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Invalid dataset path: {path}")
        cases.append({key: str(path) for key, path in resolved.items()})

    checkpoint_hash = file_hash(args.checkpoint)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SlimUNETR(**config["slim_unetr"])
    model.load_state_dict(weights, strict=True)
    del weights
    model.requires_grad_(False)
    model.to(args.device).eval()
    before = state_hash(model)
    print(f"Official checkpoint loaded strictly: {args.checkpoint}", flush=True)
    print(f"SHA256: {checkpoint_hash}", flush=True)
    print(f"Device: {args.device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("BraTS 2021 checkpoint applied to MSD Task01; this is not the paper's BraTS benchmark.", flush=True)

    # The existing validation preprocessing, with no training loader or random transforms.
    transform = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", is2019=True),
        monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    dataset = monai.data.Dataset(cases, transform=transform)
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=(config["trainer"]["image_size"],) * 3,
        sw_batch_size=1, overlap=0.5, sw_device=args.device, device="cpu",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "case_results.jsonl"
    if progress_path.exists():
        raise FileExistsError(f"Use a fresh output directory to preserve existing results: {progress_path}")
    report = {
        "evaluation_label": "OUR LOCAL MSD TASK01 VALIDATION EVALUATION — NOT REPRODUCED PAPER RESULTS",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "official_release": RELEASE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_training_dataset": "BraTS 2021 (per authors' release)",
        "evaluation_dataset": str(root),
        "limitation": "Cross-dataset demonstration; training/test patient overlap with BraTS 2021 is not verified. Not an independent benchmark or paper reproduction.",
        "split": {"method": "existing manifest order and configured train_ratio", "validation_start_index": split,
                  "total_labelled_cases": len(records), "requested_cases": len(cases),
                  "dataset_manifest_sha256": file_hash(root / "dataset.json"),
                  "case_ids": [Path(case["image"]).name.removesuffix(".nii.gz") for case in cases]},
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "monai": monai.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "inference": {"roi": [config["trainer"]["image_size"]] * 3, "overlap": 0.5, "threshold": 0.5, "training": False, "synthetic_inputs": False},
        "metric_policy": {
            "dice": "Per-case per-region Dice; empty ground-truth regions are undefined/excluded, matching MONAI ignore_empty=True. Nonempty GT and empty prediction scores zero.",
            "hd95": "MONAI symmetric Euclidean HD95: maximum of the two directed 95th-percentile surface distances, using physical voxel spacing in mm.",
            "empty_hd95": "Both masks empty: undefined/excluded. Exactly one empty: Infinity retained in primary mean. Finite-only means are separate and exclude these failures.",
        },
        "regions": ["TC", "WT", "ET"],
        "nifti_space": "Preprocessed RAS orientation and 1 mm spacing; region volume last axis is TC, WT, ET",
        "cases": [],
    }
    with torch.inference_mode(), progress_path.open("x", encoding="utf-8") as progress:
        for index in range(len(dataset)):
            case_id = Path(cases[index]["image"]).name.removesuffix(".nii.gz")
            print(f"Loading {case_id} ({index + 1}/{len(dataset)})", flush=True)
            item = dataset[index]
            image, label = item["image"], item["label"]
            if image.shape[0] != 4 or not torch.isfinite(image).all():
                raise ValueError(f"Invalid image tensor for {case_id}: {image.shape}")
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = inferer(image.unsqueeze(0), model)
            pred = torch.sigmoid(logits[0]).gt(0.5).cpu().numpy()
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            gt = label.cpu().numpy().astype(bool)
            intersection = (pred & gt).sum(axis=(1, 2, 3), dtype=np.float64)
            denominator = pred.sum(axis=(1, 2, 3)) + gt.sum(axis=(1, 2, 3))
            # Empty ground-truth classes are omitted, as in MONAI DiceMetric's default policy.
            scores = [float(2 * intersection[k] / denominator[k]) if gt[k].any() else None for k in range(3)]
            dice = dict(zip(report["regions"], scores))
            affine = image.affine.cpu().numpy()
            spacing = np.linalg.norm(affine[:3, :3], axis=0).tolist()
            hd95_values, hd95_status = hd95_regions(pred, gt, spacing)
            hd95 = dict(zip(report["regions"], hd95_values))
            z = int(gt[1].sum(axis=(0, 1)).argmax())
            flair = image[0].cpu().numpy()
            png = args.output_dir / f"{case_id}.png"
            if index < args.save_figures:
                fig, axes = plt.subplots(1, 3, figsize=(13, 5))
                for ax, title, overlay in zip(axes, ["Existing MRI (FLAIR)", "Dataset ground truth", "Official Slim UNETR prediction"], [None, label_map(gt), label_map(pred)]):
                    ax.imshow(np.rot90(flair[..., z]), cmap="gray")
                    if overlay is not None:
                        ax.imshow(np.rot90(overlay[..., z]), cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")
                score_text = "  ".join(f"{key} {value:.3f}" if value is not None else f"{key} n/a" for key, value in dice.items())
                fig.suptitle(f"{case_id} | whole-volume Dice: {score_text}\nOUR LOCAL EVALUATION | BraTS 2021 weights on MSD Task01 | slice {z}", fontsize=11)
                fig.text(0.5, 0.025, "Green: edema / WT-only   Yellow: core excluding ET   Red: enhancing tumour", ha="center", fontsize=10)
                fig.tight_layout(rect=(0, 0.055, 1, 0.9))
                fig.savefig(png, dpi=140)
                plt.close(fig)
            regions_path = args.output_dir / f"{case_id}_pred_regions.nii.gz"
            nib.save(nib.Nifti1Image(np.moveaxis(pred.astype(np.uint8), 0, -1), affine), regions_path)
            case_report = {
                "case": case_id, "source_image": cases[index]["image"], "source_label": cases[index]["label"],
                "image_sha256": file_hash(cases[index]["image"]), "label_sha256": file_hash(cases[index]["label"]),
                "image_shape": list(image.shape), "spacing_mm": spacing, "dice": dice, "hd95_mm": hd95,
                "hd95_status": dict(zip(report["regions"], hd95_status)),
                "ground_truth_voxels": dict(zip(report["regions"], [int(mask.sum()) for mask in gt])),
                "prediction_voxels": dict(zip(report["regions"], [int(mask.sum()) for mask in pred])),
                "inference_seconds": elapsed,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                "displayed_axial_slice": z if index < args.save_figures else None,
                "png": str(png.resolve()) if index < args.save_figures else None,
                "prediction_regions_nifti": str(regions_path.resolve()),
            }
            report["cases"].append(case_report)
            progress.write(json.dumps(case_report, allow_nan=False) + "\n")
            progress.flush()
            print(f"Completed {index + 1}/{len(dataset)} {case_id}: Dice={dice}; HD95 mm={hd95}", flush=True)
            del item, image, label, logits, pred, gt
    report["model_state_unchanged"] = state_hash(model) == before
    report["checkpoint_file_unchanged"] = file_hash(args.checkpoint) == checkpoint_hash
    if not report["model_state_unchanged"] or not report["checkpoint_file_unchanged"]:
        raise RuntimeError("Model state or checkpoint changed during inference")
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = aggregate_cases(report["cases"], report["regions"])
    report["model_state_sha256_before"] = before
    report["model_state_sha256_after"] = state_hash(model)
    with (args.output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case"] + [f"{metric}_{r}" for metric in ("dice", "hd95_mm") for r in report["regions"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow({"case": case["case"], **{f"{metric}_{r}": case[metric][r] for metric in ("dice", "hd95_mm") for r in report["regions"]}})
    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, allow_nan=False), flush=True)
    print(f"Done. Model and checkpoint unchanged. Report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
