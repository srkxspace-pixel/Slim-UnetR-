"""Frozen official-checkpoint inference for validated local NIfTI uploads."""

import io
import json
import math
import time
import zipfile
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import yaml

from inference_official import file_hash, hd95_regions, state_hash
from src.SlimUNETR.SlimUNETR import SlimUNETR

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model_store/official_brats2021/brats21_pytorch_model.bin"
EXPECTED_SHA = "8adbc8ce26c67a6783da8aea03edb95f530c16b4416185501a2711472677d5f4"
MODALITIES = ("flair", "t1", "t1ce", "t2")
REGIONS = ("TC", "WT", "ET")
MAX_VOXELS = 20_000_000


def save_volume(path, array, affine):
    volume = nib.Nifti1Image(array, affine)
    volume.header.set_xyzt_units('mm')
    nib.save(volume, path)


def inspect_volume(path, dimensions):
    try:
        volume = nib.load(str(path))
    except Exception as error:
        raise ValueError("Cannot read this file as a NIfTI volume. Use .nii or .nii.gz files.") from error
    if not isinstance(volume, (nib.Nifti1Image, nib.Nifti2Image)) or volume.ndim != dimensions:
        raise ValueError(f"Expected a {dimensions}D NIfTI volume; received shape {volume.shape}.")
    shape = volume.shape[:3]
    affine = volume.affine
    if any(n < 32 for n in shape) or math.prod(shape) > MAX_VOXELS:
        raise ValueError("Use a full brain volume with at least 32 voxels per axis and at most 20 million spatial voxels.")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError("The NIfTI spatial transform is missing or invalid.")
    if volume.header.get_xyzt_units()[0] != "mm":
        raise ValueError("NIfTI spatial units must explicitly be millimetres (mm). Unknown units cannot be assumed.")
    spacing = nib.affines.voxel_sizes(affine)
    target_shape = np.ceil((np.asarray(shape) - 1) * spacing).astype(np.int64) + 1
    if not np.isfinite(spacing).all() or np.any(spacing <= 0) or math.prod(map(int, target_shape)) > MAX_VOXELS:
        raise ValueError("This image would exceed the 20-million-voxel limit after 1 mm resampling.")
    if getattr(volume.dataobj, "offset", 0) > 1024 * 1024 or volume.get_data_dtype().kind not in "uif":
        raise ValueError("Unsupported NIfTI data layout or pixel type.")
    if volume.get_data_dtype().itemsize > 8:
        raise ValueError("Unsupported NIfTI pixel size.")
    return volume


def validate_inputs(files, mode, label_scheme):
    """Header and content checks; never fabricate a missing modality or infer label meanings."""
    if mode not in ("four_files", "four_dimensional"):
        raise ValueError("Choose four 3D modalities or one four-channel 4D NIfTI.")
    required = list(MODALITIES) if mode == "four_files" else ["image"]
    expected = set(required) | ({"reference"} if "reference" in files else set())
    if set(files) != expected:
        raise ValueError("Provide exactly the requested MRI files; all four modalities are required.")
    volumes = {name: inspect_volume(files[name], 3 if mode == "four_files" else 4) for name in required}
    first = volumes[required[0]]
    if mode == "four_dimensional" and first.shape[3] != 4:
        raise ValueError("The 4D NIfTI must have exactly four channels: FLAIR, T1, T1ce, T2, in that order.")
    for name, volume in volumes.items():
        if volume.shape[:3] != first.shape[:3] or not np.allclose(volume.affine, first.affine, rtol=1e-5, atol=1e-3):
            raise ValueError(f"{name}: modality grids do not match. Co-register the modalities before uploading; this app does not register them.")
        # Force decoding now so corrupt, non-finite, and constant inputs fail before model execution.
        array = volume.get_fdata(dtype=np.float32, caching="unchanged")
        channels = [array] if array.ndim == 3 else [array[..., i] for i in range(4)]
        for channel in channels:
            if not np.isfinite(channel).all() or np.ptp(channel) == 0:
                raise ValueError(f"{name}: each modality must contain finite, nonconstant MRI intensities.")
        del channels, array
    if "reference" in files:
        if label_scheme not in ("msd", "brats"):
            raise ValueError("Select the reference label scheme: MSD Task01 or BraTS 2021.")
        reference = inspect_volume(files["reference"], 3)
        if reference.shape != first.shape[:3] or not np.allclose(reference.affine, first.affine, rtol=1e-5, atol=1e-3):
            raise ValueError("Reference mask and MRI grids do not match.")
        values = np.unique(np.asarray(reference.dataobj))
        allowed = [0, 1, 2, 3] if label_scheme == "msd" else [0, 1, 2, 4]
        if not np.isin(values, allowed).all():
            raise ValueError(f"Reference labels do not match {label_scheme}; allowed values are {allowed}.")
    return {"shape": list(first.shape[:3]), "affine": first.affine.tolist(), "spacing_mm": nib.affines.voxel_sizes(first.affine).tolist()}


class Engine:
    def __init__(self, device='auto'):
        torch.set_num_threads(4)
        if not CHECKPOINT.is_file() or file_hash(CHECKPOINT) != EXPECTED_SHA:
            raise ValueError("The verified official checkpoint is missing or its SHA256 has changed. Restore the official checkpoint before running.")
        config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))["brain2019"]["slim_unetr"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == 'auto' else device
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('Choose auto, cpu, or cuda for the inference device.')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Start with --device cpu or --device auto.')
        self.model = SlimUNETR(**config)
        self.model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
        self.model.requires_grad_(False).to(self.device).eval()
        self.state = state_hash(self.model)
        self.info = {"device": self.device, "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                     "checkpoint_sha256": EXPECTED_SHA, "parameters": sum(p.numel() for p in self.model.parameters())}

    @torch.inference_mode()
    def predict(self, directory, files, names, mode, label_scheme, progress):
        started = time.perf_counter()
        progress("Validating MRI files", "Checking volume content, units and matching spatial grids.")
        original = validate_inputs(files, mode, label_scheme)
        source_hashes = {name: file_hash(path) for name, path in files.items()}
        if file_hash(CHECKPOINT) != EXPECTED_SHA or state_hash(self.model) != self.state:
            raise RuntimeError("Checkpoint or model state changed. Restart with the verified official weights.")
        progress("Preprocessing", "RAS orientation, 1 mm spacing, and per-modality intensity normalization.")
        image_input = [str(files[name]) for name in MODALITIES] if mode == "four_files" else str(files["image"])
        image_transform = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.Spacingd(keys="image", pixdim=(1., 1., 1.), mode="bilinear"),
            monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
        image = image_transform({"image": image_input})["image"]
        if image.shape[0] != 4 or math.prod(image.shape[1:]) > MAX_VOXELS or not torch.isfinite(image).all():
            raise ValueError("Preprocessed MRI shape or intensities are invalid.")
        affine = image.affine.cpu().numpy()
        target = None
        if "reference" in files:
            from src.loader import ConvertToMultiChannelBasedOnBratsClassesd
            reference_transform = monai.transforms.Compose([
                monai.transforms.LoadImaged(keys="label"),
                monai.transforms.EnsureTyped(keys="label"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label", is2019=label_scheme == "msd"),
                monai.transforms.Orientationd(keys="label", axcodes="RAS"),
                monai.transforms.Spacingd(keys="label", pixdim=(1., 1., 1.), mode="nearest"),
            ])
            label = reference_transform({"label": str(files["reference"])})["label"]
            if tuple(label.shape[1:]) != tuple(image.shape[1:]) or not np.allclose(label.affine.cpu().numpy(), affine, atol=1e-3):
                raise ValueError("Reference and MRI grids differ after preprocessing.")
            target = label.cpu().numpy().astype(bool)
            del label
        windows = 0
        padded_shape = tuple(max(128, int(n)) for n in image.shape[1:])
        total = len(monai.data.utils.dense_patch_slices(padded_shape, (128,)*3, (64,)*3))

        def window_predictor(batch):
            nonlocal windows
            result = self.model(batch)
            windows += 1
            progress("Predicting", f"Window {windows} of {total}", windows, total)
            return result

        inferer = monai.inferers.SlidingWindowInferer(roi_size=(128,)*3, sw_batch_size=1, overlap=.5,
                                                    sw_device=self.device, device="cpu")
        inference_start = time.perf_counter()
        logits = inferer(image.unsqueeze(0), window_predictor)
        prediction = torch.sigmoid(logits[0]).gt(.5).cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        del logits
        progress("Preparing result", "Saving prediction volumes and interactive slice data.")
        np.save(directory / "prediction.npy", prediction.astype(np.uint8))
        display = np.empty(tuple(image.shape), dtype=np.uint8)
        for i in range(4):
            channel = image[i].cpu().numpy()
            nonzero = channel[channel != 0]
            lo, hi = np.percentile(nonzero if nonzero.size else channel, [1, 99])
            display[i] = (np.clip((channel-lo) / max(float(hi-lo), 1e-6), 0, 1)*255).astype(np.uint8)
        np.save(directory / "display.npy", display)
        downloads = ["report.json", "FLAIR_processed.nii.gz"]
        save_volume(directory / "FLAIR_processed.nii.gz", image[0].cpu().numpy(), affine)
        for i, region in enumerate(REGIONS):
            name = f"prediction_{region}.nii.gz"
            save_volume(directory / name, prediction[i].astype(np.uint8), affine)
            downloads.append(name)
        metrics = None
        if target is not None:
            progress("Calculating metrics", "Whole-volume Dice and HD95; empty regions are reported explicitly.")
            np.save(directory / "reference.npy", target.astype(np.uint8))
            spacing = np.linalg.norm(affine[:3, :3], axis=0).tolist()
            hd95, statuses = hd95_regions(prediction, target, spacing)
            metrics = {}
            for i, region in enumerate(REGIONS):
                p, g = int(prediction[i].sum()), int(target[i].sum())
                dice = float(2 * (prediction[i] & target[i]).sum() / (p+g)) if g else None
                metrics[region] = {"dice": dice, "hd95_mm": hd95[i], "hd95_status": statuses[i],
                                   "reference_voxels": g, "predicted_voxels": p}
                name = f"reference_{region}.nii.gz"
                save_volume(directory / name, target[i].astype(np.uint8), affine)
                downloads.append(name)
        unchanged = state_hash(self.model) == self.state and file_hash(CHECKPOINT) == EXPECTED_SHA
        if not unchanged:
            raise RuntimeError("Model or checkpoint changed during inference; result withheld.")
        initial_mask = target[1] if target is not None and target[1].any() else prediction[1]
        default_slice = int(initial_mask.sum(axis=(0, 1)).argmax()) if initial_mask.any() else int(image.shape[3] // 2)
        report = {
            "label": "Local inference using official BraTS 2021 weights — not reproduced paper results",
            "input_mode": mode, "uploaded_names": names, "source_sha256": source_hashes,
            "original_grid": original, "processed_shape": list(image.shape[1:]), "processed_affine": affine.tolist(),
            "modalities": list(MODALITIES), "regions": list(REGIONS), "reference_provided": target is not None,
            "reference_label_scheme": label_scheme if target is not None else None,
            "checkpoint_release": "https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0",
            **self.info, "model_state_sha256_before": self.state, "model_state_sha256_after": state_hash(self.model),
            "model_state_unchanged": unchanged, "checkpoint_unchanged": unchanged,
            "training": False, "synthetic_inputs": False, "threshold": .5, "threshold_operator": ">",
            "roi": [128]*3, "overlap": .5, "normalization": "Nonzero, per-channel; RAS, 1 mm spacing",
            "output_space": "Preprocessed RAS / 1 mm. Use FLAIR_processed.nii.gz as the matching background.",
            "metrics": metrics, "default_slice": default_slice,
            "metric_policy": "Dice excludes empty reference regions; HD95 is symmetric max of directed 95th percentiles in mm. One empty mask: Infinity. Both empty: undefined.",
            "inference_seconds": inference_seconds, "total_processing_seconds": time.perf_counter()-started,
            "downloads": downloads + ["prediction_bundle.zip"],
            "versions": {"torch": torch.__version__, "monai": monai.__version__, "nibabel": nib.__version__, "numpy": np.__version__},
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        with zipfile.ZipFile(directory / "prediction_bundle.zip", "w", compression=zipfile.ZIP_STORED) as bundle:
            for name in downloads:
                bundle.write(directory / name, name)
        return report


def render_slice(directory, axis, index, modality, panel, region, opacity):
    if axis not in ("axial", "coronal", "sagittal") or modality not in MODALITIES or panel not in ("mri", "prediction", "reference"):
        raise ValueError("Invalid slice view.")
    if region not in ("all", *REGIONS) or not 0 <= opacity <= 1:
        raise ValueError("Invalid overlay selection.")
    image = np.load(directory / "display.npy", mmap_mode="r", allow_pickle=False)
    dimension = {"axial": 2, "coronal": 1, "sagittal": 0}[axis]
    if not 0 <= index < image.shape[dimension+1]:
        raise ValueError("Slice index is out of range.")
    def cut(array):
        return np.rot90(np.take(array, index, axis=dimension))
    gray = cut(image[MODALITIES.index(modality)])
    rgb = np.stack([gray]*3, axis=-1).astype(np.float32)
    if panel != "mri":
        path = directory / f"{panel}.npy"
        if not path.is_file():
            raise ValueError("No reference mask was provided for this result.")
        masks = np.load(path, mmap_mode="r", allow_pickle=False)
        # Overlapping TC/WT/ET are displayed with ET > TC > WT priority; downloads retain all masks.
        selected = [1, 0, 2] if region == "all" else [REGIONS.index(region)]
        palette = {0: (255, 210, 50), 1: (30, 220, 120), 2: (255, 70, 90)}
        overlay = np.zeros_like(rgb)
        visible = np.zeros(gray.shape, bool)
        for i in selected:
            mask = cut(masks[i]).astype(bool)
            overlay[mask] = palette[i]
            visible |= mask
        rgb[visible] = (1-opacity)*rgb[visible] + opacity*overlay[visible]
    result = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(result, format="PNG")
    return result.getvalue()
