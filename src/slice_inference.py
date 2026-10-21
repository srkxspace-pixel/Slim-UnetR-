"""Offline binary segmentation of single T1 MRI slices with pinned BRISC SegFormer weights."""
import io
import json
import math
from pathlib import Path
import time
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from setup_2d_model import MODEL, FILES, HEAD_REVISION, BASE_REVISION, digest

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 4_000_000
MODEL_NAME = 'BRISC SegFormer-B5 · 2D binary tumour segmentation'
LIMITATION = ('Research demo for individual T1-weighted brain MRI slices, preferably contrast-enhanced and similar to BRISC. '
              'Unvalidated on arbitrary screenshots, other sequences or unseen scanners. A mask is a model prediction, not a diagnosis. '
              'An empty mask does not rule out tumour. No TC/WT/ET or physical-volume measurement is available from a JPG/PNG.')


def read_image(path, reference=False):
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError('Each 2D image must be nonempty and at most 16 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.format not in ('JPEG', 'PNG') or getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Use a single-frame JPG or PNG image.')
                if reference and source.format != 'PNG':
                    raise ValueError('Reference masks must be lossless binary PNG files.')
                expected = 'PNG' if path.suffix.lower() == '.png' else 'JPEG'
                if source.format != expected:
                    raise ValueError('The image content does not match its JPG/PNG extension.')
                if min(source.size) < 32 or max(source.size) > 4096 or math.prod(source.size) > MAX_PIXELS:
                    raise ValueError('Image dimensions must be 32–4096 pixels per side and at most 4 million pixels.')
                if source.mode not in ('L', 'RGB', 'RGBA', 'LA', 'P'):
                    raise ValueError('Use an 8-bit grayscale or RGB image. Export 16-bit images deliberately before uploading.')
                oriented = ImageOps.exif_transpose(source)
                rgba = oriented.convert('RGBA')
                if np.asarray(rgba.getchannel('A')).min() != 255:
                    raise ValueError('Transparent images are not supported. Export an opaque image first.')
                rgb = rgba.convert('RGB')
                rgb.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('The JPG/PNG image is corrupt, unsupported, or too large to decode safely.') from error
    array = np.asarray(rgb)
    if reference:
        if not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])):
            raise ValueError('Use a grayscale binary reference mask, not a colour overlay.')
        if not set(np.unique(array[..., 0])).issubset({0, 1, 255}):
            raise ValueError('Reference PNG values must be 0 for background and 1 or 255 for tumour.')
        return array[..., 0] > 0
    if int(array.max()) - int(array.min()) < 2:
        raise ValueError('The image is blank or constant. Supply a brain MRI slice.')
    return rgb


def prepare_image(image):
    """Match upstream centre-padding, PIL bilinear 512 resize and RGB /255 preprocessing."""
    width, height = image.size
    side = max(width, height)
    offset = ((side - width) // 2, (side - height) // 2)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(image, offset)
    resized = canvas.resize((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(array), side, offset


def mask_metrics(prediction, reference):
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'true_positive_pixels': tp, 'false_positive_pixels': fp, 'false_negative_pixels': fn,
            'policy': 'Original oriented image pixel grid; both masks empty gives Dice=IoU=1; one empty gives 0.'}


def state_digest(model):
    import hashlib
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class SliceEngine:
    def __init__(self, device='auto'):
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        torch.set_num_threads(4)
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable. Start with --device cpu.')
        self.verify_files()
        config = SegformerConfig.from_json_file(str(MODEL / 'config.json'))
        config.num_labels = 2
        config.id2label = {0: 'background', 1: 'tumour'}
        config.label2id = {'background': 0, 'tumour': 1}
        model = SegformerForSemanticSegmentation(config)
        # Both public checkpoints contain tensors only. Never import upstream training code or unpickle objects.
        base = torch.load(MODEL / 'base_model.bin', map_location='cpu', weights_only=True)
        head = torch.load(MODEL / 'seg_best.pt', map_location='cpu', weights_only=True)
        prefix = 'segformer.'
        if not head or any(not key.startswith(prefix + 'decode_head.') for key in head):
            raise RuntimeError('The BRISC checkpoint does not contain the expected decoder weights.')
        # Transformers 5 renamed linear_c to linear_projections; tensor layout is unchanged.
        # Canonicalize this known decoder alias, then require complete, exact keys and shapes.
        decoder = {key[len(prefix):].replace('decode_head.linear_projections.', 'decode_head.linear_c.'): value for key, value in head.items()}
        expected = model.state_dict()
        if set(decoder) != {key for key in expected if key.startswith('decode_head.')}:
            raise RuntimeError('BRISC decoder keys are incomplete or unexpected.')
        merged = {key: decoder[key] if key.startswith('decode_head.') else base[key] for key in expected}
        model.load_state_dict(merged, strict=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.initial_state = state_digest(self.model)
        self.info = {'name': MODEL_NAME, 'device': self.device, 'gpu': torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU',
                     'head_revision': HEAD_REVISION, 'base_revision': BASE_REVISION,
                     'transformers_version': __import__('transformers').__version__, 'torch_version': torch.__version__,
                     'decoder_key_alias': 'linear_projections -> linear_c; exact decoder keys and tensor shapes required'}

    @staticmethod
    def verify_files():
        for name, _, expected, kind in FILES[:3]:
            path = MODEL / name
            if not path.is_file() or digest(path, kind) != expected:
                raise RuntimeError('2D model files are missing or changed. Run setup_2d_model.py to install the pinned weights.')

    def predict(self, directory, files, names, progress):
        import torch
        import torch.nn.functional as functional
        started = time.perf_counter()
        progress('Validating 2D image', 'Checking the image and optional binary reference mask.')
        image = read_image(files['image'])
        reference = read_image(files['reference'], reference=True) if 'reference' in files else None
        if reference is not None and reference.shape != (image.height, image.width):
            raise ValueError('Reference dimensions must match the oriented input image exactly. Masks are never stretched to fit.')
        self.verify_files()
        if state_digest(self.model) != self.initial_state:
            raise RuntimeError('2D model state changed; prediction refused.')
        array, side, (left, top) = prepare_image(image)
        inputs = torch.from_numpy(array).unsqueeze(0).to(self.device)
        mean = inputs.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = inputs.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        progress('Predicting 2D tumour mask', 'Running the frozen BRISC SegFormer-B5 model on this slice.')
        inference_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(pixel_values=(inputs - mean) / std).logits
            if logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise RuntimeError('The model returned invalid segmentation logits.')
            square_logits = functional.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
            # Preserve upstream's 512-grid argmax; restore that binary mask with nearest neighbour.
            mask512 = square_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        inference_seconds = time.perf_counter() - inference_start
        restored = Image.fromarray(mask512 * 255).resize((side, side), Image.Resampling.NEAREST)
        prediction = np.asarray(restored.crop((left, top, left + image.width, top + image.height))) > 0
        self.verify_files()
        if state_digest(self.model) != self.initial_state:
            raise RuntimeError('2D model state changed during inference; result withheld.')
        progress('Saving 2D result', 'Writing the original-size mask, overlay and provenance report.')
        image.save(directory / 'image.png')
        Image.fromarray(prediction.astype(np.uint8) * 255).save(directory / 'mask.png')
        np.save(directory / 'mask_512.npy', mask512)
        if reference is not None:
            Image.fromarray(reference.astype(np.uint8) * 255).save(directory / 'reference.png')
        metrics = mask_metrics(prediction, reference) if reference is not None else None
        result = {'input_mode': 'single_slice', 'model_name': MODEL_NAME, 'model': self.info,
                  'uploaded_names': names, 'original_size': list(image.size), 'processed_shape': [512, 512],
                  'reference_provided': reference is not None, 'metrics': metrics,
                  'predicted_pixels': int(prediction.sum()), 'predicted_area_percent': float(prediction.mean() * 100),
                  'inference_seconds': inference_seconds, 'total_processing_seconds': time.perf_counter() - started,
                  'model_state_unchanged': True, 'checkpoint_unchanged': True, 'model_state_sha256': self.initial_state,
                  'checkpoint_sha256': {name: expected for name, _, expected, kind in FILES[:2]},
                  'input_sha256': {role: digest(path) for role, path in files.items()},
                  'preprocessing': 'EXIF transpose; RGB display channels (not MRI modalities); black centre-pad square; PIL bilinear 512x512; /255; ImageNet mean/std.',
                  'mask_geometry': 'Argmax of bilinear-upsampled 512x512 logits; nearest resize to padded square; crop to original oriented image. PNG labels 0 background / 255 predicted tumour.',
                  'limitations': LIMITATION, 'metric_units': 'pixels; physical spacing unknown', 'training_performed': False,
                  'downloads': ['prediction_bundle.zip', 'report.json', 'mask.png', 'overlay.png', 'image.png']}
        Image.open(io.BytesIO(render_slice(directory, 'prediction', .55))).save(directory / 'overlay.png')
        if reference is not None:
            result['downloads'].append('reference.png')
        (directory / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        with zipfile.ZipFile(directory / 'prediction_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in result['downloads']:
                if name != 'prediction_bundle.zip':
                    archive.write(directory / name, name)
        return result


def render_slice(directory, panel='prediction', opacity=.55):
    if panel not in ('mri', 'prediction', 'reference') or not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Invalid 2D panel or opacity.')
    with Image.open(directory / 'image.png') as source:
        image = source.convert('RGB')
    if panel != 'mri':
        path = directory / ('reference.png' if panel == 'reference' else 'mask.png')
        if not path.is_file():
            raise ValueError('No reference mask was supplied.')
        with Image.open(path) as source:
            mask = np.asarray(source) > 0
        pixels = np.asarray(image).copy()
        colour = np.array([255, 70, 90] if panel == 'prediction' else [30, 220, 120])
        pixels[mask] = np.rint(pixels[mask] * (1 - opacity) + colour * opacity).astype(np.uint8)
        image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()
