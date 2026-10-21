"""2D input/error boundaries using acquired BRISC images, never generated MRI fixtures."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
from werkzeug.test import Client
from werkzeug.wrappers import Response

import local_ui
from src.slice_inference import read_image, prepare_image, mask_metrics, render_slice, MAX_IMAGE_BYTES


class SliceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = Path(__file__).resolve().parents[1] / 'outputs/verification_2d/source/brisc.zip'
        if not archive.is_file():
            # Structural tests need real acquired pixels, not an entire additional dataset.
            # Extract one T1-contrast slice and its ET reference from the bundled real 3D case.
            # This is only an image-validation fixture, never a model-quality evaluation.
            import nibabel as nib
            from src.demo_case import demo_case
            files = demo_case()
            volume = nib.load(files['image'])
            labels = np.asarray(nib.load(files['reference']).dataobj)
            index = int((labels == 3).sum(axis=(0, 1)).argmax())
            pixels = np.asarray(volume.dataobj[:, :, index, 2], dtype=np.float32)
            low, high = np.percentile(pixels[pixels != 0], [1, 99])
            gray = (np.clip((pixels - low) / max(float(high - low), 1e-6), 0, 1) * 255).astype(np.uint8)
            image_stream, mask_stream = io.BytesIO(), io.BytesIO()
            Image.fromarray(gray).convert('RGB').save(image_stream, format='JPEG')
            Image.fromarray((labels[:, :, index] == 3).astype(np.uint8) * 255).save(mask_stream, format='PNG')
            cls.jpeg, cls.mask = image_stream.getvalue(), mask_stream.getvalue()
            return
        with zipfile.ZipFile(archive) as source:
            name = sorted(n for n in source.namelist() if '/segmentation_task/test/images/' in n)[0]
            cls.jpeg = source.read(name)
            cls.mask = source.read(name.replace('/images/', '/masks/').replace('.jpg', '.png'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'scan.jpg'
        self.image.write_bytes(self.jpeg)
        self.reference = self.root / 'reference.png'
        with Image.open(io.BytesIO(self.mask)) as source:
            # Match the dataset author's nonzero-foreground contract; preserve raw source in cls.mask.
            Image.fromarray((np.asarray(source.convert('L')) > 0).astype(np.uint8) * 255).save(self.reference)

    def test_real_jpeg_and_lossless_png_preprocess_identically(self):
        png = self.root / 'scan.png'
        image = read_image(self.image)
        image.save(png)
        np.testing.assert_array_equal(prepare_image(image)[0], prepare_image(read_image(png))[0])

    def test_corrupt_and_mislabeled_image_rejected(self):
        wrong = self.root / 'wrong.png'
        wrong.write_bytes(self.jpeg)
        with self.assertRaisesRegex(ValueError, 'extension'):
            read_image(wrong)
        wrong.write_bytes(b'not a valid image')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            read_image(wrong)

    def test_transparency_blank_and_oversized_input_rejected(self):
        image = read_image(self.image).convert('RGBA')
        image.putalpha(0)
        path = self.root / 'transparent.png'
        image.save(path)
        with self.assertRaisesRegex(ValueError, 'Transparent'):
            read_image(path)
        read_image(self.image).point(lambda value: 0).save(path)
        with self.assertRaisesRegex(ValueError, 'blank'):
            read_image(path)
        with path.open('wb') as stream:
            stream.seek(MAX_IMAGE_BYTES)
            stream.write(b'0')
        with self.assertRaisesRegex(ValueError, '16 MB'):
            read_image(path)

    def test_reference_must_be_binary_and_lossless(self):
        reference = read_image(self.reference, reference=True)
        self.assertEqual(reference.dtype, np.bool_)
        with self.assertRaisesRegex(ValueError, 'lossless'):
            read_image(self.image, reference=True)
        path = self.root / 'badmask.png'
        Image.open(self.reference).convert('L').point(lambda value: 128 if value else 0).save(path)
        with self.assertRaisesRegex(ValueError, 'values'):
            read_image(path, reference=True)

    def test_non_square_geometry_and_padding(self):
        image = read_image(self.image).crop((10, 0, 210, 256))
        array, side, (left, top) = prepare_image(image)
        self.assertEqual(side, 256)
        self.assertEqual((left, top), (28, 0))
        self.assertEqual(array.shape, (3, 512, 512))
        self.assertTrue(np.all(array[..., :50] == 0))

    def test_exif_rotation_preserves_expected_pixel_orientation(self):
        image = read_image(self.image).crop((0, 0, 200, 256))
        metadata = image.getexif()
        metadata[274] = 6  # EXIF: rotate 90 degrees clockwise for display.
        path = self.root / 'rotated.jpg'
        image.save(path, exif=metadata)
        with Image.open(path) as stored:
            pixels = np.asarray(stored.convert('RGB'))
        decoded = read_image(path)
        self.assertEqual(decoded.size, (256, 200))
        np.testing.assert_array_equal(np.asarray(decoded), np.rot90(pixels, k=-1))

    def test_grayscale_supported_and_16_bit_image_rejected(self):
        path = self.root / 'gray.png'
        image = read_image(self.image).convert('L')
        image.save(path)
        pixels = np.asarray(read_image(path))
        np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
        Image.fromarray(np.asarray(image).astype(np.uint16)).save(path)
        with self.assertRaisesRegex(ValueError, '8-bit'):
            read_image(path)

    def test_empty_metric_policy_and_real_reference_identity(self):
        mask = read_image(self.reference, reference=True)
        empty = np.zeros_like(mask)
        self.assertEqual(mask_metrics(mask, mask)['dice'], 1)
        self.assertEqual(mask_metrics(empty, empty)['iou'], 1)
        self.assertEqual(mask_metrics(empty, mask)['dice'], 0)

    def test_overlay_zero_preserves_pixels_and_invalid_parameters_fail(self):
        read_image(self.image).save(self.root / 'image.png')
        read_image(self.reference, reference=True)
        (self.root / 'mask.png').write_bytes(self.mask)
        actual = np.asarray(Image.open(io.BytesIO(render_slice(self.root, opacity=0))))
        np.testing.assert_array_equal(actual, np.asarray(read_image(self.image)))
        for opacity in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError):
                render_slice(self.root, opacity=opacity)
        with self.assertRaises(ValueError):
            render_slice(self.root, panel='../image')

    def test_http_invalid_inputs_do_not_start_worker_or_hold_lock(self):
        # Only model availability is simulated: these invalid uploads must never load a model.
        from setup_2d_model import FILES
        weights = self.root / 'unloaded_weights'
        weights.mkdir()
        for spec in FILES[:3]:
            (weights / spec[0]).touch()
        with patch.object(local_ui, 'RUNS', self.root / 'runs'), patch('setup_2d_model.MODEL', weights):
            app = local_ui.LocalApp(warm_model=False)
            client = Client(app, Response)
            base = 'http://127.0.0.1:8765'
            client.get('/', base_url=base)
            headers = {'X-UI-Token': app.token}
            bad_cases = [
                {'mode': 'single_slice'},
                {'mode': 'single_slice', 'confirmed': 'yes'},
                {'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(b'invalid'), 'bad.png')},
                {'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(self.jpeg), 'scan.nii')},
                {'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(self.jpeg), 'scan.jpg'), 'reference': (io.BytesIO(self.jpeg), 'reference.jpg')},
                {'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(self.jpeg), 'scan.jpg'), 't2': (io.BytesIO(self.jpeg), 'extra.jpg')},
            ]
            smaller = io.BytesIO()
            Image.open(self.reference).crop((0, 0, 64, 64)).save(smaller, format='PNG')
            bad_cases.append({'mode': 'single_slice', 'confirmed': 'yes', 'image': (io.BytesIO(self.jpeg), 'scan.jpg'), 'reference': (io.BytesIO(smaller.getvalue()), 'smaller.png')})
            with patch.object(local_ui.threading, 'Thread') as worker:
                for data in bad_cases:
                    response = client.post('/api/predict', base_url=base, headers=headers, data=data)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertFalse(app.busy.locked())
                worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
