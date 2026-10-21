"""Portable sample and verified-download failure handling; no model execution."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from setup_2d_model import download
from src.demo_case import demo_case


class PortableDemoTests(unittest.TestCase):
    def test_bundled_real_case_hashes_are_valid(self):
        self.assertEqual(set(demo_case()), {'image', 'reference'})

    def test_missing_demo_file_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch('src.demo_case.EXAMPLE', Path(folder)):
            with self.assertRaisesRegex(ValueError, 'missing'):
                demo_case()

    def test_corrupt_existing_download_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact.bin'
            path.write_bytes(b'original-invalid-content')
            with self.assertRaisesRegex(RuntimeError, 'unexpected hash'):
                download(Path(folder), ('artifact.bin', 'https://example.invalid/weights', '0' * 64, 'sha256'))
            self.assertEqual(path.read_bytes(), b'original-invalid-content')

    def test_checksum_failure_is_bounded_and_not_installed(self):
        expected = hashlib.sha256(b'expected-artifact').hexdigest()
        with tempfile.TemporaryDirectory() as folder, patch('setup_2d_model.urllib.request.urlopen', side_effect=lambda *a, **k: io.BytesIO(b'bad-download')) as network, patch('setup_2d_model.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Checksum failed'):
                download(Path(folder), ('artifact.bin', 'https://example.invalid/weights', expected, 'sha256'))
            self.assertEqual(network.call_count, 3)
            self.assertFalse((Path(folder) / 'artifact.bin').exists())
            self.assertFalse((Path(folder) / 'artifact.bin.partial').exists())


if __name__ == '__main__':
    unittest.main()
