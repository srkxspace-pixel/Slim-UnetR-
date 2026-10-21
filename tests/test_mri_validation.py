"""Input rejection checks using existing acquired MRI and its true annotation."""
from pathlib import Path
import unittest

import yaml


class RealMRIValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.demo_case import demo_case
        files = demo_case()
        cls.image = files['image']
        cls.reference = files['reference']
        if not cls.image.is_file() or not cls.reference.is_file():
            raise unittest.SkipTest('The existing real BRATS_409 MRI and reference are required.')
        from src.local_inference import validate_inputs
        cls.validate = staticmethod(validate_inputs)

    def test_single_3d_volume_cannot_replace_four_modalities(self):
        with self.assertRaisesRegex(ValueError, 'Expected a 4D NIfTI'):
            self.validate({'image': self.reference}, 'four_dimensional', '')

    def test_missing_modality_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'all four modalities are required'):
            self.validate({'flair': self.image}, 'four_files', '')

    def test_msd_reference_cannot_be_mislabelled_as_brats(self):
        with self.assertRaisesRegex(ValueError, 'Reference labels do not match brats'):
            self.validate({'image': self.image, 'reference': self.reference}, 'four_dimensional', 'brats')


if __name__ == '__main__':
    unittest.main()
