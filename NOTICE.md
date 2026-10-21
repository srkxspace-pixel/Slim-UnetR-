# Attribution and scope

## Slim UNETR

Core architecture and original training/evaluation code: copyright (c) 2023 deepang-ai, under the [MIT license](LICENSE). The upstream copyright and license are retained.

- Authors' repository: https://github.com/deepang-ai/Slim-UNETR
- Release used for the model: https://github.com/deepang-ai/Slim-UNETR/releases/tag/v1.0.0
- Paper: Pang et al., *Slim UNETR: Scale Hybrid Transformers to Efficient 3D Medical Image Segmentation Under Limited Computational Resources*, IEEE TMI, DOI https://doi.org/10.1109/TMI.2023.3326188.

Local additions include the browser UI, frozen inference adapters, setup scripts, validation tests, downloads, portable examples, and documentation. The MSD enhancing-tumour label mapping follows the dataset manifest (label 3); see [evaluation notes](docs/EVALUATION.md). These additions do not assert reproduction of the paper's benchmark scores.

## Bundled MRI data

`examples/BRATS_409/image.nii.gz` and `reference.nii.gz` are public MSD Task01 data under **CC BY-SA 4.0**, not MIT. See the [sample attribution](examples/BRATS_409/README.md). The full dataset, uploaded scans, and generated results are not committed.

## Optional 2D model

JPG/PNG segmentation uses a separate pretrained BRISC SegFormer-B5 model from `imbenjita/BrainTumorClassifier`. No downloaded model weights are committed. Its source card describes research-use terms, and its NVIDIA backbone has separate licensing. See [model provenance](docs/models/BRISC_PROVENANCE.md) and the retained [NVIDIA license](docs/models/NVIDIA_LICENSE.txt). Installing that optional model does not change the identity or claimed results of Slim UNETR.
