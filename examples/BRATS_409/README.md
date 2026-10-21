# Real 3D MRI example: BRATS_409

This is one unchanged, publicly released Medical Segmentation Decathlon (MSD) Task01 BrainTumour image/reference pair. It is bundled so a fresh clone can demonstrate the authors' Slim UNETR model without downloading a whole dataset.

| File | Contents |
| --- | --- |
| `image.nii.gz` | 240 × 240 × 155 × 4 MRI; channel order **FLAIR, T1, T1 contrast, T2** |
| `reference.nii.gz` | Matching 240 × 240 × 155 expert reference segmentation |

Reference values: **0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour**. Choose **MSD Task01** in the app. Do not use the BraTS2021 numeric-label mapping for this file.

In the app choose **One 4D file**, upload `image.nii.gz`, add `reference.nii.gz`, choose MSD Task01, confirm the modality requirements and run prediction. Alternatively, **Run real local example · BRATS_409** loads these same bundled files and runs fresh inference. The reference is used for scoring, never as model input.

## Data license and attribution

These two files are **CC BY-SA 4.0**, separately from the repository's MIT source-code license. You may redistribute them under those terms with attribution. They are an unmodified subset of the public MSD Task01 release, sourced locally as `imagesTr/BRATS_409.nii.gz` and `labelsTr/BRATS_409.nii.gz`; only their filenames/location here changed. This is acquired MRI, not generated or synthetic content.

- [Medical Segmentation Decathlon: data and license](https://medicaldecathlon.com/)
- Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications 13, 4128 (2022), [doi:10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9).
- Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*, [arXiv:1902.09063](https://arxiv.org/abs/1902.09063).
- Brain-tumour source acknowledgement: BraTS contributors, Bakas et al., *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features*, [doi:10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117).
- [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) and [full legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).

SHA256:

```text
image.nii.gz      74a4d182237e887ba5ec58b5fc4762c3b7913388d3122d1285525aacf9eeead2
reference.nii.gz  a4c0573207472dc2a9cd334d36aee31dd555a28de5ad301bbcdaeac89ecf021e
```

These files are used for a research demonstration, not as an independently established held-out benchmark. Patient overlap with the released checkpoint's training population has not been verified. Their presence does not reproduce the paper's BraTS2021 result.
