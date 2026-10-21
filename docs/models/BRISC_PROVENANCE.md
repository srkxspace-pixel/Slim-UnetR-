# Local BRISC 2D model provenance

Purpose: single-slice T1-weighted MRI binary tumour-segmentation research demo. This is a different model from the project's 3D Slim UNETR. No classifier, Grad-CAM substitute, generated MRI, training or fine-tuning is used.

Source: [imbenjita/BrainTumorClassifier](https://huggingface.co/imbenjita/BrainTumorClassifier), revision `04c94c44998bd793e97dd8c1a539fbfbd21c958b`.

- `seg_best.pt`: trained BRISC decoder, SHA256 `20947b5b737b47de3631fdb9930f6d81c79b92703c8a65f554b948e1ddfd07ee`.
- `upstream_reference.py`: source retained for inspection, never imported or executed.
- `UPSTREAM_MODEL_CARD.md`: source model card, including research-use terms and limitations. Its classification performance numbers are **not segmentation scores** and are not used by our UI.

Frozen encoder: [nvidia/segformer-b5-finetuned-ade-640-640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640), revision `739f5d4692954e4a185eac280dec1ba5a7d52f1d`.

- `base_model.bin`: SHA256 `a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df`.
- Only encoder tensors from this file are used; every decoder tensor comes from the BRISC checkpoint. Complete model loading is strict and shape-checked. No randomly initialized decoder is used.
- `config.json`: upstream Git blob `3cf1f37a73b0e34d1fd6fda13760f9037a91c46e`.
- NVIDIA's linked SegFormer license is retained in `NVIDIA_LICENSE.txt`. The source model card describes the fine-tuned checkpoint as research-use. No unrestricted commercial license is asserted for the combined model.

Original validation runtime: PyTorch 2.5.1+cu121, Transformers 4.57.6, Pillow 12.3.0. New installations use the README's pinned PyTorch 2.6.0 CPU/CUDA build; optional additions are pinned in `requirements-2d.txt`. Transformers 5 calls decoder projections `linear_projections`; the pinned 4.x runtime calls them `linear_c`. The loader applies only this exact key alias and then requires the exact complete decoder key set and matching shapes. Encoder tensors are used unchanged.

Preprocessing follows the source: RGB display channels (a grayscale slice is represented in RGB; these are not substituted MRI modalities), centred black square padding, PIL bilinear resize to 512×512, float /255, ImageNet mean/std. Bilinear upsample logits to 512×512 with align_corners=False, argmax binary labels; restore binary mask by nearest-neighbour resizing/cropping to the oriented input pixel grid. The report records this geometry; no physical spacing is invented.

Verification data: [the BRISC author's public release](https://huggingface.co/datasets/amirreza-fateh/brisc), revision `6aafaf9cfe043e74a1d2a45222339096f220046b`; archive SHA256 `85511543c2ad9abff44e68ee5abab353a5089d5b6f241960c2ea7462aaf4262b`. That release labels its dataset Apache-2.0; retain source attribution and consult original dataset terms for redistribution. Only selected source members are extracted safely by name. Source reference bytes are retained, with the author's nonzero-foreground mapping recorded for binary evaluation.

Acceptance is functioning local inference and explicit model/input boundaries, not correct segmentation of every new picture. The small evaluation includes failures, and patient-level independence from checkpoint training is not established. See `outputs/verification_2d/verification.json`, `cpu_verification.json` and `LOCAL_UI.md`.
