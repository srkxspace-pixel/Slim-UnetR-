# Live demonstration checklist

## Original paper model: three-minute presentation

Start the app using the README. Select **One 4D file** and upload:

- MRI: `examples/BRATS_409/image.nii.gz`
- Optional reference: `examples/BRATS_409/reference.nii.gz`
- Reference label scheme: **MSD Task01**

Read the four-modality confirmation, tick it and click **Run prediction**. The shortcut **Run real local example · BRATS_409** reads these same files and runs a fresh prediction. It does not display a cached result.

Say: “This is the authors' released pretrained Slim UNETR network. One complete MRI case contains four acquired modalities: FLAIR, T1, contrast-enhanced T1 and T2. The model predicts tumour core, whole tumour and enhancing tumour. Our application provides the local upload interface, visualization and export.”

After completion, switch the modality, plane, slice, region and opacity. Compare the predicted overlay with the supplied reference. Download the ZIP/report. Explain that the reference is used for scoring, never as an input to the network. Repeat without a reference to demonstrate that prediction does not require an annotation; no Dice/HD95 is then shown.

“The results are local model predictions, not a reproduction of every benchmark in the paper. Predictions can include missed regions and false positives.”

## If an assessor supplies new input

For the Slim UNETR demonstration request four aligned, skull-stripped 3D NIfTI modalities, or a four-channel 4D NIfTI with documented FLAIR/T1/T1ce/T2 ordering. NIfTI headers must provide valid millimetre geometry; the app does not register images or perform skull stripping. Correct file format alone cannot guarantee model suitability. Ask for a matching reference and its label meanings only if scoring is required.

For a JPG/PNG, select **JPG / PNG slice** after completing optional 2D setup. State clearly: “For this single T1 slice we use our additional BRISC SegFormer model. This result is separate from Slim UNETR.” It cannot provide TC/WT/ET, 3D anatomy or physical volume from one picture. Prefer clean T1-weighted, contrast-enhanced individual slices; other sequences/screenshots are unvalidated.

## Before presenting

1. Complete downloads and run one rehearsal while Internet is available; inference then runs offline.
2. Keep the app terminal open, connect laptop power, and close applications you do not need yourself.
3. Use CPU if there is no compatible NVIDIA GPU. CPU can take longer, particularly for the first case.
4. Keep at least 2 GB free disk space. Archive old `outputs/ui_runs` folders locally if 20 runs have accumulated.
5. Verify the browser upload on the actual presentation laptop. Automated browser visual verification was unavailable in the original development session; model and HTTP tests do not replace that rehearsal.
