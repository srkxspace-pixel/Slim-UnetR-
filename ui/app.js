'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="ui-token"]').content;
let ready = false, ready3d = false, running = false, externalBusy = false, currentJob = null, result = null;
let pollTimer, elapsedTimer, renderTimer, renderVersion = 0;
const form = $('prediction-form');
const isSliceInput = () => form.elements.mode.value === 'single_slice';
const isSliceResult = () => result && result.input_mode === 'single_slice';

function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers: {'X-UI-Token': token, ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function buttons() {
  $('run-button').disabled = !ready || running || externalBusy;
  $('real-example').disabled = !ready3d || running || externalBusy;
}
async function modelStatus() {
  try {
    const data = await api('/api/status');
    ready3d = data.model.state === 'ready';
    const selected = isSliceInput() ? data.slice_model : data.model;
    ready = isSliceInput() ? ['ready', 'available', 'failed'].includes(selected.state) : ready3d;
    externalBusy = data.busy && !running;
    $('model-status').className = `model-status ${selected.state}`;
    $('model-status').textContent = isSliceInput() ? selected.message : ready ? `● Ready · ${selected.gpu || 'CPU'} · frozen checkpoint` : selected.message;
    buttons();
  } catch (error) { $('model-status').textContent = 'Local server unavailable. Keep its terminal running, then reload this page.'; ready = false; buttons(); }
}
function setMode() {
  const four = form.elements.mode.value === 'four_files';
  const slice = isSliceInput();
  $('four-files').hidden = !four;
  $('four-dimensional').hidden = four;
  for (const role of ['flair', 't1', 't1ce', 't2']) {
    form.elements[role].disabled = !four;
    form.elements[role].required = four;
  }
  form.elements.image.disabled = four;
  form.elements.image.required = !four;
  form.elements.image.accept = slice ? '.jpg,.jpeg,.png' : '.nii,.nii.gz';
  form.elements.reference.accept = slice ? '.png' : '.nii,.nii.gz';
  // Do not silently reuse a file or acknowledgement after changing the input contract.
  form.elements.image.value = ''; form.elements.reference.value = ''; form.elements.confirmed.checked = false;
  form.elements.label_scheme.disabled = slice;
  form.elements.label_scheme.required = false;
  $('label-scheme-field').hidden = slice;
  $('image-input-label').textContent = slice ? 'One T1 MRI slice · JPG / PNG' : 'Four-channel MRI';
  $('input-description').textContent = slice ? '2D BRISC SegFormer-B5: one individual T1-weighted brain MRI slice, preferably contrast-enhanced. Binary tumour mask only.' : 'Aligned, skull-stripped 3D brain MRI in NIfTI format. All four modalities are required.';
  $('image-input-hint').textContent = slice ? 'Use an individual scan without collage, colour annotations or viewer controls. Other MRI sequences and screenshots are unvalidated. This model cannot produce TC/WT/ET or reconstruct a 3D scan.' : 'Channel order must be FLAIR → T1 → T1ce → T2. A 4D time series is not a substitute for these modalities.';
  $('reference-input-label').textContent = slice ? 'Binary reference PNG · same pixel dimensions' : 'Reference NIfTI';
  $('reference-help').textContent = slice ? 'Optional: 0 = background, 1 or 255 = tumour. Enables slice Dice and IoU. A coloured overlay is not a reference mask.' : 'Add a matching annotation to compare ground truth and calculate Dice / HD95.';
  $('confirmation-text').textContent = slice ? 'I confirm this is an individual T1-weighted brain MRI slice. I understand this is an unvalidated research prediction and an empty mask does not rule out tumour.' : 'I confirm these are the four stated brain MRI modalities, co-registered and skull-stripped, in the stated order.';
  $('upload-limit').textContent = slice ? 'Maximum 16 MB per image; 32–4096 pixels per side; at most 4 million pixels. Inputs and results stay on this laptop.' : 'Maximum combined upload: 384 MB. Files and generated results stay on this laptop.';
  modelStatus();
}
for (const radio of form.elements.mode) radio.addEventListener('change', setMode);
form.elements.reference.addEventListener('change', () => { form.elements.label_scheme.required = !isSliceInput() && form.elements.reference.files.length > 0; });

function beginProgress() {
  running = true; externalBusy = false; buttons(); clearError();
  result = null; currentJob = null; ++renderVersion;
  $('result').hidden = true; $('empty-state').hidden = false;
  $('job-progress').hidden = false;
  $('elapsed').textContent = '0 s';
  $('stage').textContent = 'Uploading MRI files';
  $('progress-detail').textContent = 'Transferring the selected files to the local application.';
  $('progress-bar').removeAttribute('value');
  const start = Date.now(); clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => { $('elapsed').textContent = `${Math.floor((Date.now() - start) / 1000)} s`; }, 1000);
}
function endProgress() { running = false; clearInterval(elapsedTimer); buttons(); }
function sendUpload(data) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/predict'); xhr.setRequestHeader('X-UI-Token', token);
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) { $('progress-bar').max = event.total; $('progress-bar').value = event.loaded; $('progress-detail').textContent = `Uploaded ${(event.loaded / 1048576).toFixed(1)} of ${(event.total / 1048576).toFixed(1)} MB to this laptop.`; }
    };
    xhr.onload = () => {
      try { const body = JSON.parse(xhr.responseText); if (xhr.status >= 400) reject(new Error(body.error)); else resolve(body); }
      catch (_) { reject(new Error('The local server returned an unreadable response. Check its terminal.')); }
    };
    xhr.onerror = () => reject(new Error('Upload connection failed. Check that the local server is running.'));
    xhr.send(data);
  });
}
form.addEventListener('submit', async event => {
  event.preventDefault(); if (!form.reportValidity() || running) return;
  const data = new FormData(form);
  let size = 0;
  for (const [key, value] of [...data.entries()]) if (value instanceof File) {
    if (!value.name) { data.delete(key); continue; }
    const allowed = isSliceInput() ? (key === 'reference' ? /\.png$/i : /\.(?:jpe?g|png)$/i) : /\.nii(?:\.gz)?$/i;
    if (!allowed.test(value.name)) { showError(isSliceInput() ? 'Select a JPG/PNG MRI and optionally a binary PNG reference.' : 'Select .nii or .nii.gz volumes, or switch to JPG / PNG slice mode.'); return; }
    if (isSliceInput() && value.size > 16 * 1048576) { showError('Each 2D image must be at most 16 MB.'); return; }
    size += value.size;
  }
  if (size > 384 * 1048576) { showError('The selected files exceed the combined upload limit of 384 MB.'); return; }
  beginProgress();
  try { const job = await sendUpload(data); watch(job.id); }
  catch (error) { endProgress(); $('job-progress').hidden = true; showError(error.message); }
});
$('real-example').addEventListener('click', async () => {
  if (running) return; beginProgress();
  $('stage').textContent = 'Opening real local example';
  $('progress-detail').textContent = 'Reading the existing BRATS_409 MRI and reference mask for fresh inference.';
  try { const job = await api('/api/real-example', {method: 'POST'}); watch(job.id); }
  catch (error) { endProgress(); $('job-progress').hidden = true; showError(error.message); }
});
function watch(id) {
  currentJob = id;
  sessionStorage.setItem('slimJob', id);
  clearTimeout(pollTimer);
  const poll = async () => {
    try {
      const job = await api(`/api/runs/${id}`);
      $('stage').textContent = job.stage;
      $('progress-detail').textContent = job.detail;
      if (job.windows_total && job.stage === 'Predicting') { $('progress-bar').max = job.windows_total; $('progress-bar').value = job.windows_done; }
      else $('progress-bar').removeAttribute('value');
      if (job.state === 'completed') {
        endProgress(); $('job-progress').hidden = true; result = job.result; displayResult(); modelStatus(); return;
      }
      if (job.state === 'failed') { endProgress(); $('job-progress').hidden = true; showError(job.detail); modelStatus(); return; }
      pollTimer = setTimeout(poll, 800);
    } catch (error) { endProgress(); showError(`Could not read run status: ${error.message} Reload to reconnect to the saved run.`); }
  };
  poll();
}
function displayResult() {
  $('empty-state').hidden = true; $('result').hidden = false;
  $('result-title').textContent = 'Prediction ready';
  const roles = result.input_mode === 'four_files' ? ['flair', 't1', 't1ce', 't2'] : ['image'];
  const names = roles.map(role => result.uploaded_names[role]).join(' · ');
  const slice = isSliceResult();
  $('result-title').textContent = slice ? '2D tumour mask ready' : '3D prediction ready';
  $('result-context').textContent = slice ? `${names} | ${result.model_name} | ${result.original_size.join(' × ')} pixels | inference ${result.inference_seconds.toFixed(2)} s, processing ${result.total_processing_seconds.toFixed(1)} s` : `${names} | ${result.processed_shape.join(' × ')} voxels at 1 mm | inference ${result.inference_seconds.toFixed(2)} s, total processing ${result.total_processing_seconds.toFixed(1)} s`;
  for (const id of ['axis', 'modality', 'region']) $(id).closest('label').hidden = slice;
  $('slice-control').hidden = slice; $('volume-legend').hidden = slice; $('slice-legend').hidden = !slice;
  $('prediction-caption').textContent = slice ? 'BRISC SegFormer · predicted tumour' : 'Slim UNETR prediction';
  $('slice-result-note').hidden = !slice;
  $('slice-result-note').textContent = slice ? `${result.predicted_pixels.toLocaleString()} predicted pixels (${result.predicted_area_percent.toFixed(2)}% of image). ${result.limitations}` : '';
  $('dice-heading').textContent = slice ? 'Slice Dice' : 'Whole-volume Dice';
  $('distance-heading').textContent = slice ? 'Slice IoU' : 'HD95';
  $('metric-note').textContent = slice ? 'Metrics use the original oriented pixel grid. Both masks empty: Dice and IoU = 1; one empty: 0. Physical distance and tumour volume cannot be measured from these images.' : 'An empty reference region has undefined Dice. If exactly one mask is empty, HD95 is infinite. Metrics describe this uploaded case, not reproduced paper results.';
  $('verification').textContent = result.model_state_unchanged && result.checkpoint_unchanged ? '✓ Model and checkpoint unchanged · no training' : 'Verification did not pass';
  $('download-bundle').href = `/api/runs/${currentJob}/download?file=prediction_bundle.zip`;
  $('download-report').href = `/api/runs/${currentJob}/download?file=report.json`;
  $('axis').value = 'axial'; $('modality').value = 'flair'; $('region').value = 'all';
  $('reference-image').hidden = !result.reference_provided; $('no-reference').hidden = result.reference_provided;
  $('metrics-table-wrap').hidden = !result.metrics;
  $('metrics-message').textContent = result.metrics ? (slice ? 'Binary tumour-mask comparison against your supplied reference for this slice.' : 'Per-region metrics compare the complete 3D prediction with your reference annotation.') : `No reference mask supplied. Predictions are available; ${slice ? 'Dice and IoU' : 'Dice and HD95'} cannot be calculated without reference labels.`;
  $('metrics-body').replaceChildren();
  if (slice && result.metrics) {
    const row = document.createElement('tr');
    for (const text of ['Tumour', `${(result.metrics.dice * 100).toFixed(2)}%`, `${(result.metrics.iou * 100).toFixed(2)}%`]) {
      const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
    }
    $('metrics-body').append(row);
  }
  if (!slice && result.metrics) for (const region of ['TC', 'WT', 'ET']) {
    const m = result.metrics[region], row = document.createElement('tr');
    const values = [region, m.dice === null ? 'Undefined · empty reference' : `${(m.dice * 100).toFixed(2)}%`, m.hd95_mm === 'Infinity' ? '∞ · one mask empty' : m.hd95_mm === null ? 'Undefined · both empty' : `${m.hd95_mm.toFixed(2)} mm`];
    for (const text of values) { const cell = document.createElement('td'); cell.textContent = text; row.append(cell); }
    $('metrics-body').append(row);
  }
  if (slice) updateImages(); else updateAxis(result.default_slice);
}
function updateAxis(initial) {
  if (!result) return;
  const dimension = {axial: 2, coronal: 1, sagittal: 0}[$('axis').value], length = result.processed_shape[dimension];
  $('slice').max = length - 1;
  $('slice').value = Number.isInteger(initial) ? Math.min(initial, length - 1) : Math.floor(length / 2);
  $('slice-total').textContent = ` / ${length - 1}`;
  updateImages();
}
function updateImages() {
  if (!result) return;
  const version = ++renderVersion;
  $('slice-value').textContent = $('slice').value;
  $('opacity-value').textContent = `${$('opacity').value}%`;
  $('mri-caption').textContent = isSliceResult() ? 'Uploaded MRI slice' : `MRI · ${$('modality').selectedOptions[0].textContent}`;
  $('viewer-status').textContent = isSliceResult() ? 'Loading 2D result…' : `Loading ${$('axis').value} slice…`;
  const params = new URLSearchParams({axis: $('axis').value, index: $('slice').value, modality: $('modality').value, region: $('region').value, opacity: Number($('opacity').value) / 100});
  const panels = result.reference_provided ? ['mri', 'reference', 'prediction'] : ['mri', 'prediction'];
  let loaded = 0;
  for (const panel of panels) {
    const image = $(`${panel}-image`);
    image.onload = () => { if (version === renderVersion && ++loaded === panels.length) $('viewer-status').textContent = isSliceResult() ? 'Original oriented image pixel grid. Download mask.png for the binary prediction; red overlay shows predicted tumour pixels.' : `Same ${$('axis').value} slice in all panels · preprocessed RAS / 1 mm space. Downloads preserve the separate TC / WT / ET masks.`; };
    image.onerror = () => { if (version === renderVersion) $('viewer-status').textContent = 'Slice could not load. Check the local server and reselect the slice.'; };
    image.src = `/api/runs/${currentJob}/slice?${params}&panel=${panel}`;
  }
}
for (const id of ['region', 'modality']) $(id).addEventListener('change', updateImages);
$('axis').addEventListener('change', () => updateAxis());
for (const id of ['slice', 'opacity']) $(id).addEventListener('input', () => {
  $('slice-value').textContent = $('slice').value; $('opacity-value').textContent = `${$('opacity').value}%`;
  clearTimeout(renderTimer); renderTimer = setTimeout(updateImages, 100);
});
setMode(); modelStatus(); setInterval(modelStatus, 4000);
const restored = sessionStorage.getItem('slimJob');
if (restored && /^[a-f0-9]{32}$/.test(restored)) { beginProgress(); watch(restored); }
