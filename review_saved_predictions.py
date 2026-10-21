"""Read saved local results and make explanatory figures; never run or modify a model."""

import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, label as connected_components
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/local_validation_97"
OUT = RUN / "review"
REGIONS = ["TC", "WT", "ET"]
COLORS = ListedColormap([(0, 0, 0, 0), (0.1, 0.8, 0.2, 0.6), (1, 0.85, 0, 0.65), (1, 0.1, 0.1, 0.8)])
ERROR_COLORS = ListedColormap([(0, 0, 0, 0), (0.1, 0.9, 0.4, 0.75), (1, 0.15, 0.15, 0.85), (0.15, 0.5, 1, 0.85)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_map(regions):
    result = np.zeros(regions.shape[1:], np.uint8)
    result[regions[1]] = 1
    result[regions[0]] = 2
    result[regions[2]] = 3
    return result


def errors(pred, gt):
    result = np.zeros(pred.shape, np.uint8)
    result[pred & gt] = 1
    result[pred & ~gt] = 2
    result[gt & ~pred] = 3
    return result


def show(ax, image, overlay=None, cmap=COLORS, bounds=None, marker=None):
    if bounds is None:
        bounds = (0, image.shape[0], 0, image.shape[1])
    x0, x1, y0, y1 = bounds
    ax.imshow(np.rot90(image[x0:x1, y0:y1]), cmap="gray", vmin=0, vmax=1)
    if overlay is not None:
        ax.imshow(np.rot90(overlay[x0:x1, y0:y1]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    if marker is not None:
        x, y = marker
        ax.plot(x - x0, y1 - 1 - y, marker="+", color="cyan", markersize=12, markeredgewidth=1.5)
    ax.axis("off")


def region_diagnostics(pred, gt, spacing):
    pcount, gcount = int(pred.sum()), int(gt.sum())
    cc, count = connected_components(pred, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(cc.ravel())[1:]
    intersections = np.bincount(cc[gt], minlength=count + 1)[1:]
    detached = sizes[intersections == 0]
    result = {
        "ground_truth_voxels": gcount, "predicted_voxels": pcount,
        "true_positive_voxels": int((pred & gt).sum()),
        "false_positive_voxels": int((pred & ~gt).sum()),
        "false_negative_voxels": int((gt & ~pred).sum()),
        "predicted_components_26_connectivity": int(count),
        "largest_predicted_component_voxels": int(sizes.max()) if len(sizes) else 0,
        "components_without_any_ground_truth_overlap": int(len(detached)),
        "voxels_in_components_without_gt_overlap": int(detached.sum()),
    }
    if not gcount or not pcount:
        result["distance_status"] = "one mask empty" if bool(gcount) != bool(pcount) else "both masks empty"
        point = np.argwhere(pred).mean(axis=0).astype(int) if pcount else np.argwhere(gt).mean(axis=0).astype(int)
        return result, point
    ps = np.argwhere(pred ^ binary_erosion(pred))
    gs = np.argwhere(gt ^ binary_erosion(gt))
    pg = cKDTree(gs * spacing).query(ps * spacing)[0]
    gp = cKDTree(ps * spacing).query(gs * spacing)[0]
    p95, g95 = float(np.percentile(pg, 95)), float(np.percentile(gp, 95))
    result.update({
        "pred_to_gt_surface_p95_mm": p95, "gt_to_pred_surface_p95_mm": g95,
        "predicted_surface_percent_over_50mm_from_gt": float(100 * np.mean(pg > 50)),
        "gt_surface_percent_over_50mm_from_prediction": float(100 * np.mean(gp > 50)),
        "dominant_hd95_direction": "prediction to reference" if p95 >= g95 else "reference to prediction",
    })
    points, distances, target95 = (ps, pg, p95) if p95 >= g95 else (gs, gp, g95)
    point = points[int(np.argmin(np.abs(distances - target95)))]
    return result, point


def detail(case, image, pred, gt, region, destination):
    rid = REGIONS.index(region)
    spacing = np.linalg.norm(nib.load(case["prediction_regions_nifti"]).affine[:3, :3], axis=0)
    stats, point = region_diagnostics(pred[rid], gt[rid], spacing)
    if isinstance(case["hd95_mm"][region], (float, int)):
        assert np.isclose(max(stats["pred_to_gt_surface_p95_mm"], stats["gt_to_pred_surface_p95_mm"]), case["hd95_mm"][region], atol=1e-4)
    # For empty reference masks, use the axial slice containing most predicted voxels.
    ztail = int(pred[rid].sum(axis=(0, 1)).argmax()) if not gt[rid].any() else int(point[2])
    if not gt[rid].any():
        slice_points = np.argwhere(pred[rid, ..., ztail])
        point = np.rint(slice_points.mean(axis=0)).astype(int)
    zmain = int((gt[rid] if gt[rid].any() else gt[1]).sum(axis=(0, 1)).argmax())
    x, y = int(point[0]), int(point[1])
    radius = 24
    bounds = (max(0, x-radius), min(image.shape[0], x+radius+1), max(0, y-radius), min(image.shape[1], y+radius+1))
    error = errors(pred[rid], gt[rid])
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.8))
    for row, (z, crop, row_label) in enumerate([(zmain, None, "Reference-region slice"), (ztail, None, "Error-region slice"), (ztail, bounds, "Error-region zoom")]):
        for column, (title, overlay, cmap) in enumerate([
            ("FLAIR", None, COLORS),
            (f"Reference {region}", gt[rid, ..., z].astype(np.uint8) * 2, COLORS),
            (f"Prediction {region}", pred[rid, ..., z].astype(np.uint8) * 3, COLORS),
            ("Agreement / errors", error[..., z], ERROR_COLORS),
        ]):
            show(axes[row, column], image[..., z], overlay, cmap, crop, (x, y) if row > 0 else None)
            axes[row, column].set_title(f"{title}\n{row_label}, z={z}", fontsize=10)
    d = case["dice"][region]
    dice_text = "undefined (empty reference)" if d is None else f"{100*d:.2f}%"
    h = case["hd95_mm"][region]
    hd_text = f"{h:.2f} mm" if isinstance(h, (float, int)) else "infinite (one mask empty)"
    fig.suptitle(f"{case['case']} | focus {region} | Dice {dice_text} | HD95 {hd_text}\nOUR LOCAL EVALUATION — saved outputs only; no new inference or training", fontsize=12)
    fig.text(0.5, 0.025, "Error panel: green = overlap   red = false positive   blue = missed reference   cyan cross = selected error location", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.93))
    fig.savefig(destination, dpi=135)
    plt.close(fig)
    stats.update({"region": region, "main_slice_z": zmain, "error_slice_z": ztail, "error_xy": [x, y], "detail_png": destination.name})
    return stats


def main():
    OUT.mkdir(exist_ok=True)
    report_path = RUN / "run_report.json"
    report_hash = sha(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha(report["checkpoint"])
    details = {"BRATS_026": ["ET"], "BRATS_024": ["ET", "WT"], "BRATS_128": ["ET"], "BRATS_171": ["ET"], "BRATS_012": ["WT"], "BRATS_327": ["TC"]}
    review = {"label": "LOCAL EVALUATION — SAVED PREDICTIONS, NOT LIVE INFERENCE OR PAPER REPRODUCTION", "cases": [], "diagnostics": []}
    for index, case in enumerate(report["cases"]):
        case_id = case["case"]
        raw = nib.as_closest_canonical(nib.load(case["source_image"]))
        # Read only the actual FLAIR modality; no generated or missing input channels.
        flair = np.asarray(raw.dataobj[..., 0], dtype=np.float32)
        ref = nib.as_closest_canonical(nib.load(case["source_label"]))
        labels = np.asarray(ref.dataobj)
        saved = nib.load(case["prediction_regions_nifti"])
        assert np.allclose(raw.affine, ref.affine) and np.allclose(ref.affine, saved.affine), case_id
        pred = np.moveaxis(np.asarray(saved.dataobj), -1, 0).astype(bool)
        gt = np.stack([(labels == 2) | (labels == 3), labels > 0, labels == 3])
        assert flair.shape == pred.shape[1:] == gt.shape[1:], case_id
        nonzero = flair[flair != 0]
        lo, hi = np.percentile(nonzero, [1, 99])
        image = np.clip((flair - lo) / max(float(hi-lo), 1e-6), 0, 1)
        z = int(gt[1].sum(axis=(0, 1)).argmax())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for ax, title, overlay in zip(axes, ["Existing MRI — FLAIR", "Dataset ground truth", "Saved model prediction"], [None, class_map(gt)[..., z], class_map(pred)[..., z]]):
            show(ax, image[..., z], overlay)
            ax.set_title(title, fontsize=11)
        score = "  ".join(f"{r} {100*case['dice'][r]:.1f}%" if case['dice'][r] is not None else f"{r} n/a" for r in REGIONS)
        fig.suptitle(f"{case_id} | whole-volume Dice: {score}\nLOCAL EVALUATION — saved prediction | axial slice {z}", fontsize=11)
        fig.text(0.5, 0.02, "Green: WT-only / edema   Yellow: core excluding ET   Red: enhancing tumour", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.045, 1, 0.9))
        filename = f"{case_id}_overview.png"
        fig.savefig(OUT / filename, dpi=115)
        plt.close(fig)
        item = {"case": case_id, "overview": filename, "slice_z": z, "dice": case["dice"], "hd95_mm": case["hd95_mm"], "details": []}
        for region in details.get(case_id, []):
            filename = f"{case_id}_{region}_detail.png"
            diagnostic = detail(case, image, pred, gt, region, OUT / filename)
            review["diagnostics"].append({"case": case_id, **diagnostic})
            item["details"].append({"region": region, "file": filename})
            print(json.dumps({"case": case_id, **diagnostic}), flush=True)
        review["cases"].append(item)
        if (index + 1) % 10 == 0:
            print(f"Reviewed saved cases {index + 1}/97", flush=True)
        del raw, ref, saved, flair, labels, image, pred, gt
    assert sha(report_path) == report_hash and sha(report["checkpoint"]) == checkpoint_hash
    review["original_report_unchanged"] = True
    review["checkpoint_unchanged"] = True
    (OUT / "review_data.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    options = "".join(f'<option value="{i}">{html.escape(item["case"])}</option>' for i, item in enumerate(review["cases"]))
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local MRI prediction review</title>
<style>body{font:16px system-ui;margin:0;background:#f5f6f8;color:#18212d}main{max-width:1320px;margin:auto;padding:24px}h1{font-size:26px;margin-bottom:8px}.note{line-height:1.6;color:#455266}label{display:inline-block;margin:10px 18px 10px 0}select{font:inherit;padding:8px}img{display:block;width:100%;height:auto;background:white;margin-top:16px}#metrics{line-height:1.8;font-variant-numeric:tabular-nums}a{color:#164cad}button{font:inherit;padding:8px 14px;margin-right:8px;cursor:pointer}.legend{margin:12px 0}footer{margin-top:20px;font-size:14px}</style></head><body><main>
<h1>Our local MRI evaluation — 97 saved predictions</h1><p class="note">Official BraTS 2021 checkpoint applied to local MSD Task01 cases. This viewer displays saved outputs; selecting a case does not run inference. These are not reproduced paper results. Patient overlap with training is unverified.</p>
<label>Case <select id="case">OPTIONS</select></label><label>View <select id="view"></select></label><button id="prev">Previous case</button><button id="next">Next case</button>
<div id="metrics" aria-live="polite"></div><p class="note" id="caption"></p><a id="full" target="_blank">Open full-size figure</a><img id="figure" alt="Actual MRI, ground truth and saved prediction comparison">
<footer>Overview figures show one selected axial slice, chosen by greatest reference WT area. Whole-volume metrics come from the complete 3D masks. Detailed failure figures select an error slice and zoom; they do not change predictions. Full 3D volumes remain in the parent folder for a NIfTI viewer.<br><a href="../README.md">Evaluation method</a> · <a href="../per_case_metrics.csv">All per-case metrics</a></footer></main>
<script>const cases=DATA;const cs=document.getElementById('case'),vs=document.getElementById('view');function fmt(v,pct){return v===null?'undefined':typeof v==='string'?'∞':(pct?v*100:v).toFixed(2)+(pct?'%':' mm')}function draw(){const c=cases[+cs.value],detail=c.details.find(x=>x.region===vs.value),src=detail?detail.file:c.overview;document.getElementById('figure').src=src;document.getElementById('full').href=src;document.getElementById('metrics').textContent=['TC','WT','ET'].map(r=>r+': Dice '+fmt(c.dice[r],true)+', HD95 '+fmt(c.hd95_mm[r],false)).join('   |   ');document.getElementById('caption').textContent=detail?'Failure detail: '+detail.region+'; overlap green, false positives red, misses blue. Cyan cross marks the selected error location.':'All regions: green WT-only, yellow core excluding ET, red enhancing tumour. Overview slice z='+c.slice_z+'.'}function select(){const c=cases[+cs.value];vs.replaceChildren(new Option('One-slice overview','overview'),...c.details.map(x=>new Option(x.region+' failure detail',x.region)));draw()}cs.addEventListener('change',select);vs.addEventListener('change',draw);document.getElementById('prev').onclick=()=>{cs.value=(+cs.value+cases.length-1)%cases.length;select()};document.getElementById('next').onclick=()=>{cs.value=(+cs.value+1)%cases.length;select()};cs.value=cases.findIndex(c=>c.case==='BRATS_026');select();</script></body></html>'''
    page = page.replace("OPTIONS", options).replace("DATA", json.dumps(review["cases"]))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Done: {OUT / 'index.html'}; original report and checkpoint unchanged.", flush=True)


if __name__ == "__main__":
    main()
