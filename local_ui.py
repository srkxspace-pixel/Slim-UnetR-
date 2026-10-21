"""Loopback-only MRI UI: 3D Slim UNETR and 2D BRISC SegFormer inference."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import secrets
import shutil
import threading
import uuid

from werkzeug.exceptions import HTTPException
from werkzeug.serving import run_simple
from werkzeug.utils import send_file
from werkzeug.wrappers import Request, Response

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs/ui_runs"
STATIC = ROOT / "ui"
MAX_UPLOAD = 384 * 1024 * 1024
LOG = logging.getLogger("slim_ui")


class UploadRequest(Request):
    max_content_length = MAX_UPLOAD
    max_form_memory_size = 1024 * 1024
    max_form_parts = 12


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class LocalApp:
    def __init__(self, port=8765, warm_model=True, device='auto'):
        self.port = port
        self.device = device
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.guard = threading.Lock()
        self.busy = threading.Lock()
        self.engine = None
        self.slice_engine = None
        self.slice_model_status = {"state": "available", "message": "BRISC SegFormer-B5 · loads on first 2D prediction"}
        self.model_status = {"state": "loading", "message": "Loading the official checkpoint once. You can choose files while it loads."}
        RUNS.mkdir(parents=True, exist_ok=True)
        if warm_model:
            threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            from src.local_inference import Engine
            self.engine = Engine(device=self.device)
            self.model_status = {"state": "ready", "message": "Official checkpoint ready. Inference only.", **self.engine.info}
            LOG.info("Official checkpoint loaded; device=%s", self.engine.device)
        except Exception as error:
            LOG.exception("Model initialization failed")
            self.model_status = {"state": "failed", "message": f"Model could not load: {error}"}

    def response_json(self, value, status=200):
        return Response(json.dumps(value, allow_nan=False), status=status, mimetype="application/json")

    def authorize(self, request, write=False):
        allowed = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if request.host not in allowed:
            raise ValueError("This application only accepts localhost requests.")
        origin = request.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            raise PermissionError("Cross-origin requests are not accepted.")
        if request.path.startswith("/api/"):
            if not secrets.compare_digest(request.cookies.get("slim_ui", ""), self.token):
                raise PermissionError("Open the local UI in this browser before using the API.")
            if write and not secrets.compare_digest(request.headers.get("X-UI-Token", ""), self.token):
                raise PermissionError("Invalid local request token. Reload the page.")

    def new_job(self, mode=None):
        if mode == 'single_slice':
            from setup_2d_model import MODEL, FILES
            if not all((MODEL / spec[0]).is_file() for spec in FILES[:3]):
                return None, self.response_json({"error": "Install the 2D model first: run setup_2d_model.py, then retry."}, 503)
        elif self.model_status["state"] != "ready":
            return None, self.response_json({"error": self.model_status["message"]}, 503)
        if not self.busy.acquire(blocking=False):
            return None, self.response_json({"error": "Another case is being uploaded or processed. Wait for it to finish."}, 409)
        try:
            if shutil.disk_usage(RUNS).free < 2 * 1024**3:
                raise ValueError("At least 2 GB of free disk space is required. Archive older local output folders first.")
            if len(list(RUNS.glob("*/status.json"))) >= 20:
                raise ValueError("Twenty UI runs are stored. Archive older outputs/ui_runs folders before starting another case.")
            ident = uuid.uuid4().hex
            directory = RUNS / ident
            directory.mkdir()
            job = {"id": ident, "state": "running", "stage": "Receiving inputs", "detail": "Saving the selected MRI files locally.", "started_utc": utc_now()}
            self.jobs[ident] = job
            self.save_status(job)
            return job, None
        except Exception:
            self.busy.release()
            raise

    def save_status(self, job):
        directory = RUNS / job["id"]
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(directory / "status.json")

    def failed(self, job, message):
        with self.guard:
            job.update(state="failed", stage="Could not complete prediction", detail=message, completed_utc=utc_now())
            self.save_status(job)

    def run_job(self, job, files, names, mode, scheme):
        def progress(stage, detail, done=None, total=None):
            with self.guard:
                job.update(stage=stage, detail=detail, windows_done=done, windows_total=total)
                self.save_status(job)
        try:
            if mode == 'single_slice':
                if self.slice_engine is None:
                    progress('Loading 2D model', 'Loading verified BRISC SegFormer-B5 weights locally. First run takes longer.')
                    self.slice_model_status = {"state": "loading", "message": "Loading BRISC SegFormer-B5"}
                    from src.slice_inference import SliceEngine
                    self.slice_engine = SliceEngine(device=self.device)
                    self.slice_model_status = {"state": "ready", "message": "BRISC SegFormer-B5 ready", **self.slice_engine.info}
                result = self.slice_engine.predict(RUNS / job['id'], files, names, progress)
            else:
                result = self.engine.predict(RUNS / job["id"], files, names, mode, scheme, progress)
            with self.guard:
                job.update(state="completed", stage="Prediction complete", detail="Saved masks and report are ready. Model and checkpoint unchanged.",
                           result=result, completed_utc=utc_now())
                self.save_status(job)
            LOG.info("Run %s completed", job["id"])
        except Exception as error:
            if mode == 'single_slice' and self.slice_engine is None:
                self.slice_model_status = {"state": "failed", "message": str(error)}
            LOG.exception("Run %s failed", job["id"])
            self.failed(job, str(error) if isinstance(error, (ValueError, RuntimeError)) else "The input could not be processed. Check the terminal log for details.")
            if any(engine is not None and engine.device == "cuda" for engine in (self.engine, self.slice_engine)):
                import torch
                torch.cuda.empty_cache()
        finally:
            self.busy.release()

    def upload(self, request):
        mode = request.form.get("mode", "")
        if mode == 'single_slice':
            return self.upload_slice(request)
        job, error = self.new_job()
        if error is not None:
            return error
        try:
            mode = request.form.get("mode", "")
            if request.form.get("confirmed") != "yes":
                raise ValueError("Confirm that the four modalities are co-registered, skull-stripped brain MRI inputs in the stated order.")
            if mode not in ("four_files", "four_dimensional"):
                raise ValueError("Select an input format.")
            roles = {"flair", "t1", "t1ce", "t2"} if mode == "four_files" else {"image"}
            supplied = {key for key, value in request.files.items() if value.filename}
            if supplied not in (roles, roles | {"reference"}):
                raise ValueError("Select all required MRI files. Only an optional reference mask may be added.")
            files, names = {}, {}
            for role in supplied:
                items = request.files.getlist(role)
                if len(items) != 1:
                    raise ValueError("Select one file for each requested input.")
                uploaded = items[0]
                name = uploaded.filename.replace("\\", "/").rsplit("/", 1)[-1]
                lower = name.lower()
                suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii" if lower.endswith(".nii") else None
                if suffix is None:
                    raise ValueError("Only .nii and .nii.gz medical volumes are accepted. JPG/PNG inputs are not compatible.")
                path = RUNS / job["id"] / f"input_{role}{suffix}"
                uploaded.save(path)
                if path.stat().st_size == 0:
                    raise ValueError("An uploaded file is empty.")
                files[role], names[role] = path, name[:200]
            scheme = request.form.get("label_scheme", "")
            threading.Thread(target=self.run_job, args=(job, files, names, mode, scheme), daemon=True).start()
            return self.response_json({"id": job["id"]}, 202)
        except Exception as error:
            self.failed(job, str(error))
            self.busy.release()
            raise

    def upload_slice(self, request):
        from src.slice_inference import MAX_IMAGE_BYTES, read_image
        if request.form.get('confirmed') != 'yes':
            raise ValueError('Confirm this is an individual T1-weighted brain MRI slice and the 2D model limitations are understood.')
        supplied = {key for key, value in request.files.items() if value.filename}
        if supplied not in ({'image'}, {'image', 'reference'}):
            raise ValueError('Supply one JPG/PNG MRI image and optionally one binary PNG reference mask.')
        job, error = self.new_job('single_slice')
        if error is not None:
            return error
        try:
            files, names = {}, {}
            for role in supplied:
                items = request.files.getlist(role)
                if len(items) != 1:
                    raise ValueError('Supply one file per field.')
                uploaded = items[0]
                name = uploaded.filename.replace('\\', '/').rsplit('/', 1)[-1]
                suffix = Path(name).suffix.lower()
                if suffix not in (('.png',) if role == 'reference' else ('.jpg', '.jpeg', '.png')):
                    raise ValueError('Use JPG/PNG for the MRI and lossless binary PNG for a reference mask.')
                content = uploaded.stream.read(MAX_IMAGE_BYTES + 1)
                if not 0 < len(content) <= MAX_IMAGE_BYTES:
                    raise ValueError('Each 2D image must be nonempty and at most 16 MB.')
                path = RUNS / job['id'] / f'input_{role}{suffix}'
                path.write_bytes(content)
                decoded = read_image(path, reference=role == 'reference')
                files[role], names[role] = path, name[:200]
            # Validate dimensions before starting inference; reference labels are never model inputs.
            if 'reference' in files:
                image = read_image(files['image'])
                if read_image(files['reference'], reference=True).shape != (image.height, image.width):
                    raise ValueError('Reference dimensions must match the oriented image exactly.')
            threading.Thread(target=self.run_job, args=(job, files, names, 'single_slice', ''), daemon=True).start()
            return self.response_json({'id': job['id']}, 202)
        except Exception as error:
            self.failed(job, str(error))
            self.busy.release()
            raise

    def real_example(self):
        job, error = self.new_job()
        if error is not None:
            return error
        try:
            from src.demo_case import demo_case
            files = demo_case()
            names = {key: path.name for key, path in files.items()}
            threading.Thread(target=self.run_job, args=(job, files, names, "four_dimensional", "msd"), daemon=True).start()
            return self.response_json({"id": job["id"]}, 202)
        except Exception as error:
            self.failed(job, "The existing real example is unavailable. Upload your MRI files instead.")
            self.busy.release()
            raise ValueError(f'The bundled BRATS_409 example is unavailable: {error}') from error

    def get_job(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise ValueError("Invalid run identifier.")
        if ident in self.jobs:
            with self.guard:
                return dict(self.jobs[ident])
        path = RUNS / ident / "status.json"
        if not path.is_file():
            return None
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["state"] == "running":
            job.update(state="failed", stage="Interrupted", detail="The application stopped before this run finished. Start a new prediction.")
        return job

    def dispatch(self, request):
        self.authorize(request, write=request.method == "POST")
        if request.method == "GET" and request.path == "/":
            page = (STATIC / "index.html").read_text(encoding="utf-8").replace("__UI_TOKEN__", self.token)
            response = Response(page, mimetype="text/html")
            response.set_cookie("slim_ui", self.token, httponly=True, samesite="Strict")
            return response
        if request.method == "GET" and request.path in ("/app.js", "/style.css"):
            return send_file(STATIC / request.path[1:], request.environ)
        if request.method == "GET" and request.path == "/api/status":
            from setup_2d_model import MODEL, FILES
            slice_status = dict(self.slice_model_status)
            if not all((MODEL / spec[0]).is_file() for spec in FILES[:3]):
                slice_status = {"state": "missing", "message": "Run setup_2d_model.py to install the 2D model."}
            return self.response_json({"model": self.model_status, "slice_model": slice_status, "busy": self.busy.locked(), "upload_limit_mb": 384})
        if request.method == "POST" and request.path == "/api/predict":
            return self.upload(request)
        if request.method == "POST" and request.path == "/api/real-example":
            return self.real_example()
        match = re.fullmatch(r"/api/runs/([a-f0-9]{32})(?:/(slice|download))?", request.path)
        if match and request.method == "GET":
            ident, action = match.groups()
            job = self.get_job(ident)
            if job is None:
                return self.response_json({"error": "Run not found."}, 404)
            if action is None:
                return self.response_json(job)
            if job["state"] != "completed":
                return self.response_json({"error": "This run has no completed result."}, 409)
            if action == "download":
                filename = request.args.get("file", "")
                if filename not in job["result"]["downloads"]:
                    raise ValueError("Invalid download.")
                return send_file(RUNS / ident / filename, request.environ, as_attachment=True, download_name=filename)
            if job['result'].get('input_mode') == 'single_slice':
                from src.slice_inference import render_slice
                png = render_slice(RUNS / ident, request.args.get('panel', 'prediction'), float(request.args.get('opacity', '.55')))
            else:
                from src.local_inference import render_slice
                png = render_slice(RUNS / ident, request.args.get("axis", "axial"), int(request.args.get("index", "0")),
                                   request.args.get("modality", "flair"), request.args.get("panel", "prediction"),
                                   request.args.get("region", "all"), float(request.args.get("opacity", ".55")))
            return Response(png, mimetype="image/png")
        return self.response_json({"error": "Not found."}, 404)

    def __call__(self, environ, start_response):
        request = UploadRequest(environ)
        try:
            response = self.dispatch(request)
        except PermissionError as error:
            response = self.response_json({"error": str(error)}, 403)
        except (ValueError, KeyError) as error:
            response = self.response_json({"error": str(error)}, 400)
        except HTTPException as error:
            response = self.response_json({"error": error.description}, error.code)
        except Exception:
            LOG.exception("Request failed")
            response = self.response_json({"error": "The local application encountered an error. Check its terminal log."}, 500)
        finally:
            request.close()
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
        return response(environ, start_response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = LocalApp(args.port, device=args.device)
    print(f"\nSlim UNETR local UI: http://127.0.0.1:{args.port}\nInference only. Stop with Ctrl+C.\n", flush=True)
    if args.open_browser:
        import webbrowser
        threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    run_simple("127.0.0.1", args.port, app, threaded=True, use_reloader=False, use_debugger=False)


if __name__ == "__main__":
    main()
