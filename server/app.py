"""Web UI + REST API for the MOSS-Image LoRA trainer.

    python -m server.app --host 0.0.0.0 --port 9090

Environment:
    MOSS_WORKSPACE         data root (datasets/, jobs/, cache/); default ./workspace
    MOSS_TRAINER_PASSWORD  if set, HTTP basic auth is required (user: admin)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mosstrainer.config import DEFAULTS, MODELS, NAME_RE, validate  # noqa: E402

WORKSPACE = Path(os.environ.get("MOSS_WORKSPACE", ROOT / "workspace")).resolve()
DATASETS = WORKSPACE / "datasets"
JOBS = WORKSPACE / "jobs"
for d in (DATASETS, JOBS, WORKSPACE / "cache"):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ACTIVE = {"starting", "loading", "caching_text", "caching_latents", "training", "sampling", "saving"}
PASSWORD = os.environ.get("MOSS_TRAINER_PASSWORD", "")

app = FastAPI(title="MOSS-Image LoRA Trainer")


# --------------------------------------------------------------------------- auth

@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if PASSWORD:
        header = request.headers.get("authorization", "")
        ok = False
        if header.lower().startswith("basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="MOSS Trainer"'})
    return await call_next(request)


# --------------------------------------------------------------------------- helpers

def safe_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise HTTPException(400, "Invalid name")
    return name


def safe_file(folder: Path, filename: str) -> Path:
    path = (folder / filename).resolve()
    if path.parent != folder.resolve() or filename.startswith("."):
        raise HTTPException(400, "Invalid file name")
    return path


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- job manager

class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.procs: dict[str, subprocess.Popen] = {}
        self._recover()
        threading.Thread(target=self._loop, daemon=True).start()

    def status(self, name):
        return read_json(JOBS / name / "status.json", {"state": "new"})

    def set_status(self, name, **kw):
        st = self.status(name)
        st.update(kw, updated_at=time.time())
        write_json(JOBS / name / "status.json", st)
        return st

    def _recover(self):
        for d in JOBS.iterdir():
            st = read_json(d / "status.json")
            if st and st.get("state") in ACTIVE and not pid_alive(st.get("pid")):
                self.set_status(d.name, state="error", message="Trainer process ended unexpectedly (server restart?)")

    def running(self):
        for d in JOBS.iterdir():
            st = read_json(d / "status.json")
            if st and st.get("state") in ACTIVE and pid_alive(st.get("pid")):
                return d.name
        for name, p in self.procs.items():
            if p.poll() is None:
                return name
        return None

    def _launch(self, name):
        job_dir = JOBS / name
        log = open(job_dir / "log.txt", "a")
        log.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.flush()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen([sys.executable, "-m", "mosstrainer.train", "--job-dir", str(job_dir)],
                                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        self.procs[name] = proc
        self.set_status(name, state="starting", pid=proc.pid, message="Starting trainer", step=self.status(name).get("step", 0))

    def _loop(self):
        while True:
            try:
                with self.lock:
                    for name, p in list(self.procs.items()):
                        code = p.poll()
                        if code is None:
                            continue
                        del self.procs[name]
                        st = self.status(name)
                        if st.get("state") in ACTIVE:
                            self.set_status(name, state="error", message=f"Trainer exited with code {code}; see log")
                    if not self.running():
                        queued = []
                        for d in JOBS.iterdir():
                            st = read_json(d / "status.json")
                            if st and st.get("state") == "queued":
                                queued.append((st.get("queued_at", 0), d.name))
                        if queued:
                            self._launch(sorted(queued)[0][1])
            except Exception as e:  # noqa: BLE001
                print("job loop error:", e, flush=True)
            time.sleep(2)

    def start(self, name):
        with self.lock:
            st = self.status(name)
            if st.get("state") in ACTIVE or st.get("state") == "queued":
                return st
            return self.set_status(name, state="queued", queued_at=time.time(), message="Waiting for GPU")

    def stop(self, name, force=False):
        with self.lock:
            st = self.status(name)
            if st.get("state") == "queued":
                return self.set_status(name, state="stopped", message="Removed from queue")
            pid = st.get("pid")
            if st.get("state") in ACTIVE and pid_alive(pid):
                try:
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL if force else signal.SIGTERM)
                except OSError:
                    os.kill(int(pid), signal.SIGKILL if force else signal.SIGTERM)
                if force:
                    return self.set_status(name, state="stopped", message="Killed")
                return self.set_status(name, message="Stopping after current step...")
            return st


manager = JobManager()


# --------------------------------------------------------------------------- system

@app.get("/api/info")
def info():
    return {"defaults": DEFAULTS, "models": MODELS, "workspace": str(WORKSPACE), "auth": bool(PASSWORD)}


@app.get("/api/gpu")
def gpu():
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    disk = shutil.disk_usage(WORKSPACE)
    disk = {"free": disk.free, "total": disk.total}
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"gpus": [], "disk": disk}
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        gpus.append({"index": int(parts[0]), "name": parts[1], "util": num(parts[2]), "mem_used": num(parts[3]),
                     "mem_total": num(parts[4]), "temp": num(parts[5]), "power": num(parts[6]),
                     "power_limit": num(parts[7])})
    return {"gpus": gpus, "disk": disk}


# --------------------------------------------------------------------------- datasets

def dataset_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


@app.get("/api/datasets")
def list_datasets():
    out = []
    for d in sorted(DATASETS.iterdir()):
        if not d.is_dir():
            continue
        imgs = dataset_images(d)
        captioned = sum(1 for p in imgs if p.with_suffix(".txt").exists())
        out.append({"name": d.name, "count": len(imgs), "captioned": captioned,
                    "cover": imgs[0].name if imgs else None})
    return out


@app.post("/api/datasets")
async def create_dataset(request: Request):
    body = await request.json()
    name = safe_name(body.get("name", ""))
    path = DATASETS / name
    if path.exists():
        raise HTTPException(409, "Dataset already exists")
    path.mkdir()
    return {"name": name}


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    path = DATASETS / safe_name(name)
    if not path.is_dir():
        raise HTTPException(404)
    shutil.rmtree(path)
    return {"ok": True}


@app.get("/api/datasets/{name}")
def get_dataset(name: str):
    path = DATASETS / safe_name(name)
    if not path.is_dir():
        raise HTTPException(404)
    items = []
    for p in dataset_images(path):
        txt = p.with_suffix(".txt")
        items.append({"file": p.name, "caption": txt.read_text(encoding="utf-8") if txt.exists() else "",
                      "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)})
    return {"name": name, "items": items}


def _unique(folder: Path, filename: str) -> Path:
    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem).strip().replace(" ", "_") or "image"
    target = folder / f"{stem}{suffix}"
    i = 1
    while target.exists():
        target = folder / f"{stem}_{i}{suffix}"
        i += 1
    return target


def _clean_name(filename: str) -> str:
    stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in Path(filename).stem).strip().replace(" ", "_")
    return (stem or "image") + Path(filename).suffix.lower()


@app.post("/api/datasets/{name}/upload")
async def upload(name: str, files: list[UploadFile] = File(...)):
    folder = DATASETS / safe_name(name)
    if not folder.is_dir():
        raise HTTPException(404)
    added, captions = 0, 0

    def store(filename: str, data: bytes):
        nonlocal added, captions
        filename = Path(filename).name
        if filename.startswith("."):
            return
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTS:
            _unique(folder, filename).write_bytes(data)
            added += 1
        elif ext == ".txt":
            (folder / _clean_name(filename)).write_bytes(data)
            captions += 1

    for f in files:
        data = await f.read()
        if (f.filename or "").lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if not info.is_dir() and "__MACOSX" not in info.filename:
                        store(info.filename, z.read(info))
        else:
            store(f.filename or "upload", data)
    return {"images": added, "captions": captions}


@app.put("/api/datasets/{name}/caption/{filename}")
async def set_caption(name: str, filename: str, request: Request):
    folder = DATASETS / safe_name(name)
    img = safe_file(folder, filename)
    if not img.exists():
        raise HTTPException(404)
    body = await request.json()
    img.with_suffix(".txt").write_text(body.get("caption", "").strip(), encoding="utf-8")
    return {"ok": True}


@app.post("/api/datasets/{name}/bulk")
async def bulk_caption(name: str, request: Request):
    folder = DATASETS / safe_name(name)
    body = await request.json()
    action, text, find = body.get("action"), body.get("text", ""), body.get("find", "")
    changed = 0
    for img in dataset_images(folder):
        txt = img.with_suffix(".txt")
        cap = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        if action == "prepend":
            new = f"{text}{cap}"
        elif action == "append":
            new = f"{cap}{text}"
        elif action == "replace" and find:
            new = cap.replace(find, text)
        elif action == "fill_empty":
            new = cap or text
        else:
            raise HTTPException(400, "Unknown action")
        if new != cap:
            txt.write_text(new.strip(), encoding="utf-8")
            changed += 1
    return {"changed": changed}


@app.delete("/api/datasets/{name}/item/{filename}")
def delete_item(name: str, filename: str):
    folder = DATASETS / safe_name(name)
    img = safe_file(folder, filename)
    img.unlink(missing_ok=True)
    img.with_suffix(".txt").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/datasets/{name}/image/{filename}")
def dataset_image(name: str, filename: str, thumb: int = 0):
    folder = DATASETS / safe_name(name)
    img = safe_file(folder, filename)
    if not img.exists():
        raise HTTPException(404)
    if not thumb:
        return FileResponse(img)
    thumbs = folder / ".thumbs"
    thumbs.mkdir(exist_ok=True)
    t = thumbs / f"{img.name}.{int(img.stat().st_mtime)}.jpg"
    if not t.exists():
        from PIL import Image, ImageOps
        with Image.open(img) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((448, 448))
            im.save(t, quality=85)
    return FileResponse(t, headers={"Cache-Control": "max-age=86400"})


# --------------------------------------------------------------------------- jobs

def job_summary(d: Path):
    cfg = read_json(d / "config.json", {})
    st = manager.status(d.name)
    return {"name": d.name, "dataset": cfg.get("dataset"), "model": cfg.get("model"),
            "steps": cfg.get("train", {}).get("steps"), "status": st,
            "created": (d / "config.json").stat().st_mtime if (d / "config.json").exists() else 0}


@app.get("/api/jobs")
def list_jobs():
    jobs = [job_summary(d) for d in JOBS.iterdir() if d.is_dir()]
    return sorted(jobs, key=lambda j: -j["created"])


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    start = body.pop("_start", False)
    overwrite = body.pop("_overwrite", False)
    try:
        cfg = validate(body)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(400, str(e))
    if not (DATASETS / cfg["dataset"]).is_dir():
        raise HTTPException(400, f"Dataset '{cfg['dataset']}' does not exist")
    job_dir = JOBS / cfg["name"]
    if job_dir.exists():
        if not overwrite:
            raise HTTPException(409, "A job with this name already exists")
        if manager.status(cfg["name"]).get("state") in ACTIVE | {"queued"}:
            raise HTTPException(409, "Job is running; stop it first")
    job_dir.mkdir(exist_ok=True)
    write_json(job_dir / "config.json", cfg)
    if not (job_dir / "status.json").exists():
        write_json(job_dir / "status.json", {"state": "new", "step": 0, "total_steps": cfg["train"]["steps"]})
    else:
        manager.set_status(cfg["name"], total_steps=cfg["train"]["steps"])
    if start:
        manager.start(cfg["name"])
    return {"name": cfg["name"]}


def job_dir_or_404(name):
    d = JOBS / safe_name(name)
    if not d.is_dir():
        raise HTTPException(404, "Job not found")
    return d


@app.get("/api/jobs/{name}")
def get_job(name: str):
    d = job_dir_or_404(name)
    ckpts = []
    for p in sorted((d / "checkpoints").glob("*.safetensors")) if (d / "checkpoints").exists() else []:
        ckpts.append({"file": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime})
    return {"name": name, "config": read_json(d / "config.json", {}), "status": manager.status(name),
            "checkpoints": ckpts, "samples": read_json(d / "samples" / "index.json", []),
            "resumable": (d / "checkpoints" / "training_state.pt").exists()}


@app.post("/api/jobs/{name}/start")
def start_job(name: str):
    job_dir_or_404(name)
    return manager.start(name)


@app.post("/api/jobs/{name}/stop")
def stop_job(name: str, force: int = 0):
    job_dir_or_404(name)
    return manager.stop(name, bool(force))


@app.post("/api/jobs/{name}/reset")
def reset_job(name: str):
    d = job_dir_or_404(name)
    if manager.status(name).get("state") in ACTIVE | {"queued"}:
        raise HTTPException(409, "Stop the job first")
    for sub in ("checkpoints", "samples"):
        shutil.rmtree(d / sub, ignore_errors=True)
    for f in ("metrics.jsonl", "log.txt"):
        (d / f).unlink(missing_ok=True)
    cfg = read_json(d / "config.json", {})
    write_json(d / "status.json", {"state": "new", "step": 0, "total_steps": cfg.get("train", {}).get("steps")})
    return {"ok": True}


@app.delete("/api/jobs/{name}")
def delete_job(name: str):
    d = job_dir_or_404(name)
    if manager.status(name).get("state") in ACTIVE:
        raise HTTPException(409, "Stop the job first")
    shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/jobs/{name}/metrics")
def job_metrics(name: str, since: int = 0):
    d = job_dir_or_404(name)
    rows = {}
    path = d / "metrics.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("step", 0) > since:
                    rows[r["step"]] = r  # later lines win (resumed runs)
    return [rows[k] for k in sorted(rows)]


@app.get("/api/jobs/{name}/log", response_class=PlainTextResponse)
def job_log(name: str, lines: int = 300):
    d = job_dir_or_404(name)
    path = d / "log.txt"
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 256 * 1024))
        data = f.read().decode("utf-8", "replace")
    # tqdm-style carriage returns: keep only the last segment of each line
    out = [ln.split("\r")[-1] for ln in data.splitlines()]
    return "\n".join(out[-lines:])


@app.get("/api/jobs/{name}/sample/{filename}")
def job_sample(name: str, filename: str):
    d = job_dir_or_404(name)
    path = safe_file(d / "samples", filename)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=31536000, immutable"})


@app.get("/api/jobs/{name}/checkpoint/{filename}")
def job_checkpoint(name: str, filename: str):
    d = job_dir_or_404(name)
    path = safe_file(d / "checkpoints", filename)
    if not path.exists() or path.suffix != ".safetensors":
        raise HTTPException(404)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.exception_handler(HTTPException)
async def http_error(request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, exc.status_code, headers=exc.headers)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9090)
    args = ap.parse_args()
    print(f"MOSS-Image LoRA Trainer on http://{args.host}:{args.port}  (workspace: {WORKSPACE})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
