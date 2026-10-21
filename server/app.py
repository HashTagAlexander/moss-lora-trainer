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
import re
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
from mosstrainer import captioner  # noqa: E402
from mosstrainer.config import (DEFAULTS, FAMILY_TARGETS, MODEL_SPECS, MODELS, NAME_RE, TRAINABLE,  # noqa: E402
                                validate)

WORKSPACE = Path(os.environ.get("MOSS_WORKSPACE", ROOT / "workspace")).resolve()
DATASETS = WORKSPACE / "datasets"
JOBS = WORKSPACE / "jobs"
GENERATIONS = WORKSPACE / "generations"
INPUTS = GENERATIONS / "inputs"
LORAS = WORKSPACE / "loras"
DOWNLOADS = WORKSPACE / "downloads"
SETTINGS = WORKSPACE / "settings.json"
for d in (DATASETS, JOBS, WORKSPACE / "cache", GENERATIONS, INPUTS, LORAS, DOWNLOADS):
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


# --------------------------------------------------------------------------- settings (API keys)

_ENV_HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_settings() -> dict:
    return {"hf_token": "", "openrouter_key": "", "caption_model": captioner.DEFAULT_MODEL,
            **read_json(SETTINGS, {})}


def apply_settings(s: dict):
    """The trainer, generation worker and downloads inherit HF_TOKEN from the server's environment."""
    token = s.get("hf_token") or _ENV_HF_TOKEN
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        os.environ.pop("HF_TOKEN", None)


def mask_secret(s: str) -> str:
    return "" if not s else f"{s[:3]}…{s[-4:]}" if len(s) > 10 else "•" * len(s)


apply_settings(load_settings())


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
        if "gen_manager" in globals():
            gen_manager.offload_te()  # give the trainer room on the GPU
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


# --------------------------------------------------------------------------- generation worker

class GenManager:
    """Runs mosstrainer.generate_worker as a long-lived subprocess and feeds it one request at a time."""

    def __init__(self):
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.queue: list[dict] = []
        self.current: dict | None = None
        self.state = "unloaded"
        self.model = None
        self.message = ""
        self.last_error = ""
        self.version = 0  # bumps whenever a new image lands
        self.counter = 0

    def _spawn(self):
        log = open(GENERATIONS / "worker.log", "a")
        log.write(f"\n===== worker start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.flush()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen([sys.executable, "-m", "mosstrainer.generate_worker"], cwd=str(ROOT),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
                                     text=True, bufsize=1, start_new_session=True)
        self.state, self.model, self.message = "starting", None, "Starting worker…"
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.lock:
                kind = ev.get("event")
                if kind == "status":
                    self.state = ev.get("state", self.state)
                    self.message = ev.get("message", "")
                    if "model" in ev:
                        self.model = ev["model"]
                elif kind == "progress" and self.current and self.current["id"] == ev.get("id"):
                    self.current["progress"] = ev
                elif kind == "image":
                    self.version += 1
                    if self.current:
                        self.current.setdefault("files", []).append(ev["file"])
                elif kind == "warning":
                    self.last_error = ev.get("message", "")
                elif kind in ("done", "error", "cancelled"):
                    if kind == "error":
                        self.last_error = ev.get("message", "error")
                    self.current = None
                    self._dispatch()
        with self.lock:  # worker exited
            if self.proc is proc:
                self.proc = None
                self.state, self.model, self.message = "unloaded", None, ""
                if self.current:
                    self.last_error = "Generation worker crashed; see workspace/generations/worker.log"
                    self.current = None
                self._dispatch()

    def _dispatch(self):
        if self.current or not self.queue:
            return
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()
        req = self.queue.pop(0)
        req["low_vram"] = manager.running() is not None
        req["out_dir"] = str(GENERATIONS)
        self.current = req
        try:
            self.proc.stdin.write(json.dumps({**req, "cmd": "generate"}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.current = None
            self.queue.insert(0, req)

    def submit(self, req: dict) -> str:
        with self.lock:
            self.counter += 1
            req["id"] = f"{int(time.time()) % 100000:05d}{self.counter:03d}"
            req["queued_at"] = time.time()
            self.last_error = ""
            self.queue.append(req)
            self._dispatch()
            return req["id"]

    def cancel(self, rid: str):
        with self.lock:
            self.queue = [q for q in self.queue if q["id"] != rid]
            if self.current and self.current["id"] == rid and self.proc:
                os.kill(self.proc.pid, signal.SIGUSR1)

    def offload_te(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.stdin.write(json.dumps({"cmd": "offload"}) + "\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

    def unload(self):
        with self.lock:
            self.queue.clear()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()

    def restart_if_idle(self):
        """Stop an idle worker so the next one starts with the current environment (e.g. a new HF token)."""
        with self.lock:
            if not self.current and not self.queue and self.proc and self.proc.poll() is None:
                self.proc.terminate()

    def snapshot(self):
        with self.lock:
            brief = lambda r: {"id": r["id"], "prompt": r.get("prompt", "")[:200], "count": r.get("count", 1)}  # noqa: E731
            cur = None
            if self.current:
                cur = {**brief(self.current), "progress": self.current.get("progress"),
                       "files": self.current.get("files", [])}
            return {"state": self.state, "model": self.model, "message": self.message, "current": cur,
                    "queue": [brief(q) for q in self.queue], "last_error": self.last_error,
                    "version": self.version, "training": manager.running()}


gen_manager = GenManager()


# --------------------------------------------------------------------------- model downloads

def _tail_line(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            text = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    lines = [ln.strip() for ln in re.split(r"[\r\n]", text) if ln.strip()]
    return lines[-1][-300:] if lines else ""


class DownloadManager:
    """Pre-downloads model weights (Settings page) in a subprocess that uses the HF token from Settings."""

    def __init__(self):
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _slug(model: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", model)

    def start(self, model: str):
        with self.lock:
            p = self.procs.get(model)
            if p and p.poll() is None:
                return
            slug = self._slug(model)
            (DOWNLOADS / f"{slug}.ok").unlink(missing_ok=True)
            log = open(DOWNLOADS / f"{slug}.log", "w")
            code = "import sys; from mosstrainer.backends import prefetch; print('Done:', prefetch(sys.argv[1]))"
            self.procs[model] = subprocess.Popen(
                [sys.executable, "-c", code, model], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)

    def status(self, model: str) -> dict:
        slug = self._slug(model)
        msg = _tail_line(DOWNLOADS / f"{slug}.log")
        with self.lock:
            p = self.procs.get(model)
            if p and p.poll() is None:
                return {"state": "downloading", "message": msg}
            if p and p.returncode == 0:
                (DOWNLOADS / f"{slug}.ok").touch()
            elif p:
                return {"state": "error", "message": msg or f"exited with code {p.returncode}"}
        if (DOWNLOADS / f"{slug}.ok").exists():
            return {"state": "done", "message": ""}
        return {"state": "none", "message": ""}


downloads = DownloadManager()


# --------------------------------------------------------------------------- VLM captioning (OpenRouter)

class CaptionManager:
    """Writes captions (VGL or natural language) for a dataset's images in the background, a few at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.runs: dict[str, dict] = {}

    def status(self, dataset: str) -> dict:
        with self.lock:
            st = self.runs.get(dataset)
            return {k: v for k, v in st.items() if k != "cancel"} if st else {"state": "idle"}

    def start(self, dataset: str, files: list[str], extra: str, style: str = "vgl"):
        s = load_settings()
        if not s.get("openrouter_key"):
            raise HTTPException(400, "Add your OpenRouter API key in Settings first.")
        with self.lock:
            if self.runs.get(dataset, {}).get("state") == "running":
                raise HTTPException(409, "Captioning is already running for this dataset")
            st = {"state": "running", "total": len(files), "done": 0, "failed": 0, "errors": [], "cancel": False,
                  "style": style,
                  "model": s.get("caption_model") or captioner.DEFAULT_MODEL, "finished": [], "started": time.time()}
            self.runs[dataset] = st
        threading.Thread(target=self._run, args=(dataset, files, extra, s, st), daemon=True).start()

    def cancel(self, dataset: str):
        with self.lock:
            if dataset in self.runs:
                self.runs[dataset]["cancel"] = True

    def _run(self, dataset, files, extra, s, st):
        from concurrent.futures import ThreadPoolExecutor
        folder = DATASETS / dataset

        def one(name):
            if st["cancel"]:
                return
            try:
                cap = captioner.caption_image(folder / name, s["openrouter_key"], st["model"], extra, style=st["style"])
                (folder / name).with_suffix(".txt").write_text(captioner.caption_text(cap), encoding="utf-8")
                with self.lock:
                    st["done"] += 1
                    st["finished"].append(name)
            except Exception as e:  # noqa: BLE001
                msg = str(e) if isinstance(e, captioner.CaptionError) else f"{type(e).__name__}: {e}"
                with self.lock:
                    st["failed"] += 1
                    st["errors"] = (st["errors"] + [f"{name}: {msg}"])[-20:]
                    if any(f"HTTP {c}" in msg for c in (401, 402, 403)):
                        st["cancel"] = True  # bad key or no credits: stop instead of failing every image
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(one, files))
        with self.lock:
            st["state"] = "cancelled" if st["cancel"] and st["done"] + st["failed"] < st["total"] else "done"
            st["ended"] = time.time()


captions = CaptionManager()


# --------------------------------------------------------------------------- system

@app.get("/api/info")
def info():
    specs = {k: {"label": v["label"], "family": v["family"], "trainable": bool(v.get("trainable")),
                 "gated": bool(v.get("gated")), "edit": bool(v.get("edit")), "defaults": v.get("defaults", {}),
                 "overrides": v.get("overrides", {})} for k, v in MODEL_SPECS.items()}
    return {"defaults": DEFAULTS, "models": MODELS, "trainable": TRAINABLE, "model_specs": specs,
            "families": FAMILY_TARGETS, "workspace": str(WORKSPACE), "auth": bool(PASSWORD)}


# --------------------------------------------------------------------------- settings

@app.get("/api/settings")
def get_settings():
    s = load_settings()
    return {"hf_token": mask_secret(s["hf_token"]), "has_hf_token": bool(s["hf_token"]),
            "env_hf_token": bool(_ENV_HF_TOKEN), "openrouter_key": mask_secret(s["openrouter_key"]),
            "has_openrouter_key": bool(s["openrouter_key"]), "caption_model": s["caption_model"],
            "caption_model_default": captioner.DEFAULT_MODEL}


@app.put("/api/settings")
async def put_settings(request: Request):
    body = await request.json()
    s = load_settings()
    old_token = s["hf_token"]
    for key in ("hf_token", "openrouter_key", "caption_model"):
        if isinstance(body.get(key), str):  # absent = keep; "" = clear
            s[key] = body[key].strip()
    s["caption_model"] = s["caption_model"] or captioner.DEFAULT_MODEL
    tmp = SETTINGS.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:  # owner-only: holds keys
        json.dump(s, f, indent=2)
    os.replace(tmp, SETTINGS)
    apply_settings(s)
    if s["hf_token"] != old_token:
        gen_manager.restart_if_idle()
    return get_settings()


@app.post("/api/settings/check-hf")
def check_hf():
    """Who the token belongs to, and whether it can download each gated model."""
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    out = {"user": None, "error": None, "access": {}}
    try:
        out["user"] = whoami().get("name")
    except Exception as e:  # noqa: BLE001
        out["error"] = "No valid Hugging Face token" if "token" in str(e).lower() else str(e)[:200]
    for mid, spec in MODEL_SPECS.items():
        if not spec.get("gated"):
            continue
        repo = spec.get("repo", mid)
        try:
            auth_check(repo)
            out["access"][repo] = "ok"
        except GatedRepoError:
            out["access"][repo] = "gated"
        except RepositoryNotFoundError:
            out["access"][repo] = "no_access"
        except Exception as e:  # noqa: BLE001
            out["access"][repo] = str(e)[:200]
    return out


@app.post("/api/settings/check-openrouter")
def check_openrouter():
    key = load_settings()["openrouter_key"]
    if not key:
        raise HTTPException(400, "No OpenRouter key saved")
    try:
        return captioner.check_key(key)
    except captioner.CaptionError as e:
        raise HTTPException(400, str(e))


@app.get("/api/models")
def list_models():
    return [{"id": k, "label": v["label"], "family": v["family"], "trainable": bool(v.get("trainable")),
             "gated": bool(v.get("gated")), "repo": v.get("repo", k) if "/" in v.get("repo", k) else None,
             "download": downloads.status(k)} for k, v in MODEL_SPECS.items()]


@app.post("/api/models/download")
async def download_model(request: Request):
    model = (await request.json()).get("model")
    if model not in MODEL_SPECS:
        raise HTTPException(400, "Unknown model")
    downloads.start(model)
    return downloads.status(model)


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
        elif ext == ".json":  # a structured (VGL) caption, e.g. from Bria's tools: stored as the image's .txt
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            if isinstance(obj, dict):
                (folder / _clean_name(Path(filename).stem + ".txt")).write_text(
                    captioner.caption_text(obj), encoding="utf-8")
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


@app.post("/api/datasets/{name}/autocaption")
async def autocaption(name: str, request: Request):
    """Caption images with a VLM. scope: missing | all | files (with "files"); style: vgl (Fibo 2.0) | natural."""
    folder = DATASETS / safe_name(name)
    if not folder.is_dir():
        raise HTTPException(404)
    body = await request.json()
    scope = body.get("scope", "missing")
    images = dataset_images(folder)
    if scope == "files":
        wanted = set(body.get("files") or [])
        files = [p.name for p in images if p.name in wanted]
    elif scope == "all":
        files = [p.name for p in images]
    else:
        files = [p.name for p in images
                 if not (p.with_suffix(".txt").exists() and p.with_suffix(".txt").read_text(encoding="utf-8").strip())]
    if not files:
        raise HTTPException(400, "No images to caption")
    style = body.get("style", "vgl")
    if style not in captioner.STYLES:
        raise HTTPException(400, f"Unknown caption style: {style}")
    captions.start(name, files, str(body.get("extra", ""))[:2000], style)
    return captions.status(name)


@app.get("/api/datasets/{name}/autocaption")
def autocaption_status(name: str):
    return captions.status(safe_name(name))


@app.post("/api/datasets/{name}/autocaption/cancel")
def autocaption_cancel(name: str):
    captions.cancel(safe_name(name))
    return captions.status(name)


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



# --------------------------------------------------------------------------- generation

def lora_inventory():
    out = []
    for d in sorted(JOBS.iterdir()):
        ck = d / "checkpoints"
        if not ck.is_dir():
            continue
        cfg = read_json(d / "config.json", {})
        for p in sorted(ck.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
            stem = p.stem
            step = stem.rsplit("_", 1)[-1] if stem != cfg.get("name") else str(cfg.get("train", {}).get("steps", ""))
            family = MODEL_SPECS.get(cfg.get("model"), {}).get("family")
            out.append({"id": f"jobs/{d.name}/checkpoints/{p.name}", "file": p.name, "group": d.name, "family": family,
                        "label": f"{d.name} · {'final' if stem == cfg.get('name') else 'step ' + step.lstrip('0')}",
                        "trigger": cfg.get("trigger_word", ""), "model": cfg.get("model"),
                        "mtime": p.stat().st_mtime, "size": p.stat().st_size})
    for p in sorted(LORAS.glob("*.safetensors")):
        out.append({"id": f"loras/{p.name}", "file": p.name, "group": "Uploaded", "label": p.stem, "trigger": "", "family": None,
                    "model": None, "mtime": p.stat().st_mtime, "size": p.stat().st_size})
    return out


def resolve_lora(lora_id: str) -> Path:
    path = (WORKSPACE / lora_id).resolve()
    ok_roots = (JOBS.resolve(), LORAS.resolve())
    if path.suffix != ".safetensors" or not path.is_file() or not any(path.is_relative_to(r) for r in ok_roots):
        raise HTTPException(400, f"Unknown LoRA: {lora_id}")
    return path


@app.get("/api/loras")
def list_loras():
    return lora_inventory()


@app.post("/api/loras/upload")
async def upload_lora(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        name = _clean_name(f.filename or "lora.safetensors")
        if not name.endswith(".safetensors"):
            raise HTTPException(400, "Only .safetensors LoRA files are supported")
        target = _unique(LORAS, name)
        with open(target, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        added.append(target.name)
    return {"added": added}


@app.delete("/api/loras/uploaded/{filename}")
def delete_uploaded_lora(filename: str):
    safe_file(LORAS, filename).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(400, "Write a prompt first")
    model = body.get("model") or DEFAULTS["model"]
    if model not in MODEL_SPECS:
        raise HTTPException(400, "Unknown model")
    d = MODEL_SPECS[model].get("defaults", {})
    try:
        width, height = int(body.get("width", 1024)), int(body.get("height", 1024))
        steps, count = int(body.get("steps", d.get("steps", 30))), int(body.get("count", 1))
        cfg, shift = float(body.get("cfg", d.get("cfg", 3.0))), float(body.get("shift", d.get("shift", 0.0)))
        seed, skip = int(body.get("seed", -1)), int(body.get("skip_layer", d.get("skip_layer", -1)))
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid number in settings")
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise HTTPException(400, "Width and height must be between 256 and 2048")
    if not 1 <= steps <= 150 or not 1 <= count <= 8:
        raise HTTPException(400, "Steps must be 1-150 and count 1-8")
    loras = []
    for lr in body.get("loras", []) or []:
        path = resolve_lora(str(lr.get("id", "")))
        loras.append({"id": lr["id"], "path": str(path), "name": lr.get("name") or path.stem,
                      "strength": max(-4.0, min(4.0, float(lr.get("strength", 1.0))))})
    req = {"model": model, "prompt": prompt, "negative": str(body.get("negative", "")), "width": width,
           "height": height, "steps": steps, "cfg": cfg, "shift": shift, "seed": seed, "count": count,
           "skip_layer": skip, "loras": loras}
    if body.get("image"):  # a source image to edit
        if not MODEL_SPECS[model].get("edit"):
            raise HTTPException(400, "This model can't edit images; remove the source image")
        src = safe_file(INPUTS, str(body["image"]))
        if not src.exists():
            raise HTTPException(400, "The source image is gone; upload it again")
        req.update(image=str(src), image_id=src.name)
    return {"id": gen_manager.submit(req)}


@app.post("/api/generate/input")
async def upload_input(file: UploadFile = File(...)):
    """A source image for editing (Fibo 2.0)."""
    name = _clean_name(file.filename or "image.png")
    if Path(name).suffix not in IMAGE_EXTS:
        raise HTTPException(400, "Upload an image")
    target = _unique(INPUTS, name)
    target.write_bytes(await file.read())
    return {"id": target.name}


@app.get("/api/generations/input/{filename}")
def generation_input(filename: str):
    path = safe_file(INPUTS, filename)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=31536000, immutable"})


@app.get("/api/generate/status")
def generate_status():
    return gen_manager.snapshot()


@app.post("/api/generate/cancel/{rid}")
def generate_cancel(rid: str):
    gen_manager.cancel(rid)
    return {"ok": True}


@app.post("/api/generate/unload")
def generate_unload():
    gen_manager.unload()
    return {"ok": True}


@app.get("/api/generations")
def list_generations(limit: int = 120, offset: int = 0):
    metas = sorted(GENERATIONS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for m in metas[offset:offset + limit]:
        meta = read_json(m, {})
        out.append({**meta, "file": m.with_suffix(".png").name, "thumb": m.with_suffix(".thumb.jpg").name})
    return {"items": out, "total": len(metas)}


@app.get("/api/generations/file/{filename}")
def generation_file(filename: str):
    path = safe_file(GENERATIONS, filename)
    if not path.exists() or path.suffix not in (".png", ".jpg"):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=31536000, immutable"})


@app.delete("/api/generations/{filename}")
def delete_generation(filename: str):
    png = safe_file(GENERATIONS, filename)
    stem = png.name[:-len(".png")] if png.name.endswith(".png") else png.stem
    for suffix in (".png", ".thumb.jpg", ".json"):
        (GENERATIONS / f"{stem}{suffix}").unlink(missing_ok=True)
    return {"ok": True}


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
