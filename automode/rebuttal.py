"""
AutoMode rebuttal runner, API 1.0 (see RUNNER_CONTRACT.md).

    python -m automode.rebuttal contract
    python -m automode.rebuttal preflight --manifest FILE --worker-id N --output DIR
    python -m automode.rebuttal pilot     --manifest FILE --worker-id N --output DIR
    python -m automode.rebuttal run       --manifest FILE --worker-id N --output DIR --hours H

Every command exits non-zero on failure. `contract` prints one JSON object and
imports nothing heavy. Output layout under DIR:

    source_identity.json        commit / archive / manifest identity of this DIR
    preflight_report.json       resolved_manifest.json  environment.json  preflight_pytest.log
    pilot_report.json           pilot/{method_checks/<method>,main}/...
    run_ledger.json             campaign_status.json
    core/<job_id>/attempts/attempt_NNN/{spec.json,status.json,job.log,result.json,logs,dynamic,evals,export}
    core/<job_id>/COMPLETE.json written last, only after verified success

Scientific identity (job_identity) = every outcome-relevant RunConfig field,
resolved model/dataset revisions, protocol version, commit, code hash,
extraction/target-format versions and the prompt template. Output location is
not part of it. A run is skipped only if COMPLETE.json carries exactly that
identity and its artefacts verify. Failed/interrupted attempts are kept;
every new invocation of `run` starts a fresh attempt directory for jobs that
are not verified complete (that re-invocation is the explicit retry).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REBUTTAL_API_VERSION = "1.0"
METRIC_KEY = "eval_gsm8k_maj3"
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "requirements-rebuttal.lock"
CORE_SEEDS = (8, 25, 42)
N_WORKERS = 4
PILOT_TRAIN_SAMPLES = 512
PILOT_EVAL_SAMPLES = 100
PILOT_DYNAMIC_UPDATES = 2
MIN_POST_PROMOTION_STEPS = 5
EXPORT_DIAGNOSTIC_PROMPTS = 32
ETA_SAFETY_FACTOR = 1.25
GIB = 2 ** 30


class RebuttalError(RuntimeError):
    pass


class OutputIdentityError(RebuttalError):
    """The output DIR belongs to another source/manifest: never write into it."""


class JobInterrupted(BaseException):
    """Raised inside a job by SIGTERM/SIGINT; BaseException so broad handlers can't eat it."""


# ──────────────────────────────────────────────────────────────────────────────
# Small stdlib helpers
# ──────────────────────────────────────────────────────────────────────────────

def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


@contextmanager
def file_lock(path: Path, what: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RebuttalError(f"{what} is locked by another process ({path})")
        yield


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", file=sys.stderr, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> Tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def job_config(manifest: dict, job: dict, revisions: Optional[dict] = None, **extra):
    """RunConfig = common + job overrides + protocol version (+ resolved revisions)."""
    from automode.config import RunConfig
    kw = {**manifest["common"], **job["overrides"], "protocol_version": manifest["protocol_version"],
          "device": "cuda:0", "output_root": "unused"}
    if revisions:
        kw.update(revisions)
    kw.update(extra)
    return RunConfig(**kw)


def validate_manifest(m: dict) -> List[dict]:
    """Schema-1 checks; raises RebuttalError listing every problem. Returns jobs."""
    from automode.config import RunConfig, CORE_METHODS
    errors: List[str] = []
    fields = set(RunConfig.__dataclass_fields__) - {"output_root", "device", "protocol_version"}
    if m.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for key, typ in (("protocol_version", str), ("expected_eval_count", int), ("common", dict), ("jobs", list)):
        if not isinstance(m.get(key), typ):
            errors.append(f"{key} must be {typ.__name__}")
    if not isinstance(m.get("minimum_vram_gib"), (int, float)) or m.get("minimum_vram_gib", 0) <= 0:
        errors.append("minimum_vram_gib must be a positive number")
    unknown_top = set(m) - {"schema_version", "protocol_version", "expected_eval_count", "minimum_vram_gib",
                            "common", "jobs"}
    if unknown_top:
        errors.append(f"unknown top-level keys {sorted(unknown_top)}")
    if errors:
        raise RebuttalError("; ".join(errors))
    common = m["common"]
    if set(common) - fields:
        errors.append(f"unknown common keys {sorted(set(common) - fields)}")
    if common.get("max_eval_samples") is not None or common.get("max_train_samples") is not None:
        errors.append("core runs must use the full train set and all test items (max_*_samples must be null)")
    if "gsm8k" not in common.get("eval_benchmarks", []):
        errors.append("eval_benchmarks must include gsm8k")
    ids, cells = set(), {}
    for j in m["jobs"]:
        jid = j.get("job_id")
        if not isinstance(jid, str) or not re.fullmatch(r"[A-Za-z0-9_]+", jid):
            errors.append(f"bad job_id {jid!r}")
        if jid in ids:
            errors.append(f"duplicate job_id {jid}")
        ids.add(jid)
        if j.get("worker_id") not in range(N_WORKERS):
            errors.append(f"{jid}: worker_id must be 0..{N_WORKERS - 1}")
        ov = j.get("overrides", {})
        if set(ov) - fields:
            errors.append(f"{jid}: unknown override keys {sorted(set(ov) - fields)}")
        if {"max_eval_samples", "max_train_samples"} & set(ov):
            errors.append(f"{jid}: overrides may not limit train/eval samples")
        if ov.get("method") not in CORE_METHODS:
            errors.append(f"{jid}: method {ov.get('method')!r} not in {CORE_METHODS}")
        cell = (ov.get("method"), ov.get("seed"))
        if cell in cells:
            errors.append(f"{jid}: duplicates (method, seed) of {cells[cell]}")
        cells[cell] = jid
    expected = {(mm, s) for mm in CORE_METHODS for s in CORE_SEEDS}
    if set(cells) != expected:
        errors.append(f"jobs must cover {sorted(CORE_METHODS)} x seeds {CORE_SEEDS} exactly once; "
                      f"missing {sorted(expected - set(cells), key=str)}, extra {sorted(set(cells) - expected, key=str)}")
    for w in range(N_WORKERS):
        if not any(j.get("worker_id") == w for j in m["jobs"]):
            errors.append(f"worker {w} has no jobs")
    if errors:
        raise RebuttalError("; ".join(errors))
    run_ids = {}
    for j in m["jobs"]:
        try:
            rid = job_config(m, j).run_id()
        except (TypeError, ValueError) as e:
            errors.append(f"{j['job_id']}: invalid configuration: {e}")
            continue
        if rid in run_ids:
            errors.append(f"{j['job_id']} has the same configuration identity as {run_ids[rid]}")
        run_ids[rid] = j["job_id"]
    if errors:
        raise RebuttalError("; ".join(errors))
    return m["jobs"]


def worker_jobs(manifest: dict, worker_id: int) -> List[dict]:
    jobs = [j for j in manifest["jobs"] if j["worker_id"] == worker_id]
    if not jobs:
        raise RebuttalError(f"worker {worker_id} has no jobs")
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Source / code identity
# ──────────────────────────────────────────────────────────────────────────────

def code_sha256() -> str:
    h = hashlib.sha256()
    files = sorted(list((REPO_ROOT / "automode").glob("*.py")) + [LOCK_FILE, REPO_ROOT / "pyproject.toml"])
    for f in files:
        h.update(f.relative_to(REPO_ROOT).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def source_identity(manifest_sha: str) -> dict:
    meta_path = Path(os.environ.get("AUTOMODE_SOURCE_METADATA") or REPO_ROOT.parent / "source_metadata.json")
    if not meta_path.exists():
        raise RebuttalError(f"source metadata not found at {meta_path} (set AUTOMODE_SOURCE_METADATA)")
    meta = read_json(meta_path)
    commit = str(meta.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RebuttalError("source metadata commit must be a 40-hex SHA")
    env_commit = os.environ.get("AUTOMODE_CODE_COMMIT")
    if env_commit and env_commit != commit:
        raise RebuttalError(f"AUTOMODE_CODE_COMMIT {env_commit} != source metadata commit {commit}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("archive_sha256", ""))):
        raise RebuttalError("source metadata archive_sha256 missing or malformed")
    if meta.get("manifest_sha256") != manifest_sha:
        raise RebuttalError(f"manifest sha256 {manifest_sha} != source metadata {meta.get('manifest_sha256')}")
    return {"commit": commit, "archive_sha256": meta["archive_sha256"], "manifest_sha256": manifest_sha,
            "code_sha256": code_sha256(), "dependency_lock_sha256": sha256_bytes(LOCK_FILE.read_bytes())}


IDENTITY_KEYS = ("commit", "archive_sha256", "manifest_sha256")


def bind_output_identity(out: Path, src: dict) -> None:
    """Refuse to reuse an output DIR created for different source/manifest."""
    p = out / "source_identity.json"
    if p.exists():
        old = read_json(p)
        diff = [k for k in IDENTITY_KEYS + ("code_sha256",) if old.get(k) != src.get(k)]
        if diff:
            raise OutputIdentityError(f"{out} belongs to a different source/manifest (differs in {diff}); "
                                f"use a fresh output directory")
    else:
        write_json_atomic(p, {**src, "created_at": utcnow()})


def job_identity(cfg, src: dict) -> str:
    from automode.data import GSM8K_PROMPT_TEMPLATE, TARGET_FORMAT_VERSION
    from automode.eval import GSM8K_EXTRACTION_VERSION
    payload = {
        "api_version": REBUTTAL_API_VERSION,
        "protocol_version": cfg.protocol_version,
        "config": cfg.scientific_dict(),
        "commit": src["commit"],
        "code_sha256": src["code_sha256"],
        "gsm8k_extraction_version": GSM8K_EXTRACTION_VERSION,
        "target_format_version": TARGET_FORMAT_VERSION,
        "prompt_template_sha256": sha256_bytes(GSM8K_PROMPT_TEMPLATE.encode()),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode())


# ──────────────────────────────────────────────────────────────────────────────
# Environment, dependency lock, hardware
# ──────────────────────────────────────────────────────────────────────────────

def parse_lock(path: Path = LOCK_FILE) -> Dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        name, sep, ver = line.partition("==")
        if not sep or not ver:
            raise RebuttalError(f"unpinned lock line: {line!r}")
        pins[name.strip().lower()] = ver.strip()
    return pins


def verify_installed_lock() -> Dict[str, Any]:
    from importlib import metadata
    mismatches = {}
    for name, ver in parse_lock().items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        if got != ver:
            mismatches[name] = {"locked": ver, "installed": got}
    return mismatches


def environment_record() -> dict:
    """Versions and nondeterminism settings. Deliberately no environment dump."""
    import platform
    from importlib import metadata
    import torch
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "packages": {n: metadata.version(n) for n in sorted(parse_lock())},
        "nondeterminism": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG_set": "CUBLAS_WORKSPACE_CONFIG" in os.environ,
            "note": "GPU kernels are not forced deterministic; equal seeds need not give bitwise-equal runs.",
        },
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        rec["gpu"] = gpu_info()
    return rec


def gpu_info() -> dict:
    import torch
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    major, minor = torch.cuda.get_device_capability(0)
    info = {"gpu_uuid": uuid if uuid.startswith("GPU-") else f"GPU-{uuid}", "gpu_name": props.name,
            "vram_gib": props.total_memory / GIB, "compute_capability": f"sm_{major}{minor}",
            "torch_arch_list": torch.cuda.get_arch_list(), "visible_device_count": torch.cuda.device_count()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        for line in q.stdout.splitlines():
            u, drv = [x.strip() for x in line.split(",")]
            if u == info["gpu_uuid"]:
                info["driver_version"] = drv
    except (OSError, subprocess.SubprocessError, ValueError):
        info["driver_version"] = None
    return info


def check_gpu(min_vram_gib: float) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RebuttalError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RebuttalError(f"expected exactly one visible GPU (CUDA_VISIBLE_DEVICES), saw {torch.cuda.device_count()}")
    info = gpu_info()
    if info["vram_gib"] < min_vram_gib:
        raise RebuttalError(f"GPU has {info['vram_gib']:.1f} GiB < minimum_vram_gib {min_vram_gib}")
    if not torch.cuda.is_bf16_supported():
        raise RebuttalError("GPU does not support bf16")
    # A real bf16 kernel launch: fails with "no kernel image" if torch lacks this arch.
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (a @ a).float().sum().backward()
    torch.cuda.synchronize()
    info["bf16_matmul_backward_ok"] = True
    info["arch_in_torch_build"] = info["compute_capability"] in info["torch_arch_list"]
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Revisions and data fingerprints
# ──────────────────────────────────────────────────────────────────────────────

def resolve_revisions(manifest: dict) -> dict:
    from huggingface_hub import HfApi
    api = HfApi()
    common = manifest["common"]
    model_rev = api.model_info(common["model_name"], revision=common.get("model_revision")).sha
    data_rev = api.dataset_info("openai/gsm8k", revision=common.get("dataset_revision")).sha
    return {"model_revision": model_rev, "dataset_revision": data_rev}


def fetch_and_fingerprint(manifest: dict, revisions: dict) -> dict:
    """Download the pinned model snapshot, load GSM8K, fingerprint its splits."""
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    from transformers import AutoConfig
    name = manifest["common"]["model_name"]
    snap = Path(snapshot_download(name, revision=revisions["model_revision"],
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"]))
    weights_gib = sum(f.stat().st_size for f in snap.glob("*.safetensors")) / GIB
    mcfg = AutoConfig.from_pretrained(name, revision=revisions["model_revision"])
    fp = {"model_snapshot_weights_gib": weights_gib, "model_num_hidden_layers": mcfg.num_hidden_layers}
    ds = load_dataset("openai/gsm8k", "main", revision=revisions["dataset_revision"])
    for split in ("train", "test"):
        h = hashlib.sha256()
        for ex in ds[split]:
            h.update(json.dumps([ex["question"], ex["answer"]]).encode())
        fp[f"gsm8k_{split}_rows"] = len(ds[split])
        fp[f"gsm8k_{split}_sha256"] = h.hexdigest()
    return fp


# ──────────────────────────────────────────────────────────────────────────────
# One experiment (runs in its own subprocess via `_job`)
# ──────────────────────────────────────────────────────────────────────────────

# Loader seams: tests replace these with tiny local models/data (no downloads).
def _load_tokenizer(cfg):
    from automode.models import get_tokenizer
    return get_tokenizer(cfg.model_name, cfg.model_revision)


def _load_base_model(cfg):
    from automode.models import load_base_model
    return load_base_model(cfg)


def _load_train_dataset(cfg):
    from automode.data import load_and_normalize_training_data
    return load_and_normalize_training_data(cfg)


def _load_gsm8k_test(cfg):
    from automode.eval import load_gsm8k_test
    return load_gsm8k_test(cfg)


def run_single_job(spec: dict) -> int:
    """
    Train, evaluate, (export) and verify one configuration in attempt_dir.
    Writes status.json throughout; COMPLETE.json (core jobs only) is the very
    last write and only after every check passed. Returns a process exit code.
    """
    import torch
    from automode.config import RunConfig
    from automode.data import make_train_dataloader, tokenize_examples
    from automode.eval import EVALUATORS
    from automode.export import export_full_inference_model
    from automode.models import apply_method
    from automode.train import set_seed, train_one_run, resolve_device, cuda_sync

    attempt = Path(spec["attempt_dir"])
    paths = {k: attempt / k for k in ("configs", "logs", "dynamic", "evals")}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths["run"] = attempt
    status = {"status": "running", "started_at": utcnow(), "pid": os.getpid(), "identity": spec["identity"],
              "purpose": spec["purpose"]}
    write_json_atomic(attempt / "status.json", status)

    def on_signal(signum, _frame):
        raise JobInterrupted(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        cfg = RunConfig(**spec["config"])
        write_json_atomic(paths["configs"] / "run_config.json", cfg.to_dict())
        write_json_atomic(attempt / "environment.json", environment_record())
        device = resolve_device(cfg)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        tokenizer = _load_tokenizer(cfg)
        features, data_stats = tokenize_examples(_load_train_dataset(cfg), tokenizer, cfg)
        loader = make_train_dataloader(features, data_stats, tokenizer, cfg)
        base = _load_base_model(cfg)
        set_seed(cfg.seed)  # adapter init depends only on cfg.seed
        model = apply_method(base, cfg)
        allocation = getattr(model, "automode_allocation", None)
        if allocation:
            write_json_atomic(paths["dynamic"] / "allocation.json", allocation)
        cuda_sync(device)
        setup_sec = time.perf_counter() - t0

        model, train_stats = train_one_run(cfg, model, loader, tokenizer, paths)

        cuda_sync(device)
        t_eval = time.perf_counter()
        metrics: Dict[str, Any] = {}
        for name in cfg.eval_benchmarks:
            if name == "gsm8k":
                metrics.update(EVALUATORS[name](model, tokenizer, cfg, paths, dataset=_load_gsm8k_test(cfg)))
            else:
                metrics.update(EVALUATORS[name](model, tokenizer, cfg, paths))
        cuda_sync(device)
        eval_sec = time.perf_counter() - t_eval
        key = f"gsm8k_maj{cfg.sampling_k}"
        n_total = metrics.get("gsm8k_n_total")
        if n_total != spec["expected_eval_count"]:
            raise RebuttalError(f"gsm8k_n_total={n_total} != expected {spec['expected_eval_count']}")
        if not (isinstance(metrics.get(key), float) and 0.0 <= metrics[key] <= 1.0):
            raise RebuttalError(f"invalid metric {key}={metrics.get(key)!r}")
        mem = {}
        if device.type == "cuda":
            mem = {"peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
                   "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB}

        export = None
        if cfg.save_model:
            export = export_full_inference_model(
                model, tokenizer, attempt / "export", cfg,
                {"job_id": spec["job_id"], "identity": spec["identity"], "commit": spec["source"]["commit"],
                 "dataset_revision": cfg.dataset_revision})
            if spec.get("delete_export_weights"):
                for f in (attempt / "export").rglob("*.safetensors"):
                    f.unlink()
                export["weights_deleted_after_verification"] = True
        del model

        result = {
            "status": "completed", "job_id": spec["job_id"], "purpose": spec["purpose"],
            "identity": spec["identity"], "method": cfg.method, "seed": cfg.seed, "variant": cfg.variant_label(),
            "train_track": cfg.train_track, "protocol_version": cfg.protocol_version,
            "metric_key": f"eval_{key}", f"eval_{key}": metrics[key], "eval_gsm8k_n_total": n_total,
            "metrics": metrics, "train": train_stats, "data_stats": loader.data_stats,
            "setup_time_sec": setup_sec, "train_time_sec": train_stats["train_time_sec"],
            "eval_time_sec": eval_sec, **mem, "export": export, "source": spec["source"],
            "finished_at": utcnow(),
        }
        write_json_atomic(attempt / "result.json", result)
        status.update(status="completed", finished_at=utcnow())
        write_json_atomic(attempt / "status.json", status)
        if spec["purpose"] == "core":
            write_json_atomic(Path(spec["job_dir"]) / "COMPLETE.json", {
                "identity": spec["identity"], "attempt": attempt.relative_to(spec["job_dir"]).as_posix(),
                "completed_at": utcnow()})
        return 0
    except JobInterrupted as e:
        status.update(status="interrupted", error=str(e), finished_at=utcnow())
        write_json_atomic(attempt / "status.json", status)
        return 143
    except Exception as e:
        traceback.print_exc()
        status.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=utcnow())
        write_json_atomic(attempt / "status.json", status)
        return 1


def verify_completed(job_dir: Path, identity: str, require_export: bool, expected_n: int) -> Tuple[bool, str]:
    """True only for a COMPLETE marker with this identity whose artefacts check out."""
    marker = job_dir / "COMPLETE.json"
    if not marker.exists():
        return False, "no COMPLETE marker"
    try:
        done = read_json(marker)
        if done.get("identity") != identity:
            return False, "COMPLETE identity differs from this configuration/code"
        attempt = (job_dir / done["attempt"]).resolve()
        if not attempt.is_relative_to(job_dir.resolve()):
            return False, "COMPLETE points outside the job directory"
        res = read_json(attempt / "result.json")
        if res.get("status") != "completed" or res.get("identity") != identity:
            return False, "result.json is not a completed result for this identity"
        m = res.get(METRIC_KEY)
        if not isinstance(m, float) or not 0.0 <= m <= 1.0 or res.get("eval_gsm8k_n_total") != expected_n:
            return False, "metric or evaluation count invalid"
        ev = read_json(attempt / "evals" / "gsm8k_eval.json")
        if ev.get("gsm8k_n_total") != expected_n or ("records" in ev and len(ev["records"]) != expected_n):
            return False, "gsm8k_eval.json counts do not match"
        if require_export:
            meta = read_json(attempt / "export" / "export_metadata.json")
            if meta.get("status") != "ok" or not list((attempt / "export").glob("*.safetensors")):
                return False, "requested full export missing or unverified"
            if meta.get("kind") == "full_unmerged_inference_export" and not (
                    (attempt / "export" / "layer_state.json").exists()
                    and (attempt / "export" / "adapter" / "adapter_model.safetensors").exists()):
                return False, "unmerged export is missing its adapters or layer state"
        return True, "verified"
    except (OSError, KeyError, ValueError, TypeError) as e:
        return False, f"verification error: {type(e).__name__}: {e}"


def next_attempt_dir(job_dir: Path) -> Path:
    root = job_dir / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[1]) for p in root.glob("attempt_*")] or [0])
    d = root / f"attempt_{n:03d}"
    d.mkdir()
    return d


class StopState:
    def __init__(self):
        self.stop_requested = False
        self.child: Optional[subprocess.Popen] = None
        self.requested_at = 0.0

    def install(self):
        def handler(signum, _frame):
            self.stop_requested = True
            self.requested_at = time.monotonic()
            log(f"received signal {signum}: stopping after interrupting the current job")
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def launch_job_subprocess(spec: dict, stop: StopState) -> int:
    """Fresh python process per experiment, same process group (so the external
    supervisor's group signal reaches it), stdout+stderr to attempt/job.log."""
    attempt = Path(spec["attempt_dir"])
    write_json_atomic(attempt / "spec.json", spec)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(attempt / "job.log", "a", buffering=1) as fh:
        child = subprocess.Popen([sys.executable, "-m", "automode.rebuttal", "_job", "--spec",
                                  str(attempt / "spec.json")], stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, cwd=REPO_ROOT, env=env)
        stop.child = child
        try:
            while child.poll() is None:
                time.sleep(1)
                if stop.stop_requested and time.monotonic() - stop.requested_at > 60:
                    child.kill()
            return child.wait()
        finally:
            stop.child = None


# ──────────────────────────────────────────────────────────────────────────────
# Reports and gating
# ──────────────────────────────────────────────────────────────────────────────

def _common_context(args) -> Tuple[dict, str, dict, Path]:
    manifest, msha = load_manifest(args.manifest)
    validate_manifest(manifest)
    src = source_identity(msha)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bind_output_identity(out, src)
    return manifest, msha, src, out


def _reported_phase(args, report_name: str, body: Callable[..., dict]) -> int:
    """Run a phase that must leave a status report. A failed report is written
    for every failure except an output directory owned by another source."""
    out = Path(args.output).resolve()
    report: Dict[str, Any] = {"status": "failed", "api_version": REBUTTAL_API_VERSION,
                              "worker_id": args.worker_id, "started_at": utcnow()}
    try:
        manifest, msha, src, out = _common_context(args)
        report.update(src)
        report.update(body(manifest, msha, src, out, report))
        report.update(status="ok", finished_at=utcnow())
    except OutputIdentityError:
        raise
    except Exception as e:
        traceback.print_exc()
        report.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=utcnow())
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / report_name, report)
    return 0 if report["status"] == "ok" else 1


def _require_report(out: Path, name: str, src: dict, worker_id: int) -> dict:
    p = out / name
    if not p.exists():
        raise RebuttalError(f"{name} missing: run the previous phase first")
    rep = read_json(p)
    if rep.get("status") != "ok" or rep.get("worker_id") != worker_id or any(
            rep.get(k) != src[k] for k in IDENTITY_KEYS):
        raise RebuttalError(f"{name} is not a successful report for this worker/source/manifest")
    return rep


def _resolved(out: Path, msha: str) -> dict:
    rm = read_json(out / "resolved_manifest.json")
    if rm.get("manifest_sha256") != msha:
        raise RebuttalError("resolved_manifest.json was produced for a different manifest")
    return rm["revisions"]


def _spec(purpose: str, job_id: str, cfg, identity: str, attempt: Path, job_dir: Path,
          expected: int, src: dict, **extra) -> dict:
    return {"purpose": purpose, "job_id": job_id, "config": cfg.to_dict(), "identity": identity,
            "attempt_dir": str(attempt), "job_dir": str(job_dir), "expected_eval_count": expected,
            "source": src, **extra}


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_contract(_args) -> int:
    from automode.config import CORE_METHODS
    print(json.dumps({"api_version": REBUTTAL_API_VERSION, "supported_methods": list(CORE_METHODS),
                      "metric_key": METRIC_KEY}))
    return 0


def cmd_preflight(args) -> int:
    return _reported_phase(args, "preflight_report.json", lambda *a: _preflight(args, *a))


def _preflight(args, manifest, msha, src, out, report) -> dict:
    with file_lock(out / ".worker.lock", f"worker {args.worker_id} output"):
        jobs = worker_jobs(manifest, args.worker_id)
        mism = verify_installed_lock()
        if mism:
            raise RebuttalError(f"installed packages differ from requirements-rebuttal.lock: {mism}")
        env = environment_record()
        write_json_atomic(out / "environment.json", env)
        report.update(torch_version=env["torch"], python=env["python"])
        report.update(check_gpu(manifest["minimum_vram_gib"]))

        log("running test suite (with AUTOMODE_TEST_DEVICE=cuda)")
        with open(out / "preflight_pytest.log", "w") as fh:
            rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT,
                                stdout=fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "AUTOMODE_TEST_DEVICE": "cuda"}).returncode
        report["tests_passed"] = rc == 0
        if rc != 0:
            raise RebuttalError(f"pytest failed (exit {rc}); see preflight_pytest.log")

        rm_path = out / "resolved_manifest.json"
        if rm_path.exists():
            revisions = _resolved(out, msha)  # never re-resolve: keep the frozen revisions
        else:
            revisions = resolve_revisions(manifest)
        for k, v in revisions.items():
            if manifest["common"].get(k) not in (None, v):
                raise RebuttalError(f"{k} resolved to {v} but manifest pins {manifest['common'][k]}")
        fp = fetch_and_fingerprint(manifest, revisions)
        if fp["gsm8k_test_rows"] != manifest["expected_eval_count"]:
            raise RebuttalError(f"GSM8K test has {fp['gsm8k_test_rows']} rows, "
                                f"manifest expects {manifest['expected_eval_count']}")
        identities = {}
        for j in manifest["jobs"]:
            cfg = job_config(manifest, j, revisions)
            if cfg.method == "fixed_mixed" and cfg.fixed_base_layers > fp["model_num_hidden_layers"]:
                raise RebuttalError(f"{j['job_id']}: fixed_base_layers exceeds model layers")
            identities[j["job_id"]] = job_identity(cfg, src)
        if len(set(identities.values())) != len(identities):
            raise RebuttalError("two jobs resolve to the same scientific identity")
        n_exports = sum(bool(job_config(manifest, j).save_model) for j in jobs)
        need = (n_exports + 1) * fp["model_snapshot_weights_gib"] + 10  # +1: pilot export, deleted after check
        free = shutil.disk_usage(out).free / GIB
        report.update(disk_free_gib=free, disk_required_gib=need)
        if free < need:
            raise RebuttalError(f"{free:.0f} GiB free under {out}, need ~{need:.0f} GiB")
        if not rm_path.exists():
            write_json_atomic(rm_path, {"manifest_sha256": msha, "revisions": revisions, "manifest": manifest,
                                        "job_identities": identities, **src, "resolved_at": utcnow()})
        return dict(model_revision=revisions["model_revision"],
                    dataset_revision=revisions["dataset_revision"], dataset_fingerprint=fp,
                    assigned_jobs=[j["job_id"] for j in jobs], job_identities=identities)


def pilot_config(manifest: dict, job: dict, revisions: dict, **extra):
    cfg = job_config(manifest, job, revisions, max_train_samples=PILOT_TRAIN_SAMPLES, epochs=1,
                     max_eval_samples=PILOT_EVAL_SAMPLES, **extra)
    if cfg.method in ("automode", "select_once"):
        cfg.dynamic_updates = PILOT_DYNAMIC_UPDATES
    return cfg


def estimate_job_hours(cfg, n_train: int, expected_eval: int, timing: dict) -> float:
    steps = cfg.epochs * math.ceil(math.ceil(n_train / cfg.train_batch_size) / cfg.grad_accum_steps)
    sec = (timing["setup_sec"] + steps * timing["step_sec"] + expected_eval * timing["eval_sec_per_item"]
           + (timing["export_sec"] if cfg.save_model else 0.0))
    return ETA_SAFETY_FACTOR * sec / 3600


def cmd_pilot(args) -> int:
    return _reported_phase(args, "pilot_report.json", lambda *a: _pilot(args, *a))


def _pilot(args, manifest, msha, src, out, report) -> dict:
    stop = StopState()
    stop.install()
    with file_lock(out / ".worker.lock", f"worker {args.worker_id} output"):
        pre = _require_report(out, "preflight_report.json", src, args.worker_id)
        revisions = _resolved(out, msha)
        jobs = worker_jobs(manifest, args.worker_id)
        pilot_root = out / "pilot"

        def run(purpose, name, cfg, expected, **extra):
            job_dir = pilot_root / name
            attempt = next_attempt_dir(job_dir)
            spec = _spec(purpose, name, cfg, job_identity(cfg, src), attempt, job_dir, expected, src, **extra)
            log(f"pilot: {name} -> {attempt}")
            rc = launch_job_subprocess(spec, stop)
            if stop.stop_requested:
                raise RebuttalError("pilot interrupted")
            if rc != 0:
                st = read_json(attempt / "status.json") if (attempt / "status.json").exists() else {}
                raise RebuttalError(f"pilot {name} failed: {st.get('error')} (log {attempt / 'job.log'})")
            return read_json(attempt / "result.json")

        checks = {}
        for method in dict.fromkeys(j["overrides"]["method"] for j in jobs):
            job = next(j for j in jobs if j["overrides"]["method"] == method)
            cfg = job_config(manifest, job, revisions, max_train_samples=64, epochs=1, grad_accum_steps=4,
                             max_eval_samples=2, max_new_tokens=32, save_model=False)
            if method in ("automode", "select_once"):
                cfg.dynamic_updates = PILOT_DYNAMIC_UPDATES
            r = run("method_check", f"method_checks/{method}", cfg, 2)
            checks[method] = {"ok": True, "total_opt_steps": r["train"]["total_opt_steps"],
                              "n_switch_events": r["train"]["n_switch_events"],
                              "branch_step": r["train"]["branch_step"],
                              "final_trainable_params": r["train"]["final_trainable_params"],
                              "peak_vram_reserved_gib": r.get("peak_vram_reserved_gib")}

        methods = {j["overrides"]["method"] for j in jobs}
        focus = "automode" if "automode" in methods else "lora" if "lora" in methods else None
        if focus is None:
            raise RebuttalError("worker has neither automode nor lora jobs to pilot")
        if "automode" in methods and methods - {"automode", "select_once", "fixed_mixed", "static_qv"}:
            raise RebuttalError("pilot timing from automode cannot cover this worker's methods")
        if focus == "lora" and methods != {"lora"}:
            raise RebuttalError("LoRA pilot timing cannot cover non-LoRA methods")
        job = next(j for j in jobs if j["overrides"]["method"] == focus)
        save = any(job_config(manifest, j).save_model for j in jobs)
        cfg = pilot_config(manifest, job, revisions, save_model=save)
        r = run("pilot", "main", cfg, PILOT_EVAL_SAMPLES, delete_export_weights=True)
        t = r["train"]
        post = t["post_promotion_optimizer_steps"] if focus == "automode" else None
        if focus == "automode" and (post or 0) < MIN_POST_PROMOTION_STEPS:
            raise RebuttalError(f"only {post} post-promotion optimizer steps (< {MIN_POST_PROMOTION_STEPS})")
        step_sec = t["post_promotion_mean_step_sec"] if focus == "automode" else t["mean_step_sec"]
        timing = {"setup_sec": r["setup_time_sec"], "step_sec": step_sec,
                  "eval_sec_per_item": r["metrics"]["gsm8k_eval_time_sec"] / r["eval_gsm8k_n_total"],
                  "export_sec": (r["export"] or {}).get("export_time_sec", 0.0)}
        if save and not r["export"]:
            raise RebuttalError("pilot did not exercise the requested export")
        n_train = pre["dataset_fingerprint"]["gsm8k_train_rows"]
        estimates = {j["job_id"]: estimate_job_hours(job_config(manifest, j, revisions), n_train,
                                                     manifest["expected_eval_count"], timing) for j in jobs}
        return dict(
            n_eval=r["eval_gsm8k_n_total"],
            peak_vram_gib=max([r.get("peak_vram_reserved_gib") or 0.0]
                              + [c["peak_vram_reserved_gib"] or 0.0 for c in checks.values()]),
            post_promotion_optimizer_steps=post,
            estimated_core_queue_hours=sum(estimates.values()), estimate_is_approximate=True,
            job_estimates_hours=estimates, timing_inputs=timing, pilot_method=focus,
            pilot_total_opt_steps=t["total_opt_steps"], pilot_first_promotion_step=t["first_promotion_step"],
            generation_tokens_per_sec=r["metrics"]["gsm8k_generated_tokens"] / r["metrics"]["gsm8k_eval_time_sec"],
            base_update_changed_fraction_mean=t["base_update_changed_fraction_mean"],
            dtype_census=t["dtype_census"], method_checks=checks,
            export_kind=(r["export"] or {}).get("kind"), export_checks=(r["export"] or {}).get("checks"),
            pilot_metric_not_a_result=r[r["metric_key"]],
            model_revision=revisions["model_revision"], dataset_revision=revisions["dataset_revision"],
            assumptions=[
                f"all core steps timed at the measured {'post-promotion' if focus == 'automode' else 'LoRA'} "
                f"mean step time (pre-promotion LoRA steps are faster, so this is conservative for AutoMode)",
                "static_qv/fixed_mixed/select_once assumed to cost no more per step than post-promotion "
                "AutoMode (37 base layers); static_qv trains 42 — covered only by the safety factor",
                f"generation time per item scaled linearly from {PILOT_EVAL_SAMPLES} to "
                f"{manifest['expected_eval_count']} items with a briefly trained pilot model",
                f"safety factor {ETA_SAFETY_FACTOR} on every job; not a guarantee of final runtime",
                "the pilot metric is NOT a scientific result",
            ])


LEDGER_METRICS = ("eval_gsm8k_maj3", "eval_gsm8k_n_total", "train_time_sec", "eval_time_sec", "peak_vram_gib")


def execute_queue(out: Path, worker_id: int, manifest: dict, revisions: dict, src: dict,
                  estimates: Dict[str, float], hours: float,
                  launch: Callable[[dict], int], stop: StopState) -> int:
    """Sequential core queue for one worker. Returns 0 only if every job is verified complete."""
    jobs = worker_jobs(manifest, worker_id)
    expected = manifest["expected_eval_count"]
    ledger_path = out / "run_ledger.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if ledger.get("worker_id") != worker_id or any(ledger.get(k) != src[k] for k in ("commit", "manifest_sha256")):
            raise RebuttalError("run_ledger.json belongs to another worker/source/manifest")
    else:
        ledger = {"schema_version": 1, "worker_id": worker_id, "commit": src["commit"],
                  "manifest_sha256": src["manifest_sha256"], "runs": []}
    rows = {r["job_id"]: r for r in ledger["runs"]}
    for j in jobs:
        row = rows.setdefault(j["job_id"], {"job_id": j["job_id"], "method": j["overrides"]["method"],
                                            "seed": j["overrides"]["seed"], "status": "pending"})
        if row["status"] == "running":
            row.update(status="interrupted", error="previous runner stopped while this job was running")
    ledger["runs"] = [rows[j["job_id"]] for j in jobs]

    def save():
        ledger["updated_at"] = utcnow()
        write_json_atomic(ledger_path, ledger)

    def clear(row):
        for k in LEDGER_METRICS + ("error", "log_path"):
            row.pop(k, None)

    save()
    deadline = time.monotonic() + hours * 3600
    budget_stopped = False
    for j in jobs:
        row = rows[j["job_id"]]
        cfg = job_config(manifest, j, revisions)
        identity = job_identity(cfg, src)
        job_dir = out / "core" / j["job_id"]
        ok, why = verify_completed(job_dir, identity, cfg.save_model, expected)
        if ok:
            done = read_json(job_dir / "COMPLETE.json")
            res = read_json(job_dir / done["attempt"] / "result.json")
            clear(row)
            row.update(status="completed", identity=identity,
                       result_dir=(Path("core") / j["job_id"] / done["attempt"]).as_posix(),
                       eval_gsm8k_maj3=res[METRIC_KEY], eval_gsm8k_n_total=res["eval_gsm8k_n_total"],
                       train_time_sec=res["train_time_sec"], eval_time_sec=res["eval_time_sec"],
                       peak_vram_gib=res.get("peak_vram_reserved_gib"))
            save()
            continue
        if (job_dir / "COMPLETE.json").exists():
            clear(row)
            row.update(status="failed", error=f"existing COMPLETE.json rejected: {why}")
            save()
            continue
        if stop.stop_requested:
            row.update(status="interrupted", error="runner stop requested before start")
            save()
            continue
        est = estimates.get(j["job_id"])
        if est is None:
            raise RebuttalError(f"pilot report has no estimate for {j['job_id']}")
        remaining = (deadline - time.monotonic()) / 3600
        if budget_stopped or est > remaining:
            budget_stopped = True
            clear(row)
            row.update(status="budget_exhausted", estimated_hours=est, remaining_hours=max(remaining, 0.0))
            save()
            continue
        with file_lock(job_dir / ".run.lock", f"job {j['job_id']}"):
            attempt = next_attempt_dir(job_dir)
            clear(row)
            row.update(status="running", identity=identity, estimated_hours=est, started_at=utcnow(),
                       result_dir=attempt.relative_to(out).as_posix(),
                       log_path=(attempt / "job.log").relative_to(out).as_posix())
            save()
            rc = launch(_spec("core", j["job_id"], cfg, identity, attempt, job_dir, expected, src))
            ok, why = verify_completed(job_dir, identity, cfg.save_model, expected)
            row["finished_at"] = utcnow()
            if ok:
                res = read_json(attempt / "result.json")
                row.update(status="completed", eval_gsm8k_maj3=res[METRIC_KEY],
                           eval_gsm8k_n_total=res["eval_gsm8k_n_total"], train_time_sec=res["train_time_sec"],
                           eval_time_sec=res["eval_time_sec"], peak_vram_gib=res.get("peak_vram_reserved_gib"))
            else:
                st = read_json(attempt / "status.json") if (attempt / "status.json").exists() else {}
                interrupted = stop.stop_requested or st.get("status") == "interrupted"
                row.update(status="interrupted" if interrupted else "failed",
                           error=st.get("error") or f"exit code {rc}; {why}")
            save()
    statuses = [rows[j["job_id"]]["status"] for j in jobs]
    write_json_atomic(out / "campaign_status.json", {
        "complete": all(s == "completed" for s in statuses),
        "counts": {s: statuses.count(s) for s in sorted(set(statuses))}, "updated_at": utcnow()})
    if stop.stop_requested:
        return 143
    return 0 if all(s == "completed" for s in statuses) else 1


def cmd_run(args) -> int:
    manifest, msha, src, out = _common_context(args)
    if not args.hours > 0:
        raise RebuttalError("--hours must be positive")
    stop = StopState()
    stop.install()
    with file_lock(out / ".worker.lock", f"worker {args.worker_id} output"):
        _require_report(out, "preflight_report.json", src, args.worker_id)
        pilot = _require_report(out, "pilot_report.json", src, args.worker_id)
        revisions = _resolved(out, msha)
        if any(pilot.get(k) != revisions[k] for k in ("model_revision", "dataset_revision")):
            raise RebuttalError("pilot report revisions differ from resolved_manifest.json")
        return execute_queue(out, args.worker_id, manifest, revisions, src, pilot["job_estimates_hours"],
                             args.hours, lambda spec: launch_job_subprocess(spec, stop), stop)


def cmd_job(args) -> int:
    return run_single_job(read_json(Path(args.spec)))


def cmd_export_diagnostic(args) -> int:
    return _reported_phase(args, "export_diagnostic_report.json", lambda *a: _export_diagnostic(args, *a))


def _export_diagnostic(args, manifest, msha, src, out, report) -> dict:
    """Train the AutoMode pilot configuration, skip evaluation, and measure the
    merge. Prompts come from the GSM8K train split, never the test set."""
    from automode.data import make_train_dataloader, tokenize_examples, encode_prompt
    from automode.export import base_update_assessment, export_diagnostic
    from automode.models import apply_method
    from automode.train import set_seed, train_one_run

    with file_lock(out / ".worker.lock", f"worker {args.worker_id} output"):
        _require_report(out, "preflight_report.json", src, args.worker_id)
        revisions = _resolved(out, msha)
        job = next(j for j in worker_jobs(manifest, args.worker_id) if j["overrides"]["method"] == "automode")
        cfg = pilot_config(manifest, job, revisions, save_model=True)
        attempt = next_attempt_dir(out / "diagnostics" / "export")
        paths = {k: attempt / k for k in ("configs", "logs", "dynamic", "evals")}
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        paths["run"] = attempt

        tokenizer = _load_tokenizer(cfg)
        dataset = _load_train_dataset(cfg)
        features, data_stats = tokenize_examples(dataset, tokenizer, cfg)
        loader = make_train_dataloader(features, data_stats, tokenizer, cfg)
        base = _load_base_model(cfg)
        set_seed(cfg.seed)
        model = apply_method(base, cfg)
        model, train_stats = train_one_run(cfg, model, loader, tokenizer, paths)

        # Train-split rows this pilot did not train on, taken from the end of the split.
        trained_on = {dataset[i]["prompt"] for i in range(len(dataset))}
        full = _load_train_dataset(dataclasses.replace(cfg, max_train_samples=None))
        unseen = [full[i]["prompt"] for i in reversed(range(len(full))) if full[i]["prompt"] not in trained_on]
        prompts = [encode_prompt(tokenizer, p, cfg.max_source_len)[0] for p in unseen[:EXPORT_DIAGNOSTIC_PROMPTS]]
        probe_log = read_json(paths["logs"] / "base_update_probe.json")["base_update_probe"]
        batch = next(iter(make_train_dataloader(features, data_stats, tokenizer, cfg)))  # sampler is single-use
        assessment = base_update_assessment(model, batch, probe_log, cfg)
        try:
            diagnostic = export_diagnostic(model, tokenizer, prompts, attempt / "export", cfg,
                                           {"job_id": job["job_id"], "commit": src["commit"]})
        finally:
            for f in (attempt / "export").rglob("*.safetensors"):
                f.unlink()
        diagnostic.update(base_update_assessment=assessment, train=train_stats, attempt=str(attempt))
        write_json_atomic(attempt / "export_diagnostic.json", diagnostic)
        if "export_error" in diagnostic:
            raise RebuttalError(f"{diagnostic['export_error']} (details: {attempt / 'export_diagnostic.json'})")
        return diagnostic


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m automode.rebuttal")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("contract")
    for name in ("preflight", "pilot", "run", "export-diagnostic"):
        sp = sub.add_parser(name)
        sp.add_argument("--manifest", required=True, type=Path)
        sp.add_argument("--worker-id", required=True, type=int, choices=range(N_WORKERS))
        sp.add_argument("--output", required=True, type=Path)
        if name == "run":
            sp.add_argument("--hours", required=True, type=float)
    sj = sub.add_parser("_job")
    sj.add_argument("--spec", required=True)
    args = p.parse_args(argv)
    handler = {"contract": cmd_contract, "preflight": cmd_preflight, "pilot": cmd_pilot,
               "run": cmd_run, "export-diagnostic": cmd_export_diagnostic, "_job": cmd_job}[args.cmd]
    try:
        return handler(args)
    except RebuttalError as e:
        log(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
