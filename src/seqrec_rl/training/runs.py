"""Run directories: resolved config, metrics, command line, environment info, compute accounting."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .. import PROJECT_ROOT, PROCESSED_DIR, RUNS_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_threads(n: int = 4) -> None:
    """Keep CPU usage modest (the workstation is shared with other jobs). SEQRL_THREADS overrides ``n``."""
    n = int(os.environ.get("SEQRL_THREADS", n))
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


class Run:
    def __init__(self, name: str, seed: int, cfg: dict, root: Path = RUNS_DIR):
        self.dir = root / name / f"seed{seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.cfg = cfg
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest = PROCESSED_DIR / "manifest.json"
        info = {
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "git": git_rev(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
            "data_manifest_sha": _file_hash(manifest) if manifest.exists() else None,
            "seed": seed,
        }
        (self.dir / "run_info.json").write_text(json.dumps(info, indent=2))
        self._log = open(self.dir / "log.jsonl", "a", encoding="utf-8")

    def log(self, **kw) -> None:
        kw["wall_s"] = round(time.time() - self.t0, 2)
        self._log.write(json.dumps(kw) + "\n")
        self._log.flush()

    def save_metrics(self, metrics: dict, name: str = "metrics.json") -> None:
        prev = {}
        compute = self.dir / "compute.json"
        if compute.exists():
            prev = json.loads(compute.read_text())
        elapsed = time.time() - self.t0
        gpu = torch.cuda.is_available()
        total = prev.get("total_wall_s", 0.0) + elapsed
        compute.write_text(json.dumps({"total_wall_s": total, "gpu_hours": total / 3600 if gpu else 0.0,
                                       "device": "cuda" if gpu else "cpu"}, indent=2))
        self.t0 = time.time()
        (self.dir / name).write_text(json.dumps(metrics, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _file_hash(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
