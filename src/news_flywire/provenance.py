from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def manifest(paths, **metadata):
    packages = {}
    for name in ["numpy", "pandas", "scipy", "networkx", "matplotlib", "pyarrow", "brian2"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": packages,
        "files": {str(p): sha256(p) for p in paths}, **metadata,
    }


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
