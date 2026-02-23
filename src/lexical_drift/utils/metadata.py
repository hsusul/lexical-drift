from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def config_sha256(config_obj: object) -> str:
    if is_dataclass(config_obj):
        payload = asdict(config_obj)
    elif isinstance(config_obj, dict):
        payload = dict(config_obj)
    else:
        payload = {"value": repr(config_obj)}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
