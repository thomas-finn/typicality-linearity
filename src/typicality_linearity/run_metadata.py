"""Record run metadata: git commit, config, seed, environment and status."""

import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata as package_metadata

import numpy as np

PACKAGES = ("torch", "transformers", "numpy", "pyyaml", "scikit-learn")


def run_git(args):
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_info():
    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


def package_versions(packages=PACKAGES):
    versions = {}
    for name in packages:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def now_utc():
    return datetime.now(UTC).isoformat()


def make_run_id(experiment_name):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{experiment_name}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def start_metadata(run_id, config, config_path):
    return {
        "run_id": run_id,
        "config_path": str(config_path),
        "config": config,
        "seed": config.get("seed"),
        "git": git_info(),
        "environment": environment_info(),
        "status": "running",
        "started_at": now_utc(),
        "finished_at": None,
        "error": None,
    }


def write_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
