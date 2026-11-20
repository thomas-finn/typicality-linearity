"""Record run metadata: git commit and uncommitted changes, config, seed,
environment and status."""

import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np

PACKAGES = ("torch", "transformers", "numpy", "pyyaml", "scikit-learn")


# The paths whose uncommitted changes are saved with a run. Data and outputs
# are not saved: data/processed is made by the scripts from this code.
CODE_PATHS = ("src", "scripts", "tests", "experiments", "pyproject.toml", "uv.lock")
CODE_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".md", ".lock")
GIT_DIFF_FILE = "git_diff.patch"


def run_git(args, ok_codes=(0,), strip=True):
    """Return the output of git, or None if git fails."""
    try:
        result = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        )
    except OSError:
        return None
    if result.returncode not in ok_codes:
        return None
    if strip:
        return result.stdout.strip()
    return result.stdout


def git_info():
    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def untracked_code_files(root):
    output = run_git(
        ["-C", root, "ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS]
    )
    if not output:
        return []
    return [path for path in output.splitlines() if path.endswith(CODE_SUFFIXES)]


def git_diff_text(root):
    """Return a patch of the uncommitted code changes against HEAD.

    It has the changes of tracked files and the full text of untracked code
    files. On the recorded commit, `git apply` gives the code of the run.
    """
    parts = [
        run_git(["-C", root, "diff", "HEAD", "--", *CODE_PATHS], strip=False) or ""
    ]
    for path in untracked_code_files(root):
        # Exit code 1 means that the files are different, which is expected.
        diff = run_git(
            ["-C", root, "diff", "--no-index", "--", "/dev/null", path],
            ok_codes=(0, 1),
            strip=False,
        )
        parts.append(diff or "")
    return "".join(parts)


def save_git_diff(run_dir):
    """Write the uncommitted code changes to run_dir. Return the file name,
    or None if there are no changes or git is not available."""
    root = run_git(["rev-parse", "--show-toplevel"])
    if root is None:
        return None
    text = git_diff_text(root)
    if not text:
        return None
    path = Path(run_dir) / GIT_DIFF_FILE
    path.write_text(text)
    return GIT_DIFF_FILE


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


def start_metadata(run_id, config, config_path, run_dir=None):
    """Return the metadata of a new run.

    With run_dir, the uncommitted code changes are also written to
    run_dir/git_diff.patch (see save_git_diff), and git.diff_file names it.
    """
    git = git_info()
    if run_dir is not None and git["dirty"]:
        git["diff_file"] = save_git_diff(run_dir)
    return {
        "run_id": run_id,
        "config_path": str(config_path),
        "config": config,
        "seed": config.get("seed"),
        "git": git,
        "environment": environment_info(),
        "status": "running",
        "started_at": now_utc(),
        "finished_at": None,
        "error": None,
    }


def write_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
