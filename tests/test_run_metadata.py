import subprocess

from typicality_linearity import run_metadata
from typicality_linearity.run_metadata import git_diff_text, start_metadata


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "data.csv").write_text("a\n1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


def test_git_diff_restores_tracked_and_untracked_code(tmp_path):
    root = make_repo(tmp_path)
    (root / "src" / "a.py").write_text("x = 2\n")
    (root / "src" / "b.py").write_text("y = 3\n")
    (root / "data.csv").write_text("a\n2\n")

    text = git_diff_text(str(root))
    assert "x = 2" in text
    assert "y = 3" in text
    assert "data.csv" not in text

    patch = tmp_path / "run.patch"
    patch.write_text(text)
    git(root, "checkout", "--", ".")
    (root / "src" / "b.py").unlink()
    git(root, "apply", str(patch))
    assert (root / "src" / "a.py").read_text() == "x = 2\n"
    assert (root / "src" / "b.py").read_text() == "y = 3\n"


def test_start_metadata_writes_the_diff_when_dirty(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    (root / "src" / "a.py").write_text("x = 2\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(root)

    metadata = start_metadata("id", {"seed": 0}, "config.yaml", run_dir)
    assert metadata["git"]["dirty"]
    assert metadata["git"]["diff_file"] == run_metadata.GIT_DIFF_FILE
    assert "x = 2" in (run_dir / run_metadata.GIT_DIFF_FILE).read_text()


def test_start_metadata_writes_no_diff_when_clean(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(root)

    metadata = start_metadata("id", {"seed": 0}, "config.yaml", run_dir)
    assert not metadata["git"]["dirty"]
    assert "diff_file" not in metadata["git"]
    assert not (run_dir / run_metadata.GIT_DIFF_FILE).exists()
