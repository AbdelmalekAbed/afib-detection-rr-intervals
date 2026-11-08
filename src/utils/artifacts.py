"""Non-destructive experiment directories and provenance for new runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def new_run_dir(parent: Path, name: str) -> Path:
    """Create a unique directory; never reuse a previous experiment's outputs."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(parent) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_new_directory(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} contient déjà des fichiers : choisir un nouveau répertoire.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_sources(path: Path) -> None:
    """Keep the actual executed source tree even when the working tree changes later."""
    path.mkdir(parents=True, exist_ok=False)
    sources = [p for folder in ("src", "scripts", "configs")
               for p in Path(folder).rglob("*") if p.suffix in {".py", ".yaml"}]
    sources += [Path(name) for name in ("pyproject.toml", "requirements-lock.txt", "requirements.txt", "README.md", "LICENSE", "Makefile")]
    for source in sources:
        target = path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())


def provenance(source_files: list[Path] = ()) -> dict:
    packages = {}
    for name in ("torch", "numpy", "scikit-learn", "wfdb", "optuna"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "scripts", "configs"])
        dirty = bool(diff)
        diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        head, dirty, diff_hash = None, None, None
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(), "packages": packages,
        "git_head": head, "tracked_code_dirty": dirty, "tracked_diff_sha256": diff_hash,
        "source_sha256": {str(p): file_digest(p) for p in source_files},
        "code_sha256": {str(p): file_digest(p) for folder in ("src", "scripts", "configs")
                        for p in sorted(Path(folder).rglob("*")) if p.suffix in {".py", ".yaml"}},
    }


def write_json(path: Path, value: dict) -> None:
    """Exclusive creation prevents accidental replacement, including historical results."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
