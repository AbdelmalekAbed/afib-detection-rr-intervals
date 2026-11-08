"""Link read-only scientific packages into the isolated web virtual environment."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".venv-web/bin/python"
    site = subprocess.check_output(
        [str(target), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True
    ).strip()
    # Append the existing scientific packages after web-specific dependencies.
    # Nothing is installed, removed or written in the scientific environment.
    (Path(site) / "afib_scientific.pth").write_text(
        sysconfig.get_paths()["purelib"] + "\n" + str(root) + "\n"
    )
    print(f"Environnement web lié aux bibliothèques scientifiques de {sys.prefix}")


if __name__ == "__main__":
    main()
