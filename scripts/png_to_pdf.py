"""Wrap every PNG in `reports/figures/` into a same-name PDF in `report/figures/`.

Lossless embedding (no re-rasterisation). Used to feed the LaTeX memoir with
`\\includegraphics{figures/<name>.pdf}` without re-running the notebooks.

Idempotent: skips PDFs already newer than their source PNG.

Run from the repo root:
    python scripts/png_to_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "reports" / "figures"
DST_DIR = REPO_ROOT / "report" / "figures"


def convert_one(png_path: Path, pdf_path: Path) -> str:
    if pdf_path.exists() and pdf_path.stat().st_mtime >= png_path.stat().st_mtime:
        return "skip"
    with Image.open(png_path) as img:
        rgb = img.convert("RGB") if img.mode in {"RGBA", "P", "LA"} else img
        rgb.save(pdf_path, "PDF", resolution=300.0)
    return "ok"


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"error: source directory not found: {SRC_DIR}", file=sys.stderr)
        return 1
    DST_DIR.mkdir(parents=True, exist_ok=True)

    pngs = sorted(SRC_DIR.glob("*.png"))
    if not pngs:
        print(f"no PNGs found in {SRC_DIR}", file=sys.stderr)
        return 1

    counts = {"ok": 0, "skip": 0, "err": 0}
    for png in pngs:
        pdf = DST_DIR / (png.stem + ".pdf")
        try:
            status = convert_one(png, pdf)
        except Exception as exc:
            print(f"  ERR  {png.name} → {exc}", file=sys.stderr)
            counts["err"] += 1
            continue
        marker = {"ok": "  +  ", "skip": "  =  "}[status]
        print(f"{marker}{png.name} → {pdf.relative_to(REPO_ROOT)}")
        counts[status] += 1

    print(
        f"\nconverted: {counts['ok']}  |  up-to-date: {counts['skip']}  |  errors: {counts['err']}"
    )
    return 0 if counts["err"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
