"""Evaluation entry point — `python -m src.evaluate --config configs/train.yaml`.

Produces:
- Internal test report (window-level + patient-level metrics).
- External report on LTAFDB (with ``--external ltafdb``).
- Confusion matrices and ROC/PR curves saved under ``reports/figures/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--external", type=str, default=None, help="External dataset id (e.g., ltafdb)")
    p.add_argument("--checkpoint", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "Implement evaluation in Phase 3-5: load checkpoint, score the test set, "
        "compute threshold + ranking + per-patient metrics, write JSON + figures."
    )


if __name__ == "__main__":
    main()
