"""Training entry point — `python -m src.train --config configs/train.yaml`.

The orchestration is deliberately kept minimal: this file wires together the data
pipeline, model, loss, and metrics. Hyperparameter search lives in a separate
script (``scripts/optuna_search.py``) that imports the same building blocks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    _ = load_config(args.config)
    raise NotImplementedError(
        "Implement training loop in Phase 3: build dataloaders from configs/data.yaml, "
        "instantiate CNNLSTM from configs/model_cnn_lstm.yaml, optimize with AdamW + "
        "BCEWithLogitsLoss, early-stop on val mean_patient_f1, save best checkpoint."
    )


if __name__ == "__main__":
    main()
