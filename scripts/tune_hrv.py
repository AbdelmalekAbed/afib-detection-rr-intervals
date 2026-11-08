"""Optimize HRV gradient boosting with patient-grouped inner/outer folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.baselines.search import nested_hrv_oof
from src.data.artifacts import load_windows
from src.features.hrv import FEATURE_VERSION, featurize_windows
from src.utils.artifacts import (
    file_digest,
    new_run_dir,
    provenance,
    require_new_directory,
    snapshot_sources,
    write_json,
)
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = require_new_directory(args.out_dir) if args.out_dir else new_run_dir(Path("artifacts"), "hrv_nested_v2")
    d = load_windows(args.windows, expected_window=60, expected_metadata={"dataset": "afdb", "stride": 30})
    y, groups = d["y"], d["patient_id"]
    run_provenance = provenance([args.windows, Path(__file__)])
    snapshot_sources(out / "source_snapshot")
    features = featurize_windows(d["X"])
    print(f"[HRV] {len(y)} fenêtres, {len(np.unique(groups))} patients, {args.trials} essais/pli", flush=True)

    def progress(event):
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def checkpoint(k, model, detail, indices, scores, thresholds):
        joblib.dump(model, out / f"fold{k}.joblib")
        write_json(out / f"fold{k}_selection.json", detail)
        with (out / f"fold{k}_oof.npz").open("xb") as stream:
            np.savez(stream, indices=indices, scores=scores, thresholds=thresholds)
        print(f"[HRV pli {k}] terminé, seuil={detail['threshold']:.2f}", flush=True)

    with threadpool_limits(limits=4):
        result = nested_hrv_oof(features, y, groups, n_trials=args.trials, seed=args.seed,
                                progress_callback=progress, fold_callback=checkpoint)
    scores = result["scores"]
    decisions = (scores >= result["thresholds"]).astype(np.float32)
    metrics = {**ranking_metrics(y, scores), **threshold_metrics(y, scores),
               **per_patient_f1(y, scores, groups)}
    calibrated = {**threshold_metrics(y, decisions), **per_patient_f1(y, decisions, groups)}
    calibrated.pop("threshold")
    with (out / "oof_scores.npz").open("xb") as stream:
        np.savez(stream, y=y, groups=groups, scores=scores, fold=result["fold"], thresholds=result["thresholds"])
    write_json(out / "metrics.json", {
        "protocol": "nested_hrv_cv_v2", "dataset": "afdb", "feature_version": FEATURE_VERSION,
        "window": 60, "folds": result["details"], "metrics": metrics, "calibrated_metrics": calibrated,
        "configuration": {"trials": args.trials, "seed": args.seed, "outer_splits": 5, "inner_splits": 3,
                          "objective": "calibrated_patient_f1", "early_stopping": False},
        "windows_sha256": file_digest(args.windows), "provenance": run_provenance,
        "limitations": ["Retrospective model development on previously explored public datasets."]})
    print(f"[HRV] F1 patient fixe={metrics['mean_patient_f1']:.4f}, "
          f"calibré={calibrated['mean_patient_f1']:.4f}; sorties : {out}", flush=True)


if __name__ == "__main__":
    main()
