"""External evaluation of a frozen nested HRV ensemble; never fits to external labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.artifacts import load_windows
from src.evaluate import patient_table, plot_curves
from src.features.hrv import FEATURE_VERSION, featurize_windows
from src.utils.artifacts import file_digest, new_run_dir, write_json
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / "metrics.json").read_text())
    if metadata["protocol"] != "nested_hrv_cv_v2" or metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Expérience HRV ou version des descripteurs incompatible.")
    d = load_windows(args.windows, expected_window=metadata["window"], expected_metadata={
        "dataset": "ltafdb", "stride": 30, "label_strategy": "majority", "min_rr": .3, "max_rr": 2.})
    out = new_run_dir(args.run, "evaluation_ltafdb")
    y, groups = d["y"], d["patient_id"]
    print(f"[HRV externe] {len(y)} fenêtres, {len(np.unique(groups))} patients", flush=True)
    features = featurize_windows(d["X"])
    summed = np.zeros(len(y), dtype=np.float64)
    votes = np.zeros(len(y), dtype=np.int16)
    with threadpool_limits(limits=4):
        for detail in metadata["folds"]:
            # Files created by tune_hrv in the local trusted run directory only.
            model = joblib.load(args.run / f"fold{detail['fold']}.joblib")
            scores = model.predict_proba(features)[:, 1]
            summed += scores
            votes += scores >= detail["threshold"]
    scores = (summed / len(metadata["folds"])).astype(np.float32)
    vote_fraction = votes.astype(np.float32) / len(metadata["folds"])
    result = {
        "label": "ltafdb", "training_run": str(args.run),
        "metrics": {**ranking_metrics(y, scores), **threshold_metrics(y, scores),
                    **per_patient_f1(y, scores, groups)},
        "calibrated_vote_metrics": {**threshold_metrics(y, vote_fraction),
                                    **per_patient_f1(y, vote_fraction, groups)},
        "vote_policy": "Majority using frozen AFDB inner thresholds; no LTAFDB tuning.",
        "per_patient": patient_table(y, scores, groups, .5),
        "windows_sha256": file_digest(args.windows),
    }
    write_json(out / "eval_ltafdb.json", result)
    with (out / "scores.npz").open("xb") as stream:
        np.savez(stream, y=y, groups=groups, scores=scores, calibrated_decisions=(vote_fraction >= .5).astype(np.int8))
    plot_curves(y, scores, "ltafdb_hrv", out / "figures")
    print(f"[HRV LTAFDB] F1 patient fixe={result['metrics']['mean_patient_f1']:.4f}, "
          f"vote calibré={result['calibrated_vote_metrics']['mean_patient_f1']:.4f}; {out}", flush=True)


if __name__ == "__main__":
    main()
