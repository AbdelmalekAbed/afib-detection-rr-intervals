"""Compare frozen experiments on aligned patient predictions, without fitting models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.utils.artifacts import file_digest, new_run_dir, write_json
from src.utils.metrics import ranking_metrics, threshold_metrics


def patient_f1_values(y, decisions, groups):
    rows = []
    for pid in np.unique(groups):
        mask = groups == pid
        truth, pred = y[mask].astype(bool), decisions[mask].astype(bool)
        positives = int(truth.sum())
        if not 0 < positives < len(truth):
            continue
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        rows.append({"patient": str(pid), "f1": 2 * tp / (2 * tp + fp + fn),
                     "burden": positives / len(truth), "n_windows": len(truth)})
    return rows


def aligned(reference, candidate):
    for key in ("y", "groups", "fold"):
        if (key in reference or key in candidate) and (
            key not in reference or key not in candidate or not np.array_equal(reference[key], candidate[key])
        ):
            raise ValueError(f"Archives non alignées : {key}")


def load_scores(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def bootstrap(values, seed=42):
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.percentile(sampled, 2.5)),
            "ci95_high": float(np.percentile(sampled, 97.5)), "n_patients": len(values)}


def compare(cases, contrasts):
    reference = next(iter(cases.values()))[0]
    output, vectors, rows = {}, {}, []
    for name, (data, scores, decisions) in cases.items():
        aligned(reference, data)
        patients = patient_f1_values(data["y"], decisions, data["groups"])
        if not patients:
            raise ValueError("Aucun patient notable dans la comparaison.")
        values = np.array([r["f1"] for r in patients])
        vectors[name] = values
        low = [r["f1"] for r in patients if r["burden"] < .1]
        high = [r["f1"] for r in patients if r["burden"] >= .1]
        classification = threshold_metrics(data["y"], decisions.astype(float))
        classification.pop("threshold")
        output[name] = {**bootstrap(values), **ranking_metrics(data["y"], scores),
                        "classification": classification,
                        "low_burden": {"n": len(low), "mean_f1": float(np.mean(low)) if low else None},
                        "higher_burden": {"n": len(high), "mean_f1": float(np.mean(high)) if high else None}}
        rows.extend({"model": name, **r} for r in patients)
    paired = {}
    for left, right in contrasts:
        diff = vectors[left] - vectors[right]
        paired[f"{left} minus {right}"] = {**bootstrap(diff), "n_left_better": int((diff > 1e-12).sum())}
    return {"models": output, "paired_differences": paired}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/optimization_v2"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/reliability_v1"))
    args = parser.parse_args()
    out = new_run_dir(args.root, "summary")
    paths = [args.root / "cnn_lstm/oof_scores.npz", args.root / "hrv/oof_scores.npz",
             args.reference / "cnn_lstm_nested/oof_scores.npz", args.reference / "baselines/phase2_oof_scores.npz"]
    cnn, hrv, pilot, baselines = map(load_scores, paths)

    def fixed(d, key="scores"):
        return d, d[key], d[key] >= .5

    def calibrated(d):
        return d, d["scores"], d["scores"] >= d["thresholds"]

    cases = {"CNN-LSTM pilote calibré": calibrated(pilot), "CNN-LSTM optimisé fixe": fixed(cnn),
             "CNN-LSTM optimisé calibré": calibrated(cnn), "GB référence fixe": fixed(baselines, "gb_hist"),
             "GB optimisé fixe": fixed(hrv), "GB optimisé calibré": calibrated(hrv),
             "RF référence fixe": fixed(baselines, "hrv_rf")}
    result, rows = compare(cases, [("CNN-LSTM optimisé calibré", "CNN-LSTM pilote calibré"),
                                   ("CNN-LSTM optimisé calibré", "GB référence fixe"),
                                   ("GB optimisé calibré", "GB référence fixe")])
    result["uncertainty_scope"] = "Patient bootstrap conditional on frozen predictions; no model retraining, no multiplicity correction, retrospective exploration."
    result["source_sha256"] = {str(p): file_digest(p) for p in paths}
    write_json(out / "comparison.json", result)
    with (out / "per_patient.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "patient", "f1", "burden", "n_windows"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cases)
    means = [result["models"][name]["mean"] for name in names]
    low = [means[i] - result["models"][name]["ci95_low"] for i, name in enumerate(names)]
    high = [result["models"][name]["ci95_high"] - means[i] for i, name in enumerate(names)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(means, range(len(names)), xerr=[low, high], fmt="o", capsize=4)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 moyen par patient — IC 95 % conditionnel aux prédictions")
    ax.set_title("AFDB : comparaison sous protocoles corrigés")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"comparison.{ext}", dpi=300)
    plt.close(fig)
    for name, row in result["models"].items():
        print(f"{name}: F1={row['mean']:.4f}, IC95=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]", flush=True)
    print(out)


if __name__ == "__main__":
    main()
