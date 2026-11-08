"""Index the frozen v2 candidate for the web app, without changing its artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.data.artifacts import load_windows
from src.utils.artifacts import file_digest, write_json


def prepare(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Le catalogue existe déjà : {output}")
    root = root.resolve()
    artifact_root = root.parent
    manifest = json.loads((root / "release_candidate/manifest.json").read_text())
    evidence = json.loads((root / "release_candidate/external_comparison.json").read_text())
    references = {}

    def reference(path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        name = str(path.relative_to(artifact_root))
        digest = file_digest(path)
        if expected and digest != expected:
            raise ValueError(f"Empreinte incohérente : {name}")
        references[name] = digest
        return name

    external = {}
    for name, digest in evidence["source_sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            # Evidence records paths relative to the original checkout.
            suffix = path.parts[path.parts.index(root.name) + 1 :]
            path = root.joinpath(*suffix)
        reference(path, digest)
        if path.name == "eval_ltafdb.json":
            meta = json.loads(path.read_text())
            if meta.get("variant", "hrv") == "fp32":
                external["cnn"] = path
            elif "hrv" in str(path.parent):
                external["hrv"] = path
    if set(external) != {"cnn", "hrv"}:
        raise ValueError("Évaluations externes FP32 et HRV requises.")
    cnn = json.loads((root / "cnn_lstm/metrics.json").read_text())
    hrv = json.loads((root / "hrv/metrics.json").read_text())
    compression = json.loads((root / "compression_verified/metrics.json").read_text())
    for path in (
        root / "cnn_lstm/metrics.json",
        root / "hrv/metrics.json",
        root / "compression_verified/metrics.json",
        root / "release_candidate/manifest.json",
    ):
        reference(path)
    for fold in manifest["folds"]:
        for checkpoint in fold["checkpoints"].values():
            reference(root / checkpoint["path"], checkpoint["sha256"])

    datasets = {}
    for dataset in ("afdb", "ltafdb"):
        window = (
            root / "windows" if dataset == "ltafdb" else root.parent / "reliability_v1/windows"
        ) / f"{dataset}_w60_s30_majority_v1.npz"
        expected_digest = hrv["windows_sha256"]
        if dataset == "ltafdb":
            expected_digest = manifest["external_metrics"]["fp32_batch512"]["windows_sha256"]
        reference(window, expected_digest)
        cached = load_windows(
            window,
            expected_window=60,
            expected_metadata={
                "dataset": dataset,
                "stride": 30,
                "label_strategy": "majority",
                "min_rr": 0.3,
                "max_rr": 2.0,
            },
        )
        models = {}
        for model, training in (("cnn", cnn), ("hrv", hrv)):
            path = (
                root / ("cnn_lstm" if model == "cnn" else "hrv") / "oof_scores.npz"
                if dataset == "afdb"
                else external[model].parent / "scores.npz"
            )
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(archive["y"], cached["y"]) or not np.array_equal(
                    archive["groups"].astype(str), cached["patient_id"]
                ):
                    raise ValueError(f"Observations désalignées : {dataset}/{model}")
            meta = training if dataset == "afdb" else json.loads(external[model].read_text())
            models[model] = {
                "scores": reference(path),
                "fixed": meta["metrics"],
                "calibrated": meta[
                    "calibrated_metrics" if dataset == "afdb" else "calibrated_vote_metrics"
                ],
            }
        datasets[dataset] = {
            "windows": str(window.relative_to(artifact_root)),
            "models": models,
            "n_windows": len(cached["y"]),
            "n_patients": len(np.unique(cached["patient_id"])),
        }
    folds = copy.deepcopy(manifest["folds"])
    for fold in folds:
        for checkpoint in fold["checkpoints"].values():
            checkpoint["path"] = str((root / checkpoint["path"]).relative_to(artifact_root))
    catalog = {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "references": references,
        "datasets": datasets,
        "folds": folds,
        "compression": compression["variants"],
        "latency_configuration": compression["latency_configuration"],
        "protocol": manifest["training_protocol"],
        "limitations": manifest["limitations"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/optimization_v2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/interface_v1/catalog.json"))
    args = parser.parse_args()
    catalog = prepare(args.root, args.output)
    print(f"Catalogue créé : {args.output} ({len(catalog['references'])} fichiers référencés)")


if __name__ == "__main__":
    main()
