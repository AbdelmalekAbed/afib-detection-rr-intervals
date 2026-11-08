"""Évaluation — `python -m src.evaluate --config configs/train.yaml`.

Deux modes complémentaires.

**Sans ``--external``** : relit les scores hors-pli produits par ``src.train``
et en tire le rapport complet — métriques globales, métriques par patient,
matrice de confusion, courbes ROC et précision-rappel.

**Avec ``--external <base>``** : recharge les poids sauvegardés, applique le
modèle à une base jamais vue (typiquement LTAFDB) et rapporte les mêmes
métriques. Les scores des plis sont moyennés, ce qui revient à évaluer un
ensemble des cinq modèles.

Sorties : un nouveau sous-répertoire ``<in-dir>/evaluation_<date>/`` contenant
le JSON et les figures. Les paramètres propres à chaque pli sont relus depuis
le manifeste d'entraînement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.artifacts import load_windows
from src.data.build import build_windows
from src.models.compression import dynamic_quantize
from src.models.search import model_from_params
from src.train import build_model, load_config
from src.utils.artifacts import file_digest, new_run_dir, write_json
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.validation import predict, preprocess_windows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du CNN-LSTM entraîné.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--external", type=str, default=None,
                   help="Base externe à évaluer (ex. ltafdb). Sans cette option, "
                        "les scores hors-pli existants sont analysés.")
    p.add_argument("--in-dir", type=Path, default=None,
                   help="Répertoire produit par src.train (défaut : déduit de la configuration).")
    p.add_argument("--fig-dir", type=Path, default=None)
    p.add_argument("--windows", type=Path, help="Archive externe RR versionnée (facultatif).")
    p.add_argument("--variant", choices=["fp32", "int8_dynamic", "onnx_fp32"], default="fp32")
    p.add_argument("--compression-run", type=Path, help="Compression des mêmes modèles, pour INT8 ou ONNX.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def confusion_counts(y: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "VP": int(((y_pred == 1) & (y == 1)).sum()),
        "VN": int(((y_pred == 0) & (y == 0)).sum()),
        "FP": int(((y_pred == 1) & (y == 0)).sum()),
        "FN": int(((y_pred == 0) & (y == 1)).sum()),
    }


def patient_table(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                  threshold: float) -> list[dict]:
    """Détail par patient : charge de FA, effectif, F1 (ou None si non notable)."""
    from sklearn.metrics import f1_score

    rows = []
    for pid in np.unique(groups):
        m = groups == pid
        n_pos, n_tot = int(y[m].sum()), int(m.sum())
        scorable = 0 < n_pos < n_tot
        rows.append({
            "patient": str(pid),
            "n_windows": n_tot,
            "afib_burden": n_pos / n_tot,
            "f1": (float(f1_score(y[m], (scores[m] >= threshold).astype(int), zero_division=0))
                   if scorable else None),
        })
    return sorted(rows, key=lambda r: r["afib_burden"])


def plot_curves(y: np.ndarray, scores: np.ndarray, label: str, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, scores)
    ax1.plot(fpr, tpr, lw=1.8, color="#1f77b4")
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
    ax1.set_xlabel("1 − spécificité")
    ax1.set_ylabel("sensibilité")
    ax1.set_title(f"Courbe ROC — {label}")
    ax1.grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y, scores)
    ax2.plot(rec, prec, lw=1.8, color="#d62728")
    ax2.axhline(y.mean(), ls="--", lw=1, color="#888888", label="classifieur aléatoire")
    ax2.set_xlabel("rappel")
    ax2.set_ylabel("précision")
    ax2.set_title(f"Courbe précision-rappel — {label}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"eval_{label}_roc_pr.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {fig_dir}/eval_{label}_roc_pr.{{png,pdf}}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("La taille de lot et le nombre de threads doivent être positifs.")
    cfg = load_config(args.config)
    threshold = cfg["evaluation"]["decision_threshold"]
    win_cfg = cfg["data"]["windowing"]
    pre_cfg = cfg["data"]["preprocessing"]

    if args.in_dir is None:
        candidates = sorted(p for p in Path(cfg["logging"]["out_dir"]).glob("cnn_lstm_nested_v1_*")
                            if (p / "metrics.json").is_file())
        if not candidates:
            raise FileNotFoundError("Aucune expérience corrigée complète : lancer make train.")
        in_dir = candidates[-1]
    else:
        in_dir = args.in_dir
    if not in_dir.is_dir():
        raise FileNotFoundError(
            f"{in_dir} est introuvable — lancer d'abord "
            f"`python -m src.train --config {args.config}`."
        )

    metadata = json.loads((in_dir / "metrics.json").read_text())
    if metadata.get("protocol") != "nested_patient_cv_v1":
        raise ValueError("Ce chargeur attend une expérience nested_patient_cv_v1.")
    if metadata["window"] != win_cfg["window_size"]:
        raise ValueError("La configuration et les poids n'utilisent pas la même fenêtre.")
    if cfg["data"]["windowing"] != metadata["config"]["data"]["windowing"] or pre_cfg != metadata["config"]["data"]["preprocessing"]:
        raise ValueError("Le prétraitement doit correspondre à celui de l'entraînement sauvegardé.")
    threshold = metadata["config"]["evaluation"]["decision_threshold"]
    torch.set_num_threads(args.threads)
    if args.external == metadata["dataset"]:
        raise ValueError("La base d'entraînement ne constitue pas une évaluation externe.")
    if args.variant != "fp32":
        if args.external is None or args.compression_run is None:
            raise ValueError("Une variante compressée exige une évaluation externe et --compression-run.")
        compressed = json.loads((args.compression_run / "metrics.json").read_text())
        expected_digest = next((digest for path, digest in compressed["provenance"]["source_sha256"].items()
                                if Path(path).name == "metrics.json"), None)
        if compressed["protocol"] != "frozen_nested_models_compression_v2" or expected_digest != file_digest(in_dir / "metrics.json"):
            raise ValueError("La compression ne correspond pas à cette expérience d'entraînement.")
    out_dir = new_run_dir(in_dir, f"evaluation_{args.variant}")

    if args.external is None:
        label = "interne"
        npz = np.load(in_dir / "oof_scores.npz", allow_pickle=False)
        y, groups, scores = npz["y"].astype(int), npz["groups"].astype(str), npz["scores"]
        print(f"[interne] scores hors-pli relus depuis {in_dir}/oof_scores.npz")
    else:
        label = args.external
        root = args.config.resolve().parent.parent
        if args.windows:
            cached = load_windows(args.windows, expected_window=win_cfg["window_size"], expected_metadata={
                "dataset": args.external, "stride": win_cfg["stride"], "label_strategy": win_cfg["label_strategy"],
                "min_rr": pre_cfg["clip_rr_seconds"]["min"], "max_rr": pre_cfg["clip_rr_seconds"]["max"]})
            X, y, groups = cached["X"], cached["y"], cached["patient_id"]
        else:
            X, y, groups = build_windows(
                root / cfg["data"]["paths"]["raw_dir"] / args.external,
                window=win_cfg["window_size"], stride=win_cfg["stride"],
                label_strategy=win_cfg["label_strategy"],
                min_rr=pre_cfg["clip_rr_seconds"]["min"], max_rr=pre_cfg["clip_rr_seconds"]["max"],
                verbose=True,
            )
        y = y.astype(int)
        ckpts = sorted(in_dir.glob("fold*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"aucun poids fold*.pt dans {in_dir}")
        if len(ckpts) != len(metadata["folds"]):
            raise ValueError("Nombre de checkpoints incompatible avec le manifeste.")
        print(f"[externe] {args.external} : X={X.shape}, moyenne de {len(ckpts)} modèles")

        acc = np.zeros(len(y), dtype=np.float64)
        votes = np.zeros(len(y), dtype=np.int16)
        for ckpt in ckpts:
            fold_id = int(ckpt.stem.removeprefix("fold"))
            params = metadata["folds"][fold_id]["params"]
            model = (model_from_params(params, win_cfg["window_size"]) if params else
                     build_model(metadata["config"]["model"], win_cfg["window_size"]))
            mode = metadata["folds"][fold_id].get("normalization", "zscore_per_window")
            transformed = preprocess_windows(X, mode)
            if args.variant == "onnx_fp32":
                import onnxruntime as ort
                from scipy.special import expit
                options = ort.SessionOptions()
                options.intra_op_num_threads = args.threads
                session = ort.InferenceSession(str(args.compression_run / f"fold{fold_id}.onnx"),
                                               sess_options=options, providers=["CPUExecutionProvider"])
                fold_scores = np.empty(len(y), dtype=np.float32)
                for start in range(0, len(y), args.batch_size):
                    fold_scores[start:start + args.batch_size] = expit(session.run(None, {"rr_window": transformed[start:start + args.batch_size]})[0])
            elif args.variant == "int8_dynamic":
                model = dynamic_quantize(model.eval())
                # Local checkpoint generated and verified by compress_optimized.
                model.load_state_dict(torch.load(args.compression_run / f"fold{fold_id}_int8.pt",
                                                 map_location="cpu", weights_only=False))
                fold_scores = predict(model, transformed, args.batch_size)
            else:
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                fold_scores = predict(model, transformed, args.batch_size)
            acc += fold_scores
            votes += fold_scores >= metadata["folds"][fold_id]["threshold"]
        scores = (acc / len(ckpts)).astype(np.float32)
        vote_fraction = votes.astype(np.float32) / len(ckpts)

    metrics = {
        **ranking_metrics(y, scores),
        **threshold_metrics(y, scores, threshold),
        **per_patient_f1(y, scores, groups, threshold),
    }
    counts = confusion_counts(y, (scores >= threshold).astype(int))
    rows = patient_table(y, scores, groups, threshold)

    print(f"\n=== Rapport d'évaluation — {label} ===")
    print(f"  fenêtres             : {len(y)}  (charge de FA {y.mean():.3f})")
    print(f"  patients             : {len(np.unique(groups))} "
          f"dont {metrics['n_patients_scored']} notables")
    print(f"  VP={counts['VP']}  VN={counts['VN']}  FP={counts['FP']}  FN={counts['FN']}")
    print(f"  AUROC                : {metrics['auroc']:.4f}")
    print(f"  AUPRC                : {metrics['auprc']:.4f}")
    print(f"  F1 par fenêtre       : {metrics['f1']:.4f}")
    print(f"  sensibilité / spéc.  : {metrics['sensitivity']:.4f} / {metrics['specificity']:.4f}")
    print(f"  F1 moyen par patient : {metrics['mean_patient_f1']:.4f}")

    worst = [r for r in rows if r["f1"] is not None][:5]
    print("\n  5 patients les plus difficiles (charge de FA croissante) :")
    for r in worst:
        print(f"    {r['patient']:>8}  charge={r['afib_burden']:6.3f}  "
              f"fenêtres={r['n_windows']:5d}  F1={r['f1']:.3f}")

    out = {"label": label, "threshold": threshold, "confusion": counts,
           "metrics": metrics, "per_patient": rows}
    out["training_run"] = str(in_dir)
    out["variant"] = args.variant
    out["runtime"] = {"batch_size": args.batch_size, "threads": args.threads}
    if args.windows:
        out["windows_sha256"] = file_digest(args.windows)
    if args.external is None:
        decisions = (scores >= npz["thresholds"]).astype(np.float32)
        out["calibrated_metrics"] = {**threshold_metrics(y, decisions),
                                     **per_patient_f1(y, decisions, groups)}
    else:
        out["calibrated_vote_metrics"] = {**threshold_metrics(y, vote_fraction),
                                           **per_patient_f1(y, vote_fraction, groups)}
        out["calibrated_vote_policy"] = "Majority of five classifiers using AFDB inner thresholds; no LTAFDB tuning."
    with (out_dir / "scores.npz").open("xb") as stream:
        np.savez(stream, y=y, groups=groups.astype(str), scores=scores,
                 calibrated_decisions=decisions if args.external is None else (vote_fraction >= .5).astype(np.int8))
    out_path = out_dir / f"eval_{label}.json"
    write_json(out_path, out)
    print(f"\n[sauvegarde] {out_path}")

    plot_curves(y, scores, label, args.fig_dir or out_dir / "figures")


if __name__ == "__main__":
    main()
