# Figures audit — `reports/figures/`

> Snapshot of every committed PNG in `reports/figures/` with its current resolution
> and a recommended action for the thesis report.
>
> Generated 2026-05-28 from `file <png>`. ImageMagick (`identify`) was not
> installed at the time of the audit, so DPI metadata wasn't queried.
> Matplotlib defaults to 100 dpi → all current figures were exported as
> screen-resolution PNGs, **not** print-resolution PDFs. For a LaTeX report,
> the recommendation is to re-export at 300 dpi PDF (vector preferred where
> possible).

## Re-export pattern (matplotlib)

In each notebook, replace `plt.savefig("reports/figures/XX_name.png")` with a
dual save:

```python
fig.savefig("reports/figures/XX_name.pdf", bbox_inches="tight")
fig.savefig("reports/figures/XX_name.png", dpi=300, bbox_inches="tight")
```

PDF for LaTeX `\includegraphics`, 300 dpi PNG kept for the README and the
Streamlit app (Streamlit doesn't render PDF directly).

## Audit table

Resolution column shows pixel dimensions. **Action** legend:

- **keep** — already acceptable, no re-export needed (used only in the README / Streamlit)
- **re-export PDF + 300 dpi PNG** — used in the LaTeX report, regenerate from notebook
- **replace** — known issue (low resolution, cropping, outdated content) — fix in notebook

| Figure | Resolution (px) | Size | Source notebook | Action |
|---|---|---|---|---|
| `01_dataset_summary.png` | 1564 × 684 | 82 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `02_rr_distribution.png` | 1629 × 719 | 117 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `03_class_imbalance_per_patient.png` | 1748 × 847 | 107 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `03b_patient_afib_rate_hist.png` | 1041 × 598 | 46 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `04_poincare_afib_vs_nsr.png` | 1874 × 758 | 455 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `05_raw_rr_examples.png` | 1925 × 1212 | 268 KB | `01_eda.ipynb` | re-export PDF + 300 dpi PNG |
| `06_baselines_roc_pr.png` | 1631 × 789 | 194 KB | `02_baselines.ipynb` | re-export PDF + 300 dpi PNG |
| `07_baselines_summary_bars.png` | 1164 × 624 | 55 KB | `02_baselines.ipynb` | re-export PDF + 300 dpi PNG |
| `08_baselines_per_patient_f1.png` | 1164 × 682 | 44 KB | `02_baselines.ipynb` | re-export PDF + 300 dpi PNG |
| `09_hrv_feature_importance.png` | 1065 × 600 | 48 KB | `02_baselines.ipynb` | re-export PDF + 300 dpi PNG |
| `10_optuna_search.png` | 1645 × 719 | 108 KB | `03_ablation.ipynb` | re-export PDF + 300 dpi PNG |
| `11_phase3_comparison.png` | 1397 × 772 | 108 KB | `03_ablation.ipynb` | re-export PDF + 300 dpi PNG |
| `12_phase3_ablation.png` | 1164 × 763 | 70 KB | `03_ablation.ipynb` | re-export PDF + 300 dpi PNG |
| `13_learning_curves.png` | 1529 × 719 | 125 KB | `03_ablation.ipynb` | re-export PDF + 300 dpi PNG |
| `14_phase3_roc_pr.png` | 1629 × 787 | 121 KB | `03_ablation.ipynb` | re-export PDF + 300 dpi PNG |
| `15_phase35_optuna.png` | 1645 × 719 | 118 KB | `04_phase35_w60.ipynb` | re-export PDF + 300 dpi PNG |
| `16_phase35_comparison.png` | 1397 × 763 | 98 KB | `04_phase35_w60.ipynb` | re-export PDF + 300 dpi PNG |
| `17_phase35_learning_curves.png` | 1513 × 719 | 102 KB | `04_phase35_w60.ipynb` | re-export PDF + 300 dpi PNG |
| `18_phase35_roc_pr.png` | 1629 × 787 | 133 KB | `04_phase35_w60.ipynb` | re-export PDF + 300 dpi PNG |
| `16_phase4_latency_bar.png` | 1184 × 657 | 54 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `16_phase4_latency_vs_size.png` | 926 × 701 | 50 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `16_phase4_size_vs_f1.png` | 1006 × 705 | 62 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `17_phase4b_latency_bar.png` | 1184 × 657 | 54 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `17_phase4b_latency_vs_size.png` | 926 × 701 | 49 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `17_phase4b_size_vs_f1.png` | 933 × 705 | 61 KB | `05_phase4_compression.ipynb` | re-export PDF + 300 dpi PNG |
| `18_phase5_f1_bar.png` | 1634 × 655 | 51 KB | `06_phase5_cross_dataset.ipynb` | re-export PDF + 300 dpi PNG |
| `18_phase5_finetune_gain.png` | 1334 × 658 | 55 KB | `06_phase5_cross_dataset.ipynb` | re-export PDF + 300 dpi PNG |
| `18_phase5_perpatient_box.png` | 1184 × 657 | 37 KB | `06_phase5_cross_dataset.ipynb` | re-export PDF + 300 dpi PNG |

## Naming collision to fix

Three figure prefixes collide between Phase 3.5 and Phase 4 — the `16_` and
`17_` prefixes are reused. This is fine for the file system (no two files
have identical names) but confusing in the LaTeX report. When re-exporting,
suggest renaming the Phase 4 latency bundle to a `phase4_*` prefix (no
leading number) to avoid the visual collision with `16_phase35_*` /
`17_phase35_*`.

## Sandbox screenshots (Phase 6) — to capture manually

These five PNGs are referenced by `app/README.md` and the live demo section
of the main `README.md`. They do not exist yet — the user needs to capture
them from a running browser.

| Path | Page |
|---|---|
| `docs/img/sandbox_page1.png` | Home (`streamlit_app.py`) |
| `docs/img/sandbox_page2.png` | 🔬 Patient Inspector |
| `docs/img/sandbox_page3.png` | ⚔️ Model Showdown |
| `docs/img/sandbox_page4.png` | 📦 Compression Lab |
| `docs/img/sandbox_page5.png` | 🧪 What-If |

Recommended viewport: 1440 × 900, light theme, with at least one chart
rendered per page (browse the sidebar to populate state first).
