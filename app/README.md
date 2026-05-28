# AFib Sandbox — Streamlit demo

> Phase 6 of the AFib CNN-LSTM project. An interactive playground that lets a
> visitor *touch* every result of the thesis: pick a real patient, swap a 100k
> FP32 model for a 90 KB INT8 one, push the model into a regime it has never
> seen, or draw an RR series by hand and watch the probability move in real
> time.

---

## Live demo

> [TODO: Streamlit Cloud URL] — deploy from `app/streamlit_app.py`, working
> directory `.`, Python 3.11+, requirements `requirements.txt` plus
> `streamlit>=1.32` and `plotly>=5.20` (already pulled in by
> `pip install -e ".[demo]"`).

---

## What's inside — the 5 pages

### Home — `app/streamlit_app.py`
Project pitch in 30 seconds: the four headline KPIs (best F1/patient on AFDB
and LTAFDB, smallest model size, lowest CPU latency) plus the size-vs-F1
Pareto preview that connects to the Compression Lab.

### 1. Patient Inspector — `app/pages/1_🔬_Patient_Inspector.py`
Pick any AFDB or LTAFDB patient that appeared in the held-out folds. The page
shows window-level model probability vs. ground-truth AFib mask along the
recording timeline, plus per-patient F1, sensitivity, specificity and AFib
burden. Scrub the timeline to inspect specific decision points.

### 2. Model Showdown — `app/pages/2_⚔️_Model_Showdown.py`
Four compression variants (FP32, INT8 dynamic, prune-50, prune-70 +
fine-tune) scored on the *same* AFDB patient at the *same* time. Side-by-side
probability traces make the (small) accuracy cost of quantization concrete.

### 3. Compression Lab — `app/pages/3_📦_Compression_Lab.py`
Interactive Pareto frontier across two architectures (100k vs 30k params)
and seven compression variants. Hover any point for size / F1 / latency, run
a live single-window CPU latency benchmark (warm-up + 100 iters), and read
off the model that hits the <200 KB plan target.

### 4. Cross-Dataset — `app/pages/4_🌍_Cross_Dataset.py`
The Phase 5 robustness story: AFDB-trained model deployed zero-shot on
LTAFDB vs from-scratch LTAFDB 5-fold vs fine-tuned. Per-patient F1 boxplots
quantify the generalization gap, and the fine-tuning-gain bar shows where
adaptation actually helps.

### 5. What-If — `app/pages/5_🧪_What_If.py`
Build an RR series by mixing regular (NSR) and irregular (AFib-like)
segments — vary heart rate, jitter, run length. The Phase 5 AFDB source
model reacts live to the synthetic input. Useful to develop intuition for
what triggers a positive prediction.

---

## Launch locally

```bash
# from the repo root
pip install -e ".[demo]"          # or: pip install -r requirements.txt + streamlit + plotly
streamlit run app/streamlit_app.py
```

Streamlit will open at <http://localhost:8501>. The sidebar lists the five
pages in order.

You can also use the Makefile shortcut:

```bash
make demo
```

---

## Required artifacts

The app reads everything from `reports/`. The Phase scripts produce them; if
you cloned a fresh repo, run `make data` and the Phase 4 / 5 scripts before
launching.

### Results JSON / CSV — `reports/`

| File | Used by | Produced by |
|---|---|---|
| `reports/phase35_best_params.json` | Home, What-If | `notebooks/04_phase35_w60.ipynb` |
| `reports/phase3_results.json` | Model Showdown, Compression Lab | `notebooks/03_ablation.ipynb` |
| `reports/phase4_results.json` | Home, Compression Lab | `scripts/phase4_compression.py` |
| `reports/phase4b_results.json` | Home, Compression Lab | `scripts/phase4_compression.py --params-json reports/phase35_best_params.json --out-prefix phase4b --fig-prefix 17_phase4b --ckpt-tag p35` |
| `reports/phase5_results.json` | Home, Cross-Dataset | `scripts/phase5_cross_dataset.py` |
| `reports/phase4_compression.csv` | Home, Compression Lab | Phase 4 script |
| `reports/phase4b_compression.csv` | Home, Compression Lab | Phase 4b script |
| `reports/phase4_oof_scores.npz` | Patient Inspector, Model Showdown | Phase 4 script |
| `reports/phase4b_oof_scores.npz` | Compression Lab (optional) | Phase 4b script |
| `reports/phase5_oof_scores.npz` | Patient Inspector, Cross-Dataset | Phase 5 script |

### Model checkpoints — `reports/checkpoints/`

Not committed (covered by `.gitignore` rule `checkpoints/`). Regenerate with
the Phase 4 / 5 scripts. The app expects the following files:

| File | Used by | Notes |
|---|---|---|
| `phase5_source_afdb.pt` | What-If, Patient Inspector | Phase 5 AFDB source model (archi 30k) |
| `fold0_fp32.pt` | Model Showdown, Compression Lab | Phase 4, archi 100k |
| `fold0_int8_dynamic.pt` | Model Showdown, Compression Lab | Phase 4, INT8 dynamic |
| `fold0_prune30.pt` / `fold0_prune50.pt` / `fold0_prune70.pt` | Compression Lab | Phase 4, magnitude pruning |
| `fold0_prune50_int8.pt` / `fold0_prune70_finetune.pt` | Compression Lab | Phase 4 combos |
| `fold{0..4}_fp32.onnx` | Compression Lab | Phase 4 ONNX export |
| `fold0_*_p35.pt` / `fold{0..4}_fp32_p35.onnx` | Compression Lab (archi 30k variants) | Phase 4b |

The Phase 3.5 best hyperparameters file (`reports/phase35_best_params.json`)
defines the small architecture used for `phase5_source_afdb.pt`. Loading
falls back gracefully and the page surfaces a warning instead of crashing
when a file is missing.

---

## Screenshots

> Capture these locally with the app running and drop the PNGs in
> `docs/img/`. Recommended viewport: 1440 × 900, light theme.

![Home](../docs/img/sandbox_page1.png)
![Patient Inspector](../docs/img/sandbox_page2.png)
![Model Showdown](../docs/img/sandbox_page3.png)
![Compression Lab](../docs/img/sandbox_page4.png)
![What-If](../docs/img/sandbox_page5.png)

---

## Code layout

```
app/
├── streamlit_app.py     # Home page + entry point
├── utils.py             # shared loaders, model factory, caching, KPI widgets
└── pages/               # auto-discovered by Streamlit, ordered by filename prefix
    ├── 1_🔬_Patient_Inspector.py
    ├── 2_⚔️_Model_Showdown.py
    ├── 3_📦_Compression_Lab.py
    ├── 4_🌍_Cross_Dataset.py
    └── 5_🧪_What_If.py
```

`app/utils.py` exposes the cached data loaders (`load_results_json`,
`load_compression_csv`, `load_oof_npz`), the model factory (`make_model`),
the variant loader (`load_variant_model`), the live latency benchmark
(`benchmark_latency`), and the small UI helpers (`header`, `kpi_row`).

---

## Disclaimer

Research and education only. **Not a medical device.** Predictions are
illustrative and must not drive any clinical decision.
