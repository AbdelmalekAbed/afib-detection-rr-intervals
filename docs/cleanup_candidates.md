# Cleanup candidates

> Generated 2026-05-28 from `git status --ignored` and a recursive scan of
> Python / Jupyter / build caches. **None of these have been deleted.** Review,
> then remove with `make clean` or the per-target commands below.

## What's already covered by `.gitignore`

`git status --ignored` confirms the following are ignored and will **not**
sneak into a commit:

- `.venv/`
- `.pytest_cache/`
- `afib_cnn_lstm.egg-info/`
- `app/__pycache__/`, `scripts/__pycache__/`, `src/**/__pycache__/`, `tests/__pycache__/`
- `data/` (raw + processed)
- `reports/checkpoints/` (model artifacts, multi-MB)
- `instructions report/` (local-only reference material)

## On-disk cleanup candidates

Total: about **220 KB** of caches + an empty Phase 6 page directive file. All
safe to remove; nothing is checked in.

### Python bytecode caches (~190 KB)

```bash
find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
```

| Path | Size |
|---|---|
| `scripts/__pycache__/` | 52 KB |
| `tests/__pycache__/` | 28 KB |
| `src/data/__pycache__/` | 28 KB |
| `app/__pycache__/` | 24 KB |
| `src/baselines/__pycache__/` | 24 KB |
| `src/models/__pycache__/` | 20 KB |
| `src/utils/__pycache__/` | 20 KB |
| `src/features/__pycache__/` | 12 KB |
| `src/__pycache__/` | 8 KB |

### Build / test caches (~56 KB)

```bash
rm -rf .pytest_cache/ afib_cnn_lstm.egg-info/
```

| Path | Size | Notes |
|---|---|---|
| `.pytest_cache/` | 28 KB | regenerated on next `pytest` run |
| `afib_cnn_lstm.egg-info/` | 28 KB | regenerated on `pip install -e .` |

### Jupyter checkpoints

None found at the time of the scan — `find . -type d -name .ipynb_checkpoints
-not -path './.venv/*'` returned empty. Good.

## One-shot cleanup (already in the Makefile)

```bash
make clean
```

This runs:

```
rm -rf build dist *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type d -name .pytest_cache -exec rm -rf {} +
find . -type d -name .ruff_cache -exec rm -rf {} +
```

…which covers everything above. No additional cleanup target needed.

## NOT cleanup candidates — keep these

- `reports/figures/` (2.9 MB of PNGs — the figures the jury will look at)
- `reports/checkpoints/` (6.7 MB of model artifacts — covered by `.gitignore`,
  not committed, but **needed locally** to run the Streamlit app)
- `data/` (covered by `.gitignore`, but rebuilding takes ~10 min on CPU)
- `.venv/` (covered by `.gitignore`, slow to rebuild)
