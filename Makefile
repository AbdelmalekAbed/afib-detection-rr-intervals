.PHONY: help setup data figures train eval demo test lint format clean report-pdf report-clean report-watch

PYTHON_VERSION ?= 3.12
VENV           ?= .venv
PY             := $(VENV)/bin/python
UV             := uv
VENV_RUN       := VIRTUAL_ENV=$(VENV) $(UV) pip install

help:
	@echo "Targets:"
	@echo "  setup    - create venv and install project (CPU PyTorch) via uv"
	@echo "  data     - download PhysioNet datasets and build processed splits"
	@echo "  figures  - regenerate the phase 1 EDA figures and their PDF versions"
	@echo "  train    - train the default CNN-LSTM model (configs/train.yaml)"
	@echo "  eval     - evaluate on internal + external test sets"
	@echo "  demo     - launch the Streamlit demo"
	@echo "  test     - run unit tests"
	@echo "  lint     - run ruff checks"
	@echo "  format   - run black + ruff --fix"
	@echo "  clean    - remove caches and build artifacts"
	@echo ""
	@echo "Report (LaTeX):"
	@echo "  report-pdf   - compile report/main.tex → report/main.pdf (needs texlive + biber)"
	@echo "  report-watch - rebuild on file change (latexmk -pvc)"
	@echo "  report-clean - remove LaTeX intermediates (report/build/)"

setup:
	$(UV) venv $(VENV) --python $(PYTHON_VERSION)
	$(VENV_RUN) --index-url https://download.pytorch.org/whl/cpu torch
	$(VENV_RUN) -e ".[dev,demo]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

data:
	$(PY) -m scripts.download_data --config configs/data.yaml
	$(PY) -m scripts.build_windows  --config configs/data.yaml

figures:
	$(PY) -m scripts.eda_figures
	$(PY) scripts/png_to_pdf.py

train:
	$(PY) -m src.train --config configs/train.yaml

eval:
	$(PY) -m src.evaluate --config configs/train.yaml --external ltafdb

demo:
	$(VENV)/bin/streamlit run app/streamlit_app.py

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src tests

format:
	$(VENV)/bin/black src tests
	$(VENV)/bin/ruff check --fix src tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

# -----------------------------------------------------------------------------
# Report (LaTeX) — needs texlive-{latex,fonts,bibtex,science,pictures} + biber
# -----------------------------------------------------------------------------
report-pdf:
	cd report && latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex
	cp report/build/main.pdf report/main.pdf
	@echo "PDF: report/main.pdf"

report-watch:
	cd report && latexmk -pdf -pvc -outdir=build -interaction=nonstopmode main.tex

report-clean:
	rm -rf report/build report/main.pdf
	# Les intermédiaires laissés à la racine de report/ par une compilation
	# manuelle font échouer latexmk (il lance bibtex au lieu de biber en
	# trouvant un main.bbl périmé). On les supprime aussi.
	rm -f report/main.aux report/main.bbl report/main.bcf report/main.blg \
	      report/main.fdb_latexmk report/main.fls report/main.lof \
	      report/main.log report/main.lot report/main.out \
	      report/main.run.xml report/main.toc
