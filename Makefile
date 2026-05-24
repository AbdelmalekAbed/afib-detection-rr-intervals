.PHONY: help setup data train eval demo test lint format clean

PYTHON_VERSION ?= 3.12
VENV           ?= .venv
PY             := $(VENV)/bin/python
UV             := uv
VENV_RUN       := VIRTUAL_ENV=$(VENV) $(UV) pip install

help:
	@echo "Targets:"
	@echo "  setup    - create venv and install project (CPU PyTorch) via uv"
	@echo "  data     - download PhysioNet datasets and build processed splits"
	@echo "  train    - train the default CNN-LSTM model (configs/train.yaml)"
	@echo "  eval     - evaluate on internal + external test sets"
	@echo "  demo     - launch the Streamlit demo"
	@echo "  test     - run unit tests"
	@echo "  lint     - run ruff checks"
	@echo "  format   - run black + ruff --fix"
	@echo "  clean    - remove caches and build artifacts"

setup:
	$(UV) venv $(VENV) --python $(PYTHON_VERSION)
	$(VENV_RUN) --index-url https://download.pytorch.org/whl/cpu torch
	$(VENV_RUN) -e ".[dev,demo]"
	@echo "Done. Activate with: source $(VENV)/bin/activate"

data:
	$(PY) -m scripts.download_data --config configs/data.yaml
	$(PY) -m scripts.build_windows  --config configs/data.yaml

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
