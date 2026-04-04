VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
DVC = $(VENV)/bin/dvc

.PHONY: help setup install pipeline extract preprocess train mlflow-up mlflow-down clean

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make install     - Install/update dependencies only"
	@echo "  make pipeline    - Run full DVC pipeline (extract -> preprocess -> train)"
	@echo "  make extract     - Run DVC extract stage only"
	@echo "  make preprocess  - Run DVC preprocess stage only"
	@echo "  make train       - Run DVC train stage only"
	@echo "  make mlflow-up   - Start MLflow tracking server (Docker)"
	@echo "  make mlflow-down - Stop MLflow tracking server"
	@echo "  make clean       - Remove generated artifacts (models/, metrics/)"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

pipeline:
	$(DVC) repro

extract:
	$(DVC) repro extract

preprocess:
	$(DVC) repro preprocess

train:
	$(DVC) repro train

mlflow-up:
	docker compose up -d
	@echo "MLflow UI available at http://localhost:5050"

mlflow-down:
	docker compose down

clean:
	rm -rf models/*.joblib metrics/metrics.json
