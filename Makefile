VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
DVC = $(VENV)/bin/dvc

.PHONY: help setup install pipeline extract preprocess train stack-up stack-down stack-logs stack-ps restart-api clean

help:
	@echo "Setup:"
	@echo "  make setup        - Create .venv and install dependencies"
	@echo "  make install      - Reinstall deps into existing .venv"
	@echo ""
	@echo "Training (logs run + registers model version on DagsHub MLflow):"
	@echo "  make pipeline     - Full DVC pipeline (extract -> preprocess -> train)"
	@echo "  make extract      - DVC extract stage only (downloads Kaggle dataset)"
	@echo "  make preprocess   - DVC preprocess stage only"
	@echo "  make train        - DVC train stage only (most common manual run)"
	@echo ""
	@echo "Stack (Docker — API + Web):"
	@echo "  make stack-up     - Build and start api + web containers"
	@echo "  make stack-down   - Stop and remove containers"
	@echo "  make stack-ps     - List running containers"
	@echo "  make stack-logs   - Tail API logs"
	@echo "  make restart-api  - Restart api container (re-pulls latest model from DagsHub)"
	@echo ""
	@echo "Other:"
	@echo "  make clean        - Remove generated artifacts (models/, metrics/)"

# ───────────────────────────── Setup ────────────────────────────────
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-ml.txt

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-ml.txt

# ───────────────────────────── Training ─────────────────────────────
pipeline:
	$(DVC) repro

extract:
	$(DVC) repro extract

preprocess:
	$(DVC) repro preprocess

train:
	$(DVC) repro train

# ───────────────────────────── Stack ────────────────────────────────
stack-up:
	docker compose up -d --build
	@echo ""
	@echo "Stack up:"
	@echo "  Frontend:  http://localhost:8501"
	@echo "  API docs:  http://localhost:8000/docs"
	@echo "  Health:    http://localhost:8000/health"
	@echo "  DagsHub:   https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow"

stack-down:
	docker compose down

stack-ps:
	docker compose ps

stack-logs:
	docker compose logs -f api

restart-api:
	docker compose restart api
	@echo "API restarted. On startup it pulls 'models:/gaming-mental-health/latest' from DagsHub."

# ───────────────────────────── Misc ─────────────────────────────────
clean:
	rm -rf models/*.joblib metrics/metrics.json
