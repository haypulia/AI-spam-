PYTHON ?= python3
COMPOSE ?= docker compose -f docker/docker-compose.yml

.PHONY: install install-dev test prepare-data train metrics analyze docker-build docker-metrics docker-analyze clean

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests

prepare-data:
	$(PYTHON) scripts/prepare_dataset.py

train:
	$(PYTHON) scripts/train_model.py

metrics:
	$(PYTHON) scripts/evaluate_metrics.py --split test

analyze:
	$(PYTHON) scripts/run_analysis.py

docker-build:
	$(COMPOSE) build

docker-metrics:
	$(COMPOSE) run --rm metrics

docker-analyze:
	$(COMPOSE) run --rm detector

clean:
	rm -rf data/interim/ai_assisted_spam data/processed/analysis_results
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
