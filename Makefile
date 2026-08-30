# Micromouse challenge — common commands
.DEFAULT_GOAL := help
UV ?= uv
VENV := .venv
PY := $(VENV)/bin/python
PORT ?= 8000

.PHONY: help venv install test test-watch cov lint fmt run race train dash \
        docker-build docker-up docker-down docker-test docker-sh clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the uv virtualenv
	$(UV) venv --python 3.13

install: venv ## Install project + all extras (editable)
	$(UV) pip install -e ".[rl,dash,dev]"

test: ## Run the test suite
	$(PY) -m pytest

test-watch: ## Re-run tests on file change
	$(PY) -m pytest -q --looponfail 2>/dev/null || \
		while true; do $(PY) -m pytest -q; sleep 2; done

cov: ## Test suite with coverage report
	$(PY) -m pytest --cov=micromouse --cov-report=term-missing

run: ## Run one bot on one maze: make run BOT=floodfill SEED=1
	$(PY) -m micromouse.cli run --bot $(or $(BOT),floodfill) --seed $(or $(SEED),1)

race: ## Race every bot over N mazes: make race MAZES=20
	$(PY) -m micromouse.cli race --mazes $(or $(MAZES),20)

train: ## Train a learning bot: make train BOT=dynaq EPISODES=2000
	$(PY) -m micromouse.cli train --bot $(or $(BOT),dynaq) --episodes $(or $(EPISODES),2000)

dash: ## Serve the dashboard on http://localhost:$(PORT)
	$(PY) -m micromouse.cli dash --port $(PORT)

docker-build: ## Build the docker image
	docker compose build

docker-up: ## Start the dashboard in docker
	docker compose up

docker-down: ## Stop and remove containers
	docker compose down -v

docker-test: ## Run the test suite inside docker
	docker compose run --rm --no-deps app pytest

docker-sh: ## Shell into the container
	docker compose run --rm --no-deps app bash

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
