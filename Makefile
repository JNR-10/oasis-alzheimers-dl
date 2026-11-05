# Reproducible entrypoints. Every experiment in the report should be runnable
# from a line in this file. `make help` lists what exists.

.DEFAULT_GOAL := help
PYTHON ?= python
EXP    ?= baseline_single_scan

.PHONY: help setup install lint format typecheck test check cohort preprocess train evaluate ablation report clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------

setup:  ## Create a virtualenv and install the package with dev extras
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev,notebooks]"
	.venv/bin/pre-commit install

install:  ## Install the package in editable mode into the active environment
	$(PYTHON) -m pip install -e ".[dev]"

# --- quality gates ---------------------------------------------------------

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Format with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck:  ## Type-check with mypy
	mypy

test:  ## Run tests that need no cohort and no GPU
	pytest -m "not requires_data and not requires_gpu"

check: lint typecheck test  ## Run every gate CI runs

# --- pipeline --------------------------------------------------------------

cohort:  ## Build the subject-level cohort manifest and partitions
	$(PYTHON) -m oasis_dl.cli.build_cohort

preprocess:  ## Run the MRI preprocessing pipeline over the cohort
	$(PYTHON) -m oasis_dl.cli.preprocess

train:  ## Train one experiment:  make train EXP=proposed_b_anatomy
	$(PYTHON) -m oasis_dl.cli.train experiment=$(EXP)

evaluate:  ## Evaluate one experiment on the held-out test subjects
	$(PYTHON) -m oasis_dl.cli.evaluate experiment=$(EXP)

ablation:  ## Run the two core ablations across all five experiments
	$(PYTHON) -m oasis_dl.cli.train --multirun \
		experiment=baseline_single_scan,proposed_a_longitudinal,proposed_b_anatomy

report:  ## Regenerate result tables and figures from recorded runs
	$(PYTHON) -m oasis_dl.cli.evaluate --multirun experiment='glob(*)'

# --- housekeeping ----------------------------------------------------------

clean:  ## Remove caches and build artifacts (never touches data/ or results/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info outputs multirun
