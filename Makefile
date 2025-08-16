SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "install        : Install dependencies into virtual environment."
	@echo "install-dev    : Install all, including dev, dependencies into virtual environment for local development."
	@echo "update         : Install new requriements into the virtual environment."
	@echo "test           : Run pytests."
	@echo "test-notebooks : Execute all notebooks in nbs/."
	@echo "coverage       : Run pytest with coverage report"


.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --all-extras --dev && \
	uv run pre-commit install

.PHONY: update
update:
	uv sync --reinstall

.PHONY: test
test:
	uv run pytest -vx tests

.PHONY: test-notebooks
test-notebooks:
	set -e; for notebook in nbs/*.ipynb; do \
		uv run jupyter execute "$$notebook"; \
	done

.PHONY: coverage
coverage:
	uv run pytest --cov=src --cov-report html tests
