SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "install  : install dependencies into virtual environment."
	@echo "install-dev  : install all, including dev, dependencies into virtual environment for local development."
	@echo "update   : Install new requriements into the virtual environment."
	@echo "test     : Run pytests."
	@echo "coverage : Run pytest with coverage report"


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

.PHONY: coverage
coverage:
	uv run pytest --cov=src --cov-report html tests
