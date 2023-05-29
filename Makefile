SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates the virtual environment in .venv."
	@echo "install : install dependencies into virtual environment."
	@echo "compile : update the environment requirements after changes to dependencies in pyproject.toml."
	@echo "update  : pip install new requriements into the virtual environment."
	@echo "test    : run pytests."

# create a virtual environment
.PHONY: venv
venv:
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install pip==23.0.1 setuptools==67.6.1 wheel==0.40.0 && \
	pip install pip-tools==6.12.3

# ==============================================================================
# install requirements
# ==============================================================================

req-file := config/requirements.txt


# environment for production
.PHONY: install
install: venv
	source .venv/bin/activate && \
	pip-sync $(req-file) && \
	pip install -e . && \
	pre-commit install

# ==============================================================================
# compile requirements
# ==============================================================================

.PHONY: compile
compile:
	source .venv/bin/activate && \
	pip-compile pyproject.toml -o $(req-file) --resolver=backtracking

# ==============================================================================
# update requirements and virtual env
# ==============================================================================

.PHONY: update
update:
	source .venv/bin/activate && \
	pip-sync $(req-file) && \
	pip install -e .

# ==============================================================================
# run tests
# ==============================================================================

.PHONY: test
test:
	source .venv/bin/activate && \
	pytest -vx .
