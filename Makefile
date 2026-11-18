.PHONY: check lint typecheck test format

# O Python do .venv, para o make funcionar mesmo sem o ambiente ativado no terminal.
PYTHON ?= .venv/bin/python

# Tudo o que precisa passar antes de entregar ou commitar.
check: lint typecheck test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
