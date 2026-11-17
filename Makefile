.PHONY: check lint typecheck test format

# Tudo o que precisa passar antes de entregar ou commitar.
check: lint typecheck test

lint:
	python -m ruff check .
	python -m ruff format --check .

typecheck:
	python -m mypy src tests

test:
	python -m pytest

format:
	python -m ruff format .
	python -m ruff check --fix .
