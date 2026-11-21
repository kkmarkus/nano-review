.PHONY: check lint typecheck test format docker-build docker-run

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

# Imagem Docker da API (Etapa 4.5). O modelo precisa estar em models/ antes do build: dvc pull
# ou o download do Release v0.1.0 (README).
IMAGE ?= nano-review:0.1
# Porta desta máquina onde a API aparece. Se a 8000 estiver ocupada: make docker-run PORT=8001
PORT ?= 8000

docker-build:
	docker build -t $(IMAGE) .

# A porta é publicada só em 127.0.0.1: a API não tem autenticação, então fica visível só nesta máquina.
docker-run:
	docker run --rm --name nano-review -p 127.0.0.1:$(PORT):8000 $(IMAGE)
