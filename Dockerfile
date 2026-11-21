# Imagem da API do nano-review (Etapa 4.5): o modelo final e a página de demonstração, prontos para
# rodar em qualquer máquina com Docker, sem .venv, DVC nem dados.
# Versão exata da imagem base: "3.12-slim" muda sozinha quando sai um Python ou um Debian novo.
FROM python:3.12.16-slim-trixie

# O Python não grava .pyc enquanto roda (o usuário app nem poderia), a saída não tem buffer (o
# log aparece na hora) e o pip não guarda cache (imagem menor).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Roda sem privilégios de root: se a API tiver uma falha, ela não controla o contêiner inteiro.
# Fica aqui em cima porque nunca muda, e as camadas de baixo reaproveitam esta.
RUN useradd --create-home --uid 1000 app

# As camadas vão do que muda menos para o que muda mais: o Docker reaproveita as de cima enquanto
# elas não mudam, e um ajuste no código não reinstala o torch.

# 1. O torch CPU, na versão do .venv, pelo índice da PyTorch (o PyPI traria a versão com CUDA).
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2. O resto, com todas as versões fixas (diretas e indiretas): a imagem sai igual em qualquer
#    máquina e igual ao .venv em que o projeto foi testado.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 3. O modelo final, trazido para models/ antes do build (pelo DVC ou pelo Release v0.1.0).
COPY models/transformer.pt models/tokenizer.json ./models/

# 4. O pacote, numa instalação normal (não editável). O código-fonte é apagado em seguida: a API
#    roda do pacote instalado, o que prova que a página (HTML, CSS e JS) entrou nele.
COPY pyproject.toml ./
COPY src ./src
#    --no-build-isolation: usa o setuptools que já está na imagem, em vez de baixar o mais novo a
#    cada build.
RUN pip install --no-deps --no-build-isolation . && rm -rf src build

USER app

ENV NANO_REVIEW_CHECKPOINT=/app/models/transformer.pt \
    NANO_REVIEW_TOKENIZER=/app/models/tokenizer.json

EXPOSE 8000

# O Docker pergunta ao /health se a API está de pé. Antes de o modelo carregar, a conexão é recusada,
# e isso também conta como "fora do ar". O start-period dá tempo para o carregamento.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

# 0.0.0.0 só dentro do contêiner: quem decide o que fica visível fora é a porta publicada no
# docker run (o make docker-run publica só em 127.0.0.1).
CMD ["uvicorn", "nano_review.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
