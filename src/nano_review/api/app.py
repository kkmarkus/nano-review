"""API do produto (Etapa 4.4): nota de 1 a 5 entra, resenhas sintéticas saem.

Rotas:

- ``GET /health``: o modelo está carregado? Quantos parâmetros, qual checkpoint, quais padrões.
- ``POST /generate``: as resenhas para uma nota, com temperatura, top-k e semente opcionais.
- ``GET /``: a página de demonstração (``static/index.html``), que chama o ``/generate``. O
  estilo e o comportamento dela (``style.css`` e ``app.js``) são servidos em ``/static``.
- ``GET /docs``: a documentação automática do FastAPI, onde dá para testar cada rota.

O modelo carrega **uma vez**, quando a API sobe, e não a cada pedido: carregar 5 milhões de pesos
custa mais que gerar uma resenha. Os caminhos vêm das variáveis de ambiente
``NANO_REVIEW_CHECKPOINT`` e ``NANO_REVIEW_TOKENIZER`` (o padrão é ``models/...``), para o Docker
poder apontar para onde o modelo estiver.

Uso, na pasta do projeto::

    uvicorn nano_review.api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from nano_review.inference.generator import MAX_SEED, GenerationError, ReviewGenerator, Settings

LOGGER = logging.getLogger("nano_review.api")
STATIC = Path(__file__).parent / "static"
DEFAULTS = Settings()
# A menor temperatura aceita. Abaixo disso, a divisão dos logits estoura o float32 e o sorteio
# quebra (um 1e-40 virava erro 500); 0,1 já é bem mais "seguro" que qualquer valor útil.
MIN_TEMPERATURE = 0.1
# A maior semente que a API sorteia: 2^53 - 1, o maior inteiro que o JavaScript guarda sem perder
# precisão. Acima disso, a página mostraria uma semente arredondada, que não repete o texto.
MAX_DRAWN_SEED = 2**53 - 1
# A seção de ética do charter, aplicada ao produto: toda resposta diz o que o texto é.
NOTICE = "Texto sintético para testar software. Não publique como avaliação de um cliente real."


class GenerateRequest(BaseModel):
    """O pedido. ``strict``: ``"5"`` ou ``true`` não viram nota; ``extra``: campo errado é erro."""

    model_config = ConfigDict(strict=True, extra="forbid")

    rating: int = Field(ge=1, le=5, description="A nota pedida, de 1 a 5.")
    n: int = Field(default=1, ge=1, le=DEFAULTS.max_reviews, description="Quantas resenhas.")
    temperature: float | None = Field(
        default=None,
        ge=MIN_TEMPERATURE,
        le=DEFAULTS.max_temperature,
        description="Omitida: 1,0. Abaixo de 1, o texto fica mais seguro e menos variado.",
    )
    top_k: int | None = Field(
        default=None, ge=0, description="Omitido: sem top-k. 0 pede explicitamente sem top-k."
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED,
        description="Com semente, o mesmo pedido repete o texto. Omitida: uma é sorteada.",
    )


class ReviewOut(BaseModel):
    rating: int
    text: str
    attempts: int


class GenerateResponse(BaseModel):
    reviews: list[ReviewOut]
    temperature: float
    top_k: int | None
    # A semente usada: a pedida ou, se omitida, a sorteada. Com ela, um bom resultado aleatório
    # pode ser repetido depois.
    seed: int
    seconds: float
    synthetic: bool = True
    notice: str = NOTICE


class HealthResponse(BaseModel):
    status: str
    parameters: int
    vocab_size: int
    checkpoint_md5: str | None
    defaults: dict[str, object]


@dataclass(frozen=True)
class Loaded:
    """O gerador já carregado e o hash do checkpoint, para o ``/health`` dizer qual modelo é."""

    generator: ReviewGenerator
    checkpoint_md5: str | None


def load_from_env() -> Loaded:
    """Carrega o checkpoint e o tokenizador dos caminhos das variáveis de ambiente."""
    checkpoint = Path(os.environ.get("NANO_REVIEW_CHECKPOINT", "models/transformer.pt"))
    tokenizer = Path(os.environ.get("NANO_REVIEW_TOKENIZER", "models/tokenizer.json"))
    generator = ReviewGenerator.from_files(checkpoint, tokenizer)
    return Loaded(generator, hashlib.md5(checkpoint.read_bytes()).hexdigest())


def resolved_decoding(
    settings: Settings, temperature: float | None, top_k: int | None
) -> tuple[float, int | None]:
    """A decodificação que o gerador vai usar, com a mesma regra de ``ReviewGenerator.generate``.

    Omitido vira o padrão do gerador carregado, e top-k 0 vira "sem top-k".
    """
    used_temperature = settings.temperature if temperature is None else temperature
    if top_k is None:
        return used_temperature, settings.top_k
    return used_temperature, None if top_k == 0 else top_k


def validation_error(request: Request, error: Exception) -> JSONResponse:
    """O 422 do pydantic, sem ecoar o valor recebido.

    O FastAPI devolve, por padrão, o valor que falhou. Um ``NaN`` ou ``Infinity`` no corpo (o JSON
    do Python aceita) não cabe numa resposta JSON, e o próprio 422 quebrava com erro 500.
    """
    assert isinstance(error, RequestValidationError)
    detail = [{key: e[key] for key in ("type", "loc", "msg")} for e in error.errors()]
    LOGGER.warning("pedido inválido: %s", detail)
    return JSONResponse(status_code=422, content={"detail": detail})


def create_app(load: Callable[[], Loaded] = load_from_env) -> FastAPI:
    """Monta a API. Os testes passam um ``load`` com um modelo minúsculo, sem o checkpoint real."""
    state: dict[str, Loaded] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # O uvicorn configura só os loggers dele; sem isto, as mensagens da API não aparecem.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
        state["loaded"] = load()
        LOGGER.info("modelo carregado: %s", state["loaded"].checkpoint_md5)
        yield
        state.clear()

    app = FastAPI(
        title="nano-review",
        description=f"Resenhas sintéticas de produto a partir de uma nota de 1 a 5. {NOTICE}",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_error)

    def loaded() -> Loaded:
        if "loaded" not in state:
            raise HTTPException(status_code=503, detail="o modelo ainda não foi carregado")
        return state["loaded"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = loaded()
        settings = current.generator.settings  # os padrões do gerador carregado, e não os do código
        return HealthResponse(
            status="ok",
            parameters=sum(p.numel() for p in current.generator.model.parameters()),
            vocab_size=current.generator.tokenizer.get_vocab_size(),
            checkpoint_md5=current.checkpoint_md5,
            defaults={
                "temperature": settings.temperature,
                "top_k": settings.top_k,
                "max_reviews": settings.max_reviews,
                "max_text_tokens": settings.max_text_tokens,
            },
        )

    # ``def`` (e não ``async def``): gerar é trabalho de CPU, e o FastAPI roda funções comuns numa
    # thread separada, sem travar as outras rotas (o /health continua respondendo).
    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        generator = loaded().generator
        temperature, top_k = resolved_decoding(
            generator.settings, request.temperature, request.top_k
        )
        # Sem semente, a API sorteia uma e a devolve: o resultado continua aleatório, mas pode ser
        # repetido depois com a mesma semente.
        seed = secrets.randbelow(MAX_DRAWN_SEED + 1) if request.seed is None else request.seed
        try:  # a segunda validação, a do gerador (por exemplo, top-k maior que o vocabulário)
            generator.check_request(request.rating, request.n, temperature, top_k, seed)
        except ValueError as error:
            LOGGER.warning("pedido recusado pelo gerador: %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error
        start = time.perf_counter()
        try:
            reviews = generator.generate(
                request.rating, request.n, request.temperature, request.top_k, seed
            )
        except GenerationError as error:
            # Uma resenha que esgota as tentativas derruba o pedido inteiro: o produto não entrega
            # menos resenhas do que as pedidas sem avisar.
            LOGGER.warning("tentativas esgotadas: %s", error)
            raise HTTPException(
                status_code=503,
                detail={"message": str(error), "reasons": error.reasons},
            ) from error
        seconds = round(time.perf_counter() - start, 3)
        LOGGER.info("nota %d, %d resenhas, %.2f s", request.rating, request.n, seconds)
        return GenerateResponse(
            reviews=[ReviewOut(rating=r.rating, text=r.text, attempts=r.attempts) for r in reviews],
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            seconds=seconds,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    # O CSS e o JavaScript da página, cada um no seu arquivo (o HTML aponta para /static/...).
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app


app = create_app()
