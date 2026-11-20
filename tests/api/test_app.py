"""A API: health, geração, validação (422), tentativas esgotadas (503) e a página."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer

from nano_review.api.app import NOTICE, Loaded, create_app, load_from_env, resolved_decoding
from nano_review.inference.generator import GenerationError, ReviewGenerator, Settings
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.tokenization.bpe import train_tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return train_tokenizer(["Produto bom, chegou rápido.", "Não gostei do produto."] * 5, 80)


def _config(tokenizer: Tokenizer) -> TransformerConfig:
    return TransformerConfig(
        tokenizer.get_vocab_size(), block_size=24, d_model=16, n_layers=1, n_heads=2, dropout=0.0
    )


def _loaded(tokenizer: Tokenizer) -> Loaded:
    torch.manual_seed(0)
    model = TransformerLanguageModel(_config(tokenizer))
    # Filtros frouxos: o modelo minúsculo escreve qualquer coisa e esgotaria as 5 tentativas.
    generator = ReviewGenerator(model, tokenizer, Settings(max_text_tokens=20, max_attempts=50))
    return Loaded(generator, checkpoint_md5="abc123")


@pytest.fixture()
def client(tokenizer: Tokenizer) -> Iterator[TestClient]:
    # O "with" roda o lifespan: é ali que a API carrega o modelo.
    with TestClient(create_app(lambda: _loaded(tokenizer))) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client: TestClient, tokenizer: Tokenizer) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["checkpoint_md5"] == "abc123"
    assert body["vocab_size"] == tokenizer.get_vocab_size()
    expected = sum(p.numel() for p in TransformerLanguageModel(_config(tokenizer)).parameters())
    assert body["parameters"] == expected
    assert body["defaults"]["temperature"] == 1.0 and body["defaults"]["top_k"] is None
    # Os padrões vêm do gerador carregado (o do teste tem filtros frouxos), e não do código.
    assert body["defaults"]["max_text_tokens"] == 20


def test_without_the_model_loaded_health_and_generate_are_503(tokenizer: Tokenizer) -> None:
    # Sem o "with", o lifespan não roda e o modelo não carrega: as rotas precisam dizer isso.
    unloaded = TestClient(create_app(lambda: _loaded(tokenizer)))
    assert unloaded.get("/health").status_code == 503
    assert unloaded.post("/generate", json={"rating": 3}).status_code == 503


def test_generate_returns_n_reviews_for_the_rating(client: TestClient) -> None:
    response = client.post("/generate", json={"rating": 4, "n": 3, "seed": 11})
    assert response.status_code == 200
    body = response.json()
    assert [review["rating"] for review in body["reviews"]] == [4, 4, 4]
    assert all(review["text"] and review["attempts"] >= 1 for review in body["reviews"])
    assert body["synthetic"] is True and body["notice"] == NOTICE
    assert (body["temperature"], body["top_k"], body["seed"]) == (1.0, None, 11)


def test_without_a_seed_the_drawn_one_is_returned_and_repeats_the_texts(
    client: TestClient,
) -> None:
    first = client.post("/generate", json={"rating": 1, "n": 2}).json()
    # A semente sorteada cabe num número do JavaScript, para a página mostrá-la sem arredondar.
    assert isinstance(first["seed"], int) and 0 <= first["seed"] <= 2**53 - 1
    again = client.post("/generate", json={"rating": 1, "n": 2, "seed": first["seed"]}).json()
    assert again["reviews"] == first["reviews"]


def test_same_seed_same_reviews(client: TestClient) -> None:
    first = client.post("/generate", json={"rating": 2, "n": 2, "seed": 5}).json()["reviews"]
    again = client.post("/generate", json={"rating": 2, "n": 2, "seed": 5}).json()["reviews"]
    assert first == again


def test_the_response_says_which_decoding_was_used(client: TestClient) -> None:
    body = client.post("/generate", json={"rating": 5, "temperature": 0.5, "top_k": 7}).json()
    assert (body["temperature"], body["top_k"]) == (0.5, 7)
    assert client.post("/generate", json={"rating": 5, "top_k": 0}).json()["top_k"] is None
    # A página manda o número que o usuário digitou: "1" sai do JavaScript como inteiro.
    assert (
        client.post("/generate", json={"rating": 5, "temperature": 1}).json()["temperature"] == 1.0
    )


@pytest.mark.parametrize(
    "request_body",
    [
        {"rating": 0},
        {"rating": 6},
        {"rating": "5"},  # strict: texto não vira nota
        {"rating": True},  # strict: booleano não vira nota
        {"rating": 3.0},
        {"rating": 3, "n": 0},
        {"rating": 3, "n": 21},
        {"rating": 3, "temperature": 0},
        {"rating": 3, "temperature": 1e-40},  # quebrava o sorteio (erro 500)
        {"rating": 3, "temperature": 0.05},  # abaixo do mínimo de 0,1
        {"rating": 3, "temperature": 2.5},
        {"rating": 3, "top_k": -1},
        {"rating": 3, "seed": -1},
        {"rating": 3, "cor": "azul"},  # campo que não existe
        {"n": 2},  # sem a nota
    ],
)
def test_invalid_requests_are_422(client: TestClient, request_body: dict[str, object]) -> None:
    assert client.post("/generate", json=request_body).status_code == 422


@pytest.mark.parametrize(
    "raw_body",
    [
        '{"rating": 3, "temperature": NaN}',
        '{"rating": NaN}',
        '{"rating": 3, "seed": Infinity}',
        '{"rating": 3, "extra": -Infinity}',
    ],
)
def test_non_finite_json_numbers_are_422_and_not_500(client: TestClient, raw_body: str) -> None:
    # O JSON do Python aceita NaN e Infinity; o 422 não pode tentar ecoá-los de volta.
    response = client.post(
        "/generate", content=raw_body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert all(set(item) == {"type", "loc", "msg"} for item in response.json()["detail"])


def test_the_generator_second_check_also_becomes_422(
    client: TestClient, tokenizer: Tokenizer
) -> None:
    # O pydantic não conhece o vocabulário; quem recusa um top-k grande demais é o gerador.
    response = client.post("/generate", json={"rating": 3, "top_k": tokenizer.get_vocab_size() + 1})
    assert response.status_code == 422
    assert "top-k" in response.json()["detail"]


def test_exhausted_attempts_are_503_with_the_reasons(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fails(self: ReviewGenerator, *args: object, **kwargs: object) -> list[object]:
        raise GenerationError(3, ["repetition"] * 5)

    monkeypatch.setattr(ReviewGenerator, "generate", always_fails)
    response = client.post("/generate", json={"rating": 3})
    assert response.status_code == 503
    assert response.json()["detail"]["reasons"] == ["repetition"] * 5


def test_the_demo_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "nano-review" in response.text and NOTICE in response.text
    # O estilo e o comportamento ficam em arquivos próprios, e o HTML aponta para eles.
    assert "/static/style.css" in response.text and "/static/app.js" in response.text
    assert "<style>" not in response.text and "style=" not in response.text
    assert "<script>" not in response.text  # só o <script src=...>, sem código no HTML


def test_the_page_css_and_js_are_served(client: TestClient) -> None:
    css = client.get("/static/style.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    # O texto nunca vira HTML: nada de ".innerHTML" (a palavra pode aparecer num comentário).
    assert "textContent" in js.text and ".innerHTML" not in js.text
    assert client.get("/static/nao-existe.js").status_code == 404


def test_resolved_decoding_follows_the_generator_rule() -> None:
    assert resolved_decoding(Settings(), None, None) == (1.0, None)
    assert resolved_decoding(Settings(), 0.7, 0) == (0.7, None)
    assert resolved_decoding(Settings(), None, 50) == (1.0, 50)
    # Omitido vira o padrão do gerador carregado, seja qual for.
    assert resolved_decoding(Settings(temperature=0.8, top_k=40), None, None) == (0.8, 40)


def test_loading_from_the_environment(
    tokenizer: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tokenizer)
    torch.manual_seed(0)
    model = TransformerLanguageModel(config)
    checkpoint, vocab = tmp_path / "model.pt", tmp_path / "tokenizer.json"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, checkpoint)
    tokenizer.save(str(vocab))
    monkeypatch.setenv("NANO_REVIEW_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NANO_REVIEW_TOKENIZER", str(vocab))
    loaded = load_from_env()
    assert loaded.checkpoint_md5 == hashlib.md5(checkpoint.read_bytes()).hexdigest()
    assert loaded.generator.tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
