"""O gerador do produto: validação do pedido, filtros, novas tentativas, semente e checkpoint."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from nano_review.inference import generator as generator_module
from nano_review.inference.generator import (
    MAX_SEED,
    GenerationError,
    ReviewGenerator,
    Settings,
    has_loop,
    load_checkpoint,
    rejection,
)
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.tokenization.bpe import train_tokenizer

TEXTS = ["Produto bom, chegou rápido.", "Não gostei, bla bla bla.", "Meu email é ana@teste.com"]


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return train_tokenizer(TEXTS * 5, vocab_size=80)


def _config(tokenizer: Tokenizer) -> TransformerConfig:
    return TransformerConfig(
        tokenizer.get_vocab_size(), block_size=24, d_model=16, n_layers=1, n_heads=2, dropout=0.0
    )


def _generator(tokenizer: Tokenizer, settings: Settings | None = None) -> ReviewGenerator:
    torch.manual_seed(0)
    return ReviewGenerator(TransformerLanguageModel(_config(tokenizer)), tokenizer, settings)


def test_loop_is_three_equal_words_in_a_row() -> None:
    assert has_loop("Não gostei, bla bla bla.")
    assert has_loop("bom, bom, bom!")  # a pontuação não esconde o loop
    assert not has_loop("muito muito bom")  # ênfase, como em resenha real
    assert not has_loop("bom produto, bom preço, bom vendedor")  # repete, mas não em sequência


@pytest.mark.parametrize(
    ("text", "finished", "reason"),
    [
        ("Produto bom", False, "too_long"),
        ("", True, "empty"),
        ("!!! ...", True, "empty"),
        ("bla bla bla", True, "repetition"),
        ("Meu email é ana@teste.com", True, "personal_data"),
        ("Veja em www.loja.com.br", True, "personal_data"),
        ("VEJA EM WWW.LOJA.COM.BR", True, "personal_data"),  # a URL em maiúsculas também cai
        ("Muito bom muito bom muito bom", True, None),  # loop de frase: fora deste filtro
        ("Produto bom, chegou rápido.", True, None),
    ],
)
def test_rejection_reasons(text: str, finished: bool, reason: str | None) -> None:
    assert rejection(text, finished) == reason


def test_requests_out_of_range_are_refused(tokenizer: Tokenizer) -> None:
    gen = _generator(tokenizer)
    for bad in (0, 6, True):
        with pytest.raises(ValueError, match="nota"):
            gen.generate(bad)
    with pytest.raises(ValueError, match="número de resenhas"):
        gen.generate(5, n=21)
    with pytest.raises(ValueError, match="temperatura"):
        gen.generate(5, temperature=0.0)
    with pytest.raises(ValueError, match="temperatura"):
        gen.generate(5, temperature=2.5)
    with pytest.raises(ValueError, match="top-k"):
        gen.generate(5, top_k=tokenizer.get_vocab_size() + 1)
    with pytest.raises(ValueError, match="semente"):
        gen.generate(5, seed=-1)


def test_same_seed_same_reviews(tokenizer: Tokenizer) -> None:
    # Filtros frouxos, para o modelo minúsculo (que escreve qualquer coisa) não esgotar tentativas.
    gen = _generator(tokenizer, Settings(max_text_tokens=20, max_attempts=50))
    first = gen.generate(4, n=3, seed=11)
    assert first == gen.generate(4, n=3, seed=11)
    assert [review.rating for review in first] == [4, 4, 4]
    assert all(rejection(review.text, finished=True) is None for review in first)


def _fake_sample(
    tokenizer: Tokenizer, outputs: list[tuple[str, bool]]
) -> tuple[list[tuple[str, bool]], object]:
    """Uma amostragem falsa que devolve os textos dados, em ordem, com ou sem ``<eos>``."""
    queue = list(outputs)

    def fake(
        model: object, prefix: list[int], eos: int, *args: object, **kwargs: object
    ) -> list[int]:
        text, finished = queue.pop(0)
        return [*prefix, *tokenizer.encode(text).ids, *([eos] if finished else [])]

    return queue, fake


def test_rejected_texts_are_sampled_again(
    tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, fake = _fake_sample(
        tokenizer,
        [
            ("Produto bom, chegou", False),
            ("bla bla bla", True),
            ("Produto bom, chegou rápido.", True),
        ],
    )
    monkeypatch.setattr(generator_module, "sample", fake)
    (review,) = _generator(tokenizer).generate(5, seed=0)
    assert review.text == "Produto bom, chegou rápido."
    assert review.attempts == 3
    assert review.rejected == ("too_long", "repetition")
    assert queue == []


def test_exhausted_attempts_raise_with_the_reasons(
    tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, fake = _fake_sample(tokenizer, [("Meu email é ana@teste.com", True)] * 3)
    monkeypatch.setattr(generator_module, "sample", fake)
    gen = _generator(tokenizer, Settings(max_attempts=3))
    with pytest.raises(GenerationError) as error:
        gen.generate(1, seed=0)
    assert error.value.reasons == ["personal_data"] * 3
    assert error.value.rating == 1


def test_top_k_zero_means_no_top_k(tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def spy(
        model: object,
        prefix: list[int],
        eos: int,
        temperature: float,
        top_k: object,
        *a: object,
        **k: object,
    ) -> list[int]:
        seen.append(top_k)
        return [*prefix, *tokenizer.encode("Produto bom").ids, eos]

    monkeypatch.setattr(generator_module, "sample", spy)
    gen = _generator(tokenizer, Settings(top_k=50))
    gen.generate(3)  # omitido: o padrão (50)
    gen.generate(3, top_k=0)  # 0: sem top-k
    gen.generate(3, top_k=7)
    assert seen == [50, None, 7]


def test_checkpoint_round_trip_and_the_ablation_is_refused(
    tokenizer: Tokenizer, tmp_path: Path
) -> None:
    config = _config(tokenizer)
    torch.manual_seed(0)
    model = TransformerLanguageModel(config)
    path = tmp_path / "model.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)
    loaded = load_checkpoint(path)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(loaded(ids), model.eval()(ids))

    torch.save(
        {"config": asdict(config), "state_dict": model.state_dict(), "conditioning": False}, path
    )
    with pytest.raises(ValueError, match="sem a nota"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rating", 1.0),
        ("rating", "5"),
        ("n", 0),
        ("n", True),
        ("n", 2.5),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_k", False),  # não pode virar "sem top-k" sem querer
        ("top_k", -1),
        ("top_k", 5.5),
        ("seed", True),
        ("seed", 1.5),
        ("seed", "7"),
        ("seed", MAX_SEED + 1),
    ],
)
def test_wrong_types_are_refused_with_a_clear_error(
    tokenizer: Tokenizer, field: str, value: object
) -> None:
    request: dict[str, object] = {"rating": 3, "n": 1, "temperature": 1.0, "top_k": None, "seed": 0}
    request[field] = value
    with pytest.raises(ValueError):
        _generator(tokenizer).generate(**request)  # type: ignore[arg-type]


def test_the_limits_themselves_are_accepted(tokenizer: Tokenizer) -> None:
    gen = _generator(tokenizer)
    vocab = tokenizer.get_vocab_size()
    gen.check_request(rating=5, n=20, temperature=2.0, top_k=vocab, seed=MAX_SEED)
    gen.check_request(rating=1, n=1, temperature=0.01, top_k=1, seed=0)


def test_the_text_limit_leaves_room_for_the_eos(
    tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits: list[object] = []

    def spy(
        model: object, prefix: list[int], eos: int, *args: object, **kwargs: object
    ) -> list[int]:
        limits.append(kwargs["max_new_tokens"])
        return [*prefix, *tokenizer.encode("Produto bom").ids, eos]

    monkeypatch.setattr(generator_module, "sample", spy)
    _generator(tokenizer, Settings(max_text_tokens=83)).generate(2)
    assert limits == [84]  # até 83 tokens de texto e, depois deles, o <eos>
