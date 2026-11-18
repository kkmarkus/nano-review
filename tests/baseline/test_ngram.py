"""Os n-gramas dão probabilidades válidas, preferem o que viram no treino e geram com semente."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from nano_review.baseline.ngram import (
    BOS,
    EOS,
    UNK,
    InterpolatedTrigram,
    RatingUnigram,
    build_vocabulary,
    detokenize,
    rating_token,
    to_sequence,
    word_tokenize,
)

TRAIN = [
    ("produto muito bom , recomendo", 5),
    ("produto muito bom , chegou rápido", 5),
    ("produto muito ruim , não recomendo", 1),
    ("produto ruim , quebrou", 1),
]


def _fit(lambdas: tuple[float, float, float]) -> tuple[InterpolatedTrigram, set[str]]:
    vocabulary = build_vocabulary((t.split() for t, _ in TRAIN), min_count=1)
    model = InterpolatedTrigram(lambdas)
    model.fit(to_sequence(t.split(), r, vocabulary) for t, r in TRAIN)
    return model, vocabulary


def test_tokenize_and_detokenize_round_trip() -> None:
    tokens = word_tokenize("Muito bom!! Chegou (rápido), recomendo.")
    assert tokens == ["Muito", "bom", "!", "!", "Chegou", "(", "rápido", ")", ",", "recomendo", "."]
    assert detokenize(tokens) == "Muito bom!! Chegou (rápido), recomendo."


def test_rare_words_become_unk() -> None:
    vocabulary = build_vocabulary([["bom", "bom", "raro"]], min_count=2)
    assert to_sequence(["bom", "raro"], 4, vocabulary) == [BOS, rating_token(4), "bom", UNK, EOS]


@pytest.mark.parametrize(
    "history",
    [
        [BOS, rating_token(5)],  # contexto visto no treino
        [BOS, rating_token(5), "produto", "muito"],
        ["quebrou", "muito"],  # trigrama nunca visto: o peso dele vai para as ordens menores
        ["xyz", "abc"],  # nem trigrama nem bigrama vistos: sobra só o unigrama
    ],
)
def test_trigram_probabilities_sum_to_one(history: list[str]) -> None:
    # Sem isso a perplexity não significa nada: uma "probabilidade" que soma 0,7 dá números
    # artificialmente bons.
    model, vocabulary = _fit((0.2, 0.3, 0.5))
    targets = vocabulary | {EOS, UNK}
    assert sum(model.prob(word, history) for word in targets) == pytest.approx(1.0)


def test_trigram_uses_the_rating_at_the_start() -> None:
    model, _ = _fit((0.1, 0.1, 0.8))
    after_5 = [BOS, rating_token(5), "produto", "muito"]
    after_1 = [BOS, rating_token(1), "produto", "muito"]
    # "produto muito" vem antes de "bom" na nota 5 e de "ruim" na nota 1. Mas o trigrama só vê
    # "produto muito": a nota já saiu da janela, então as duas notas dão a mesma probabilidade.
    assert model.prob("bom", after_5) == model.prob("bom", after_1)
    # Na segunda palavra a nota ainda está na janela ("<nota_k> produto") e muda a previsão:
    # na nota 1, "produto" é seguido de "ruim" metade das vezes; na nota 5, nunca.
    assert model.prob("ruim", [BOS, rating_token(1), "produto"]) > model.prob(
        "ruim", [BOS, rating_token(5), "produto"]
    )


@pytest.mark.parametrize(
    "lambdas",
    [
        (0.5, 0.5, 0.5),  # não soma 1
        (0.0, 0.5, 0.5),  # sem unigrama, um histórico desconhecido não teria distribuição
    ],
)
def test_rejects_invalid_lambdas(lambdas: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        InterpolatedTrigram(lambdas)


def _fit_unigram() -> RatingUnigram:
    vocabulary = build_vocabulary((t.split() for t, _ in TRAIN), min_count=1)
    model = RatingUnigram()
    model.fit(to_sequence(t.split(), r, vocabulary) for t, r in TRAIN)
    return model


@pytest.mark.parametrize("rating", [1, 5])
def test_rating_unigram_probabilities_sum_to_one(rating: int) -> None:
    model = _fit_unigram()
    history = [BOS, rating_token(rating)]
    assert sum(model.prob(word, history) for word in model.vocabulary) == pytest.approx(1.0)


def test_rating_unigram_rejects_unknown_rating() -> None:
    with pytest.raises(ValueError):
        _fit_unigram().prob("bom", [BOS, rating_token(3)])


@pytest.mark.parametrize("model_name", ["trigram", "unigram"])
def test_sampling_follows_prob(model_name: str) -> None:
    # O sorteio usa um atalho (escolher a ordem, depois a palavra). Se o atalho estivesse errado,
    # as amostras viriam de outra distribuição, e a perplexity mediria um modelo que não é o que
    # gera os textos. 20 mil sorteios deixam o erro de amostragem abaixo de 1 ponto percentual.
    model: InterpolatedTrigram | RatingUnigram = (
        _fit((0.2, 0.3, 0.5))[0] if model_name == "trigram" else _fit_unigram()
    )
    history = [BOS, rating_token(1), "produto"]
    rng = np.random.default_rng(0)
    draws = Counter(model.sample_next(history, rng) for _ in range(20_000))
    targets = model.targets if isinstance(model, InterpolatedTrigram) else model.vocabulary
    for word in targets:
        assert draws[word] / 20_000 == pytest.approx(model.prob(word, history), abs=0.02)


def test_rating_unigram_ignores_context_but_not_rating() -> None:
    model = _fit_unigram()
    assert model.prob("bom", [BOS, rating_token(5)]) == model.prob(
        "bom", [BOS, rating_token(5), "não"]
    )
    assert model.prob("bom", [BOS, rating_token(5)]) > model.prob("bom", [BOS, rating_token(1)])
    # Laplace: palavra do vocabulário que nunca apareceu na nota 1 ainda tem probabilidade > 0.
    assert model.prob("rápido", [BOS, rating_token(1)]) > 0


def test_generation_is_repeatable_and_bounded() -> None:
    model, vocabulary = _fit((0.2, 0.3, 0.5))
    first = model.generate(5, np.random.default_rng(0), max_tokens=20)
    second = model.generate(5, np.random.default_rng(0), max_tokens=20)
    assert first == second
    assert len(first) <= 20
    assert set(first) <= vocabulary  # nunca escreve <bos>, <eos>, <unk> nem <nota_k>
