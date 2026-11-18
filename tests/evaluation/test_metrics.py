"""Cada métrica num exemplo pequeno, com o resultado esperado calculado à mão."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from nano_review.evaluation.metrics import coherence, copy_rates, distinct_n, perplexity


class _UniformModel:
    """Modelo que dá a mesma probabilidade a todas as palavras de um vocabulário de tamanho V."""

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size

    def prob(self, word: str, history: Sequence[str]) -> float:
        return 1 / self.vocabulary_size


def test_perplexity_of_uniform_model_is_vocabulary_size() -> None:
    # A definição intuitiva da perplexity: escolher entre V palavras igualmente prováveis.
    sequences = [["<bos>", "<nota_5>", "a", "b", "<eos>"], ["<bos>", "<nota_1>", "c", "<eos>"]]
    assert perplexity(_UniformModel(50), sequences) == pytest.approx(50)


class _StrictModel:
    """Probabilidade 0 para <bos> e <nota_k>: se a perplexity tentar prevê-los, dá log(0)."""

    def prob(self, word: str, history: Sequence[str]) -> float:
        return {"a": 0.5, "<eos>": 0.25}.get(word, 0.0)


def test_perplexity_counts_words_and_eos_but_not_prefix() -> None:
    # O modelo uniforme acima daria o mesmo valor com as posições erradas; este não. Só "a" e
    # "<eos>" são previstos: exp(-(ln 0,5 + ln 0,25) / 2) = raiz de 8.
    sequences = [["<bos>", "<nota_5>", "a", "<eos>"]]
    assert perplexity(_StrictModel(), sequences) == pytest.approx(8**0.5)


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        perplexity(_UniformModel(5), [])
    with pytest.raises(ValueError):
        copy_rates([], ["texto de treino"])


def test_distinct_n() -> None:
    texts = ["muito bom", "Muito bom!", "bom demais"]
    # Palavras: muito bom muito bom bom demais -> 3 únicas de 6.
    assert distinct_n(texts, 1) == pytest.approx(3 / 6)
    # Bigramas: (muito bom) (muito bom) (bom demais) -> 2 únicos de 3.
    assert distinct_n(texts, 2) == pytest.approx(2 / 3)


def test_copy_rates() -> None:
    train = ["Chegou muito rápido e o produto é de ótima qualidade, recomendo"]
    samples = [
        "chegou muito rápido e o produto é de ótima qualidade recomendo",  # cópia exata
        "nossa, chegou muito rápido e o produto é de ótima marca",  # trecho de 8 palavras
        "produto bom",  # curta demais para ter 8 palavras
        "não gostei da cor nem do tamanho do produto que comprei",  # original
    ]
    rates = copy_rates(samples, train, n=8)
    assert rates["exact"] == pytest.approx(1 / 4)
    assert rates["ngram"] == pytest.approx(2 / 4)
    assert rates["any"] == pytest.approx(2 / 4)  # a cópia exata também tem trecho de 8


def test_short_generic_text_is_not_an_exact_copy() -> None:
    # "boa qualidade" coincide com o treino sem ter sido decorado; 5 palavras idênticas já contam.
    # A de 5 palavras não tem trecho de 8, então só o "any" a soma com a cópia parcial.
    train = ["Boa qualidade.", "chegou rápido e bem embalado", "a b c d e f g h i"]
    samples = ["boa qualidade", "Chegou rápido e bem embalado!", "x a b c d e f g h", "outra"]
    rates = copy_rates(samples, train, n=8, min_exact_words=5)
    assert rates["exact"] == pytest.approx(1 / 4)  # só a de 5 palavras
    assert rates["ngram"] == pytest.approx(1 / 4)  # só "a b c d e f g h"
    assert rates["any"] == pytest.approx(2 / 4)


def test_coherence_macro_averages_classes() -> None:
    requested = [1, 2, 3, 5, 5, 5, 5]
    judged = ["negative", "positive", "neutral", "positive", "positive", "positive", "negative"]
    result = coherence(requested, judged)
    # Negativo: 1 de 2; neutro: 1 de 1; positivo: 3 de 4. Macro = (0,5 + 1 + 0,75) / 3.
    assert result["by_class"] == {"negative": 0.5, "neutral": 1.0, "positive": 0.75}
    assert result["macro"] == pytest.approx(0.75)
    assert result["by_rating"] == {"1": 1.0, "2": 0.0, "3": 1.0, "5": 0.75}


def test_coherence_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        coherence([1, 2], ["negative"])
