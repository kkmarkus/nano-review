"""O juiz agrupa as notas certo, aprende o sentimento e calcula as métricas como esperado."""

from __future__ import annotations

import pytest

from nano_review.baseline.judge import (
    EMPTY,
    build_judge,
    classification_metrics,
    majority_baseline,
    predict_sentiment,
    rating_to_sentiment,
)


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_rating_to_sentiment(rating: int, expected: str) -> None:
    assert rating_to_sentiment(rating) == expected


@pytest.mark.parametrize("rating", [0, 6])
def test_rating_to_sentiment_rejects_invalid(rating: int) -> None:
    with pytest.raises(ValueError):
        rating_to_sentiment(rating)


def test_judge_learns_sentiment_from_text() -> None:
    texts = (
        ["péssimo produto, não recomendo"] * 5
        + ["produto razoável, esperava mais"] * 5
        + ["ótimo produto, recomendo muito"] * 5
    )
    labels = ["negative"] * 5 + ["neutral"] * 5 + ["positive"] * 5
    judge = build_judge(ngram_max=2, min_df=1, c=10.0)
    judge.fit(texts, labels)

    # Textos novos, que só compartilham palavras com os de treino.
    predictions = predict_sentiment(judge, ["péssimo, não recomendo", "ótimo, recomendo"])
    assert predictions == ["negative", "positive"]


def test_empty_texts_are_never_coherent() -> None:
    # O juiz sozinho chamaria "" e "!!!" de "positive" (achado da revisão de código).
    judge = build_judge(ngram_max=1, min_df=1, c=1.0)
    judge.fit(["ruim", "ok", "bom", "bom"], ["negative", "neutral", "positive", "positive"])
    assert predict_sentiment(judge, ["", "!!!", "....", "bom"]) == [EMPTY, EMPTY, EMPTY, "positive"]


def test_classification_metrics_on_hand_computed_example() -> None:
    y_true = ["negative", "negative", "neutral", "positive"]
    y_pred = ["negative", "positive", "neutral", "positive"]
    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    # Recall: negativo acertou 1 de 2; neutro 1 de 1; positivo 1 de 1.
    assert metrics["recall"] == {"negative": 0.5, "neutral": 1.0, "positive": 1.0}
    # Linha = verdadeira, coluna = prevista (negative, neutral, positive).
    assert metrics["confusion_matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    # F1 por classe: negativo 2/3, neutro 1, positivo 2/3 -> média 0,7778.
    assert metrics["macro_f1"] == pytest.approx(0.7778, abs=1e-4)


def test_majority_baseline_repeats_most_common_class() -> None:
    assert majority_baseline(["positive", "positive", "negative"], 3) == ["positive"] * 3
