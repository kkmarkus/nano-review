"""Teste final: a sanidade que trava a abertura do teste, o controle do lote v2 e o trigrama."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nano_review.baseline.evaluate import make_sequences
from nano_review.evaluation.final_eval import check_sanity, control_sample, fit_trigram
from nano_review.evaluation.metrics import perplexity
from nano_review.tokenization.bpe import train_tokenizer


def test_sanity_passes_within_rounding() -> None:
    report = check_sanity({"a": 69.314, "b": 90.2}, {"a": 69.31, "b": 90.2})
    assert report["a"] == {"measured": 69.314, "expected": 69.31, "ok": True}


def test_sanity_failure_stops_before_the_test_is_read() -> None:
    with pytest.raises(RuntimeError, match="o teste não foi aberto") as error:
        check_sanity({"a": 69.4, "b": 90.2}, {"a": 69.31, "b": 90.2})
    assert "a: medido 69.4" in str(error.value) and "b:" not in str(error.value)


def test_control_keeps_short_reviews_and_n_per_rating() -> None:
    short, long = "Produto bom.", "Muito ruim " * 30
    test = pd.DataFrame(
        {
            "review_text": [short, long] * 15,
            "overall_rating": [r for r in (1, 2, 3, 4, 5) for _ in range(6)],
        }
    )
    tokenizer = train_tokenizer([short, long] * 5, vocab_size=60)
    control = control_sample(test, tokenizer, max_tokens=10, per_rating=2, seed=0)
    assert control["requested_rating"].value_counts().to_dict() == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert set(control["text"]) == {short}  # as longas ficam de fora
    again = control_sample(test, tokenizer, max_tokens=10, per_rating=2, seed=0)
    pd.testing.assert_frame_equal(control, again)


def test_trigram_is_refit_with_the_given_lambdas() -> None:
    train = pd.DataFrame(
        {
            "review_text": ["produto bom", "produto ruim", "bom demais"] * 3,
            "overall_rating": [5, 1, 5] * 3,
        }
    )
    trigram, vocabulary = fit_trigram(train, [0.2, 0.4, 0.4], vocab_min_count=1)
    assert {"produto", "bom", "ruim", "demais"} <= vocabulary
    assert trigram.lambdas == (0.2, 0.4, 0.4)
    # Nas próprias resenhas do treino, a perplexity é finita e maior que 1 (nenhuma palavra tem
    # probabilidade 0, graças ao unigrama, e nenhuma tem probabilidade 1).
    value = perplexity(trigram, make_sequences(train, vocabulary))
    assert math.isfinite(value) and value > 1
