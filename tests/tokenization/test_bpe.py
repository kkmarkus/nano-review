"""O tokenizador real: IDs fixos dos especiais, formato da sequência e ida e volta do texto."""

from __future__ import annotations

import string

import pytest
from tokenizers import Tokenizer

from nano_review.tokenization.bpe import (
    SPECIAL_TOKENS,
    decode,
    encode_review,
    length_stats,
    special_id,
    train_tokenizer,
)

ROUND_TRIP_TEXTS = [
    "Produto muito bom, chegou rápido.",
    "Gostei!! Muito bom...",
    "custo-benefício (nota 10).",
]
TEXTS = [
    "Não recomendo, veio quebrado.",
    "Ótimo custo benefício, recomendo!",
    *ROUND_TRIP_TEXTS,
] * 20


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return train_tokenizer(TEXTS, vocab_size=120)


def test_special_tokens_have_fixed_ids(tokenizer: Tokenizer) -> None:
    # O modelo e o DataLoader dependem destes IDs; eles não podem mudar entre treinos.
    assert [special_id(tokenizer, token) for token in SPECIAL_TOKENS] == list(range(9))


def test_encode_review_wraps_text_with_bos_rating_and_eos(tokenizer: Tokenizer) -> None:
    ids = encode_review(tokenizer, "Produto muito bom", rating=5)
    assert ids[0] == special_id(tokenizer, "<bos>")
    assert ids[1] == special_id(tokenizer, "<nota_5>")
    assert ids[-1] == special_id(tokenizer, "<eos>")
    assert len(ids) > 3


def test_decode_round_trips_the_text(tokenizer: Tokenizer) -> None:
    text = "Não recomendo, veio quebrado."
    assert decode(tokenizer, encode_review(tokenizer, text, rating=1)) == text


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_punctuation_is_split_and_text_round_trips(tokenizer: Tokenizer, text: str) -> None:
    # Achado ao rodar com os dados reais: sem separar a pontuação, "▁bom," e "▁bom" eram tokens
    # diferentes. Nenhum token pode misturar letra e pontuação, e a volta para texto continua
    # exata, sem espaço antes da vírgula.
    for token in tokenizer.encode(text).tokens:
        has_letter = any(c.isalnum() for c in token)
        has_punctuation = any(c in string.punctuation for c in token)
        assert not (has_letter and has_punctuation), token
    assert decode(tokenizer, encode_review(tokenizer, text, rating=3)) == text


def test_frequent_words_become_single_tokens(tokenizer: Tokenizer) -> None:
    # "recomendo" aparece 40 vezes no corpus: o BPE deve tê-la fundido inteira.
    assert "▁recomendo" in tokenizer.encode("recomendo").tokens


def test_length_stats() -> None:
    stats = length_stats([10, 20, 30, 200, 300])
    assert stats["p50"] == 30
    assert stats["max"] == 300
    assert stats["fraction_over_128"] == pytest.approx(0.4)
    assert stats["fraction_over_256"] == pytest.approx(0.2)
