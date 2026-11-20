"""Medidas da análise de erros: palavras inventadas, onde o texto quebra e nota rara."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from nano_review.evaluation.error_analysis import (
    alpha_words,
    analyze,
    check_types,
    read_types,
    text_profile,
    train_vocabulary,
)

TRAIN = pd.DataFrame(
    {
        "review_text": ["Produto bom.", "Produto ruim!", "Chegou rápido", "bom bom"],
        "overall_rating": [5, 1, 5, 4],
    }
)


def _scored(rows: list[tuple[str, int, str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sample_id", "requested_rating", "text", "model", "bad"])


def test_numbers_and_dates_are_not_invented_words() -> None:
    assert alpha_words("Chegou dia 29/12, 2x4GB. Ótimo!") == ["chegou", "dia", "ótimo"]


def test_profile_counts_invented_words_and_where_the_first_one_is() -> None:
    vocabulary = train_vocabulary(TRAIN["review_text"])
    assert vocabulary == {"produto", "bom", "ruim", "chegou", "rápido"}
    # 4 palavras ("29" não conta); a inventada é a 3ª palavra (índice 2): relativa 2/4, absoluta 3.
    assert text_profile("Produto bom xyzq 29 bom", vocabulary) == (4, 1, 0.5, 3.0)
    words, invented, first, first_word = text_profile("Chegou rápido, bom", vocabulary)
    assert (words, invented) == (3, 0) and math.isnan(first) and math.isnan(first_word)


def test_types_must_cover_exactly_the_bad_transformer_samples() -> None:
    scored = _scored(
        [
            ("s1", 1, "x", "transformer", True),
            ("s2", 2, "x", "transformer", False),
            ("s3", 1, "x", "trigram", True),  # ruim, mas do trigrama: fica fora
        ]
    )
    check_types(pd.DataFrame({"sample_id": ["s1"], "error_type": ["drift"]}), scored)
    with pytest.raises(ValueError, match="desconhecidos"):
        check_types(pd.DataFrame({"sample_id": ["s1"], "error_type": ["typo"]}), scored)
    with pytest.raises(ValueError, match="exatamente"):
        check_types(pd.DataFrame({"sample_id": ["s1", "s3"], "error_type": ["drift"] * 2}), scored)
    with pytest.raises(ValueError, match="exatamente"):  # falta a s1
        check_types(pd.DataFrame({"sample_id": [], "error_type": []}), scored)
    with pytest.raises(ValueError, match="repetido"):
        check_types(pd.DataFrame({"sample_id": ["s1", "s1"], "error_type": ["drift"] * 2}), scored)
    with pytest.raises(ValueError, match="desconhecidos"):  # célula vazia no CSV
        check_types(pd.DataFrame({"sample_id": ["s1"], "error_type": [None]}), scored)


def test_reads_types_saved_by_excel(tmp_path: Path) -> None:
    path = tmp_path / "types.csv"
    path.write_text("sample_id;error_type\ns1;drift\n", encoding="utf-8-sig")
    assert read_types(path).to_dict("records") == [{"sample_id": "s1", "error_type": "drift"}]


def test_analysis_by_hand() -> None:
    scored = _scored(
        [
            ("s1", 1, "Produto ruim xyzq", "transformer", True),
            ("s2", 1, "Produto ruim", "transformer", False),
            ("s3", 5, "abcd produto bom bom", "transformer", True),
            ("s4", 5, "Produto bom", "real", False),
        ]
    )
    types = pd.DataFrame({"sample_id": ["s1", "s3"], "error_type": ["drift", "garbled"]})
    result = analyze(scored, types, TRAIN)

    bad = result["by_source"]["transformer"]["bad"]  # type: ignore[index]
    assert bad["samples"] == 2
    assert bad["with_invented_word"] == 1.0
    assert bad["invented_per_100_words"] == round(100 * 2 / 7, 2)  # 2 inventadas em 3 + 4 palavras
    assert bad["median_first_invented"] == round((2 / 3 + 0 / 4) / 2, 3)
    assert bad["median_first_invented_word"] == (3 + 1) / 2  # 3ª palavra em s1, 1ª em s3

    ok = result["by_source"]["transformer"]["ok"]  # type: ignore[index]
    assert ok["samples"] == 1 and ok["median_words"] == 2.0
    # Sem palavra inventada, não há posição para resumir.
    assert ok["median_first_invented"] is None and ok["median_first_invented_word"] is None

    every = result["by_source"]["transformer"]["all"]  # type: ignore[index]
    assert every["samples"] == 3 and every["median_words"] == 3.0  # 3, 2 e 4 palavras
    assert every["with_invented_word"] == round(2 / 3, 3)
    assert result["by_source"]["real"]["all"]["with_invented_word"] == 0.0  # type: ignore[index]

    assert result["transformer_by_rating"] == {
        "1": {"train_share": 0.25, "bad_fraction": 0.5},
        "5": {"train_share": 0.5, "bad_fraction": 1.0},
    }
    assert result["error_types"]["drift"] == {"samples": 1, "by_rating": {"1": 1}}  # type: ignore[index]
    assert result["error_types"]["loop"] == {"samples": 0, "by_rating": {}}  # type: ignore[index]
