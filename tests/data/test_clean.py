"""Cada regra de limpeza com exemplos pequenos, escritos à mão.

Os exemplos vêm dos casos reais que a EDA encontrou (risada esticada, texto-modelo repetido,
mesmo texto com duas notas), para o teste falhar exatamente onde a regra importa.
"""

import pandas as pd
import pytest

from nano_review.data.clean import (
    cap_repeated_letters,
    clean_reviews,
    contains_personal_data,
    dedupe_key,
    normalize_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "me chama em fulano.teste@exemplo.com",
        "meu cpf é 123.456.789-09",
        "cpf 12345678909 sem pontos",
        "liga no (11) 98765-4321",
        "fixo (21) 3456-7890",
        "whats 98765 4321",
        "zap 987654321",
        # Formatos que o padrão anterior deixava passar (revisão do fim da Etapa 2).
        "me liga 11987654321",
        "+5511987654321",
        "(11) 9 8765-4321",
        "tel 11 9 8765 4321",
        "tel 3456-7890",
        "fone: 3456 7890",
        "(011) 3456-7890",
        "11.98765.4321",
        "98765\u20134321",  # hífen tipográfico (–)
        "123456789-09",
        "CPF 123 456 789-09",
        "123.456.789.09",
        "joao @ gmail.com",
        "joao arroba gmail ponto com",
        "joao arroba gmail.com",
        "98765\u200b-4321",  # caractere de largura zero no meio do número
    ],
)
def test_detects_personal_data(text: str) -> None:
    assert contains_personal_data(text)


@pytest.mark.parametrize(
    "text",
    [
        "comprei em 12/03/2018 e chegou em 15/03/2018",  # datas foram o falso alarme mais comum
        "paguei R$ 1.299,90 à vista",
        "nota 10, recomendo!",
        # Sequências de dígitos que não são dados pessoais. Achadas pela revisão de código: o
        # regex antigo tomava as três primeiras por telefone e a quarta por CPF.
        "modelo 2017 2018 top",
        "paguei 1500 2000 no total",
        "pedido 12345678 nao chegou",
        "codigo de barras 78912345678 ok",
        "CEP 01310-100, entregou às 18:08",
        # URL não é dado pessoal: ela é apagada do texto, mas não derruba a resenha.
        "veja em https://loja.com/produto",
        # Falsos alarmes do padrão anterior: "NN NNNN NNNN" sem hífen, parênteses nem "tel" não é
        # telefone, e "40" ou "10" não são DDD.
        "paguei 10 1500 2000",
        "de 15 2000 3000 reais",
        "tamanho 40 9000-1234",
        "modelo 2017-2018 otimo",
        "natal@!!!! muito bom",
        "celular de 32 gb com 4 gb de ram, 128 64 32 16",
        # Achados da segunda revisão: um número de 2 dígitos que é DDD válido, seguido de um
        # intervalo, não é telefone. Calçados 33 a 44 são quase todos DDDs.
        "comprei 21 1500-2000",
        "nota 11 2017-2018",
        "tamanho 42 9000-1234",
        "dou nota 9 2017-2018",
        "pedido 912345678 atrasou",
        "paguei 2 @ 5.99 cada",
        "sigam @americanas.com.br",
        "veja https://loja.com/p/11987654321 antes",
        "2x de 49,90 e voltagem 110 220",
        "chegou 11/12/2018 às 18:30",
        "EAN 7891234567895",
    ],
)
def test_ignores_what_is_not_personal_data(text: str) -> None:
    assert not contains_personal_data(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("muuuuuito bom", "muuuito bom"),
        ("kkkkkkk", "kkk"),
        ("ameiii", "ameiii"),  # exatamente no limite: fica
        ("carro", "carro"),  # letra dupla legítima
        ("custou 10000 reais", "custou 10000 reais"),  # dígitos não são letras
        ("MUUUuuuuito", "MUUUito"),  # caixa misturada conta como uma repetição só
    ],
)
def test_cap_repeated_letters(text: str, expected: str) -> None:
    assert cap_repeated_letters(text, max_repeat=3) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  bom\tproduto\r\n ", "bom produto"),
        ("ótimo😍produto", "ótimo produto"),  # emoji vira espaço para não colar as palavras
        ("veja https://loja.com/x aqui", "veja aqui"),
        ("bom<br/>produto", "bom produto"),
        ('<a href="x">link</a> bom', "link bom"),
        # "<" e ">" soltos não são tag: o texto entre eles fica.
        ("tamanho P<M mas o G>P ficou bom", "tamanho P<M mas o G>P ficou bom"),
        ("pro" + chr(0x200B) + "duto", "produto"),  # largura zero some, sem quebrar a palavra
        ("ÓTIMO, Recomendo!", "ÓTIMO, Recomendo!"),  # caixa e pontuação ficam
        ("chegou rápido… “perfeito” – nota 10", "chegou rápido… “perfeito” – nota 10"),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text, max_repeat=3) == expected


def test_normalize_text_unifies_unicode_forms() -> None:
    # "é" como um caractere só e como "e" + acento combinante: na tela são iguais.
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert normalize_text(composed, 3) == normalize_text(decomposed, 3) == composed


def test_dedupe_key_ignores_case_punctuation_and_spaces() -> None:
    assert dedupe_key("Muito bom!!") == dedupe_key("muito   bom") == "muito bom"
    assert dedupe_key("!!!") == ""


def _reviews(rows: list[tuple[str, int, str | None]]) -> pd.DataFrame:
    """DataFrame no formato do CSV bruto, com as colunas que a limpeza precisa descartar."""
    return pd.DataFrame(
        {
            "product_id": [r[0] for r in rows],
            "site_category_lv1": "Celulares",
            "overall_rating": [r[1] for r in rows],
            "review_text": [r[2] for r in rows],
            "review_title": "título",
            "reviewer_id": "abc",
            "reviewer_gender": "F",
        }
    )


def test_clean_reviews_applies_every_rule_and_counts_them() -> None:
    raw = _reviews(
        [
            ("p1", 5, "Parabéns pela agilidade!"),
            ("p2", 5, "parabéns pela   agilidade"),  # mesmo texto em outro produto: duplicata
            ("p3", 4, None),  # sem texto
            ("p4", 1, "liga no (11) 98765-4321"),  # dado pessoal
            ("p5", 3, "😍😍😍"),  # vira vazio
            ("p6", 5, "bom"),  # "bom" com duas notas: as duas saem
            ("p7", 2, "Bom."),
            ("p8", 1, "Péssimo, não recomendo"),
        ]
    )

    processed, removed = clean_reviews(raw, max_repeat=3)

    assert removed == {
        "missing_text": 1,
        "personal_data": 1,
        "empty_after_normalization": 1,
        "conflicting_rating": 2,
        "duplicate_text": 1,
    }
    assert processed["review_text"].tolist() == [
        "Parabéns pela agilidade!",
        "Péssimo, não recomendo",
    ]
    assert processed["product_id"].tolist() == ["p1", "p8"]
    assert list(processed.columns) == [
        "product_id",
        "site_category_lv1",
        "overall_rating",
        "review_text",
    ]


def test_clean_reviews_never_keeps_personal_columns() -> None:
    processed, _ = clean_reviews(_reviews([("p1", 5, "ótimo")]), max_repeat=3)
    assert not {"reviewer_id", "reviewer_gender", "review_title"} & set(processed.columns)
