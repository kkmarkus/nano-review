"""Limpeza do B2W-Reviews01: de ``data/raw`` para ``data/silver``.

Cada regra é uma função pequena e pura, para poder ser testada sozinha. As regras vêm da tabela de
decisões no fim de ``notebooks/01_eda.ipynb``; o número entre parênteses nos comentários é o da
decisão lá.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.data.clean --input data/raw/B2W-Reviews01.csv \\
        --output data/silver/reviews.parquet --metrics reports/metrics/cleaning.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# (1, 11) Só o que o modelo e o split usam. A categoria fica para a extensão prevista no charter.
# As colunas do avaliador e o título nunca saem de data/raw.
KEPT_COLUMNS = ["product_id", "site_category_lv1", "overall_rating", "review_text"]

# (3) Número solto não basta para dizer que é dado pessoal: ano, preço, pedido, tamanho e código de
# barras também são sequências de dígitos. Cada padrão exige uma "assinatura" do dado: arroba com
# domínio, parênteses no DDD, um celular de 9 dígitos começando com 9, hífen, ou a palavra "cpf",
# "tel", "whats"... Na revisão do fim da Etapa 2, o padrão anterior deixava passar "11987654321" e
# "123456789-09" e tomava "paguei 10 1500 2000" por telefone.
#
# O que fica de fora de propósito, porque é idêntico a texto comum de resenha:
# - fixo sem parênteses nem "tel" ("11 3456-7890" tem a forma de "comprei 21 1500-2000");
# - número só com espaços e sem "tel" ("11 9 8765 4321" tem a forma de uma lista de números);
# - 9 dígitos soltos começando com 9 (podem ser número de pedido);
# - "9 8765-4321" sem DDD (tem a forma de "nota 9 2017-2018").
EMAIL = re.compile(
    # Domínio com letras ("gmail.com"), para "2 @ 5.99" não contar. Espaço em volta do @ só dos
    # dois lados ("joao @ gmail.com"): "sigam @americanas.com.br" é perfil de empresa.
    r"[\w.+-]+(?:@|\s@\s)[a-z][\w-]*(?:\s?\.\s?[a-z]{2,})+"
    r"|\b[\w.+-]+\s+arroba\s+[\w-]+(?:\s+ponto\s+|\.)com\b",  # "fulano arroba gmail ponto com"
    re.IGNORECASE,
)
CPF = re.compile(
    # Com pontos: "123.456.789-09", "123.456.789.09", "123.456.789 09".
    r"(?<!\d)\d{3}\.\d{3}\.\d{3}[-.\s]\d{2}(?!\d)"
    # Sem pontos, com hífen antes do dígito verificador: "123456789-09", "123 456 789-09".
    r"|(?<!\d)\d{3}\s?\d{3}\s?\d{3}-\d{2}(?!\d)"
    # Os 11 dígitos do CPF, com ou sem separadores, logo depois da palavra "cpf".
    r"|\bcpf\b\W{0,5}\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2}(?!\d)",
    re.IGNORECASE,
)
# Só os DDDs que existem no Brasil (Anatel), com o zero antigo opcional ("011"). Isso descarta "10"
# e "40" de preços e notas, e o "78" de um código de barras como "78912345678".
_AREA = r"0?(?:1[1-9]|2[12478]|3[1-578]|4[1-9]|5[1345]|6[1-9]|7[134579]|8[1-9]|9[1-9])"
_DDD = rf"(?:\+?55[\s.-]?)?(?:\({_AREA}\)|{_AREA})"
# Hífen comum e os tipográficos (‐ ‑ ‒ – —), que a normalização mantém no texto.
_DASH = r"[-\u2010-\u2014]"
PHONE = re.compile(
    # Celular com DDD: "11 98765-4321", "(11) 9 8765-4321", "11.98765.4321", "1198765-4321".
    rf"(?<!\d){_DDD}[\s.-]?9[\s.]?\d{{4}}(?:{_DASH}|\.)\d{{4}}(?!\d)"
    # DDD entre parênteses, fixo ou celular, com qualquer separador: "(21) 3456 7890".
    rf"|(?<!\d)(?:\+?55[\s.-]?)?\({_AREA}\)\s?(?:9[\s.]?)?\d{{4}}[\s.]?{_DASH}?\d{{4}}(?!\d)"
    # Celular sem DDD, com hífen: "98765-4321".
    rf"|(?<!\d)9\d{{4}}{_DASH}\d{{4}}(?!\d)"
    # Celular com DDD, tudo junto: "11987654321", "+5511987654321".
    rf"|(?<!\d)(?:\+?55)?{_AREA}9\d{{8}}(?!\d)"
    # Depois de "tel", "cel", "whats"..., aceita qualquer forma, até sem DDD: "tel 3456-7890",
    # "zap 987654321", "tel 11 9 8765 4321".
    rf"|\b(?:tel|telefone|cel|celular|fone|whats|whatsapp|zap|wpp)\b\W{{0,3}}"
    rf"(?:{_DDD}\s?)?(?:9\s?)?\d{{4}}[\s.]?{_DASH}?\d{{4}}(?!\d)",
    re.IGNORECASE,
)
URL = re.compile(r"https?://\S+|www\.\S+")

# (4) Só tags HTML conhecidas: um "<" solto seguido de ">" bem depois ("P<M mas o G>P") apagaria
# o texto do meio.
HTML_TAG = re.compile(r"</?(?:a|b|br|div|em|i|p|span|strong|u)\b[^<>]*>", re.IGNORECASE)
# Caracteres de largura zero somem sem virar espaço, porque ficam no meio da palavra.
ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
# Tudo o que não for texto latino: ASCII imprimível, Latin-1 e Latin Extended (acentos), espaços
# e a pontuação tipográfica comum. Uma lista do que *fica* é mais segura que uma do que sai:
# emojis novos surgem todo ano.
NOT_LATIN_TEXT = re.compile(
    r"[^\s\x20-\x7E\u00A0-\u024F"  # espaços, ASCII imprimível, letras latinas com acento
    r"\u2013\u2014\u2018\u2019\u201C\u201D\u2026\u20AC]"  # – — ‘ ’ “ ” … €
)

# (5) Só letras: [^\W\d_] é "caractere de palavra que não é dígito nem _". Com \w, um preço como
# "10000" viraria "1000". IGNORECASE para "MUUUuuuuito" contar como uma repetição só.
REPEATED_LETTER = re.compile(r"([^\W\d_])\1+", re.IGNORECASE)


def contains_personal_data(text: str) -> bool:
    """Diz se o texto tem e-mail, CPF ou telefone.

    Quem escreve o próprio telefone costuma escrever o nome junto, e regex não detecta nome. Por
    isso a resenha inteira sai, em vez de só mascarar o trecho.

    O filtro roda antes da normalização, então a forma NFC e a remoção dos caracteres de largura
    zero são feitas aqui também: um telefone com um caractere invisível no meio não pode escapar.
    """
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFC", text))
    # A URL sai antes, como na normalização: o número de uma página de produto não é telefone.
    text = URL.sub(" ", text)
    return any(pattern.search(text) for pattern in (EMAIL, CPF, PHONE))


def cap_repeated_letters(text: str, max_repeat: int) -> str:
    """Limita uma letra repetida a ``max_repeat`` vezes ("muuuuuito" -> "muuuito" com 3).

    Mantém a ênfase, que é estilo real de resenha, mas corta as variações de comprimento que o
    tokenizador teria de aprender uma a uma.
    """

    def _cap(match: re.Match[str]) -> str:
        return match.group(1) * min(len(match.group(0)), max_repeat)

    return REPEATED_LETTER.sub(_cap, text)


def normalize_text(text: str, max_repeat: int) -> str:
    """Aplica as regras de normalização (4, 5, 6) e devolve o texto pronto para o modelo.

    Caixa e acentos ficam: a saída do gerador precisa parecer resenha real.
    """
    # NFC primeiro: "é" pode vir como um caractere ou como "e" + acento combinante, e as regras
    # abaixo (e o tokenizador) precisam ver uma forma só.
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH.sub("", text)
    # Daqui em diante troca por espaço, e não por nada, para não colar as palavras vizinhas.
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = NOT_LATIN_TEXT.sub(" ", text)
    text = cap_repeated_letters(text, max_repeat)
    # split() sem argumento quebra em qualquer espaço Unicode (tab, \r, espaço não separável).
    return " ".join(text.split())


def dedupe_key(text: str) -> str:
    """Chave para achar o mesmo texto com caixa, pontuação ou espaços diferentes.

    Serve só para comparar; o texto que vai para o modelo não passa por aqui.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def clean_reviews(raw: pd.DataFrame, max_repeat: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica as regras na ordem da EDA e conta quantas linhas cada uma removeu.

    A ordem importa: o dedupe (9) precisa ver o texto já normalizado, senão "Ótimo!!" e
    "ótimo" contariam como resenhas diferentes.
    """
    removed: dict[str, int] = {}
    df = raw[KEPT_COLUMNS].copy()

    def _keep(mask: pd.Series[bool], reason: str) -> None:
        nonlocal df
        removed[reason] = int((~mask).sum())
        df = df[mask]

    _keep(df["review_text"].notna(), "missing_text")  # (2)
    _keep(~df["review_text"].map(contains_personal_data), "personal_data")  # (3)

    df["review_text"] = df["review_text"].map(lambda t: normalize_text(t, max_repeat))
    df["text_key"] = df["review_text"].map(dedupe_key)
    _keep(df["text_key"] != "", "empty_after_normalization")  # (7)

    # (8) Mesmo texto com duas notas: manter uma seria escolher o rótulo ao acaso.
    ratings_per_text = df.groupby("text_key")["overall_rating"].transform("nunique")
    _keep(ratings_per_text == 1, "conflicting_rating")

    # (9) No corpus inteiro, não por produto: o texto-modelo repetido em produtos diferentes
    # vazaria entre treino e teste, já que o split é por produto.
    _keep(~df.duplicated("text_key", keep="first"), "duplicate_text")

    return df.drop(columns="text_key").reset_index(drop=True), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    # Sem default: o valor vem só do params.yaml, para não existirem dois lugares para mudar.
    parser.add_argument("--max-repeat", type=int, required=True)
    args = parser.parse_args()

    # product_id como texto: é identificador, e lido como número perderia zeros à esquerda.
    raw = pd.read_csv(args.input, sep=",", dtype={"product_id": str}, low_memory=False)
    cleaned, removed = clean_reviews(raw, args.max_repeat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False)

    # O funil linha a linha é a evidência de que nada sumiu sem motivo.
    metrics = {"rows_raw": len(raw), "removed": removed, "rows_processed": len(cleaned)}
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
