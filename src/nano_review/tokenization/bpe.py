"""Tokenizador BPE do projeto, treinado do zero e só com o conjunto de treino.

Usamos a biblioteca ``tokenizers`` para treinar (ela é rápida e bem testada) e ``bpe_demo.py`` para
entender o que ela faz. A biblioteca não traz vocabulário nem pesos prontos: tudo é aprendido aqui.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.tokenization.bpe --train data/gold/train.parquet \\
        --validation data/gold/validation.parquet --output models/tokenizer.json \\
        --metrics reports/metrics/tokenizer.json --vocab-size 4000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from nano_review.baseline.ngram import BOS, EOS, UNK, rating_token

PAD = "<pad>"
# A ordem fixa os IDs: <pad> = 0, <bos> = 1, <eos> = 2, <unk> = 3, <nota_1> = 4 ... <nota_5> = 8.
# Os mesmos nomes do baseline, para as duas abordagens falarem a mesma língua.
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, *(rating_token(r) for r in range(1, 6))]
# Comprimentos (em tokens, com <bos>, <nota_k> e <eos>) que interessam para escolher o block_size.
BLOCK_SIZE_CANDIDATES = (64, 128, 256)


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Treina um BPE de caracteres sobre palavras e pontuação já separadas.

    Pré-tokenização, nesta ordem:

    1. ``Metaspace`` troca o espaço por "▁" e o gruda no começo da palavra seguinte, então os tokens
       ficam legíveis ("▁ótimo"). O ``ByteLevel``, comum em modelos em inglês, quebra letras
       acentuadas em bytes e mostraria "Ã©" no lugar de "é".
    2. ``Punctuation`` separa a pontuação da palavra. Sem isso, "▁recomendo", "▁recomendo," e
       "▁recomendo." viravam três tokens diferentes, gastando vocabulário com a mesma palavra.

    A ordem importa: com ``Punctuation`` antes, a vírgula ganharia um "▁" e o texto voltaria como
    "bom , chegou".
    """
    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()]
    )
    tokenizer.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # Um par visto uma vez só não é padrão, é acaso: não vale um token do vocabulário.
        min_frequency=2,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def special_id(tokenizer: Tokenizer, token: str) -> int:
    """ID de um token especial; falha alto se ele não existir, em vez de devolver None."""
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"token especial ausente do vocabulário: {token}")
    return int(token_id)


def encode_reviews(
    tokenizer: Tokenizer, texts: Sequence[str], ratings: Sequence[int]
) -> list[list[int]]:
    """Textos e notas viram sequências de IDs ``<bos> <nota_k> ... <eos>``, as mesmas do baseline.

    O texto precisa ter passado pela limpeza (``nano_review.data.clean.normalize_text``), como os
    textos de treino. O tokenizador não normaliza nada sozinho: uma quebra de linha crua, por
    exemplo, viraria ``<unk>``. Na Etapa 4, a entrada da API passa pela mesma função.
    """
    bos, eos = special_id(tokenizer, BOS), special_id(tokenizer, EOS)
    bodies = tokenizer.encode_batch(list(texts))
    return [
        [bos, special_id(tokenizer, rating_token(int(rating))), *body.ids, eos]
        for rating, body in zip(ratings, bodies, strict=True)
    ]


def encode_review(tokenizer: Tokenizer, text: str, rating: int) -> list[int]:
    """Uma resenha só; ver ``encode_reviews``."""
    return encode_reviews(tokenizer, [text], [rating])[0]


def decode(tokenizer: Tokenizer, ids: Sequence[int]) -> str:
    """IDs de volta para texto, sem os tokens especiais."""
    return str(tokenizer.decode(list(ids), skip_special_tokens=True))


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    """Percentis do comprimento e fração acima de cada candidato a block_size."""
    values = np.asarray(lengths)
    stats: dict[str, float] = {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)}
    stats["max"] = float(values.max())
    for block in BLOCK_SIZE_CANDIDATES:
        stats[f"fraction_over_{block}"] = round(float((values > block).mean()), 4)
    return stats


def describe(tokenizer: Tokenizer, df: pd.DataFrame) -> dict[str, object]:
    """Como o tokenizador se comporta num conjunto: tamanho das sequências e taxa de <unk>."""
    encoded = [tokenizer.encode(text) for text in df["review_text"]]
    n_tokens = sum(len(e.ids) for e in encoded)
    n_words = sum(len(text.split()) for text in df["review_text"])
    unk_id = special_id(tokenizer, UNK)
    n_unk = sum(e.ids.count(unk_id) for e in encoded)
    return {
        # Fertilidade: quantos tokens cada palavra vira, em média. Perto de 1, o vocabulário cobre
        # palavras inteiras; bem acima de 1, as palavras são picotadas e as sequências crescem.
        "tokens_per_word": round(n_tokens / n_words, 3),
        "unk_rate": round(n_unk / n_tokens, 6),
        # +3: <bos>, <nota_k> e <eos>, que também ocupam posições no contexto do modelo.
        "sequence_length": length_stats([len(e.ids) + 3 for e in encoded]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    # Só o treino: se a validação ajudasse a escolher as fusões, o vocabulário já "conheceria"
    # palavras dela, e a perplexity medida ali ficaria otimista.
    tokenizer = train_tokenizer(train["review_text"], args.vocab_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    metrics = {
        "vocab_size": tokenizer.get_vocab_size(),
        "train": describe(tokenizer, train),
        "validation": describe(tokenizer, validation),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
