"""Treina e avalia os geradores de n-gramas com as métricas do charter.

Passos:

1. Tokeniza o treino e monta o vocabulário (palavras raras viram ``<unk>``).
2. Escolhe os pesos λ do trigrama numa fatia do **treino** separada por produto. Escolher olhando
   a validação deixaria otimista a perplexity reportada nela.
3. Treina o unigrama por nota (piso) e o trigrama (baseline) no treino inteiro.
4. Mede a perplexity na validação.
5. Gera N resenhas por nota e mede a coerência (pelo juiz), a diversidade e a cópia do treino.
6. Mede o mesmo em N resenhas **reais** da validação por nota. Elas são a referência: mostram o
   máximo de coerência que o juiz consegue enxergar e quanta diversidade e "cópia" o texto
   humano tem naturalmente.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.baseline.evaluate --train data/curated/train.parquet \\
        --validation data/curated/validation.parquet --judge models/judge.joblib \\
        --metrics reports/metrics/baseline.json --samples reports/samples/baseline.jsonl \\
        --vocab-min-count 2 --samples-per-rating 200 --max-tokens 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from nano_review.baseline.judge import EMPTY, predict_sentiment
from nano_review.baseline.ngram import (
    InterpolatedTrigram,
    RatingUnigram,
    build_vocabulary,
    detokenize,
    to_sequence,
    word_tokenize,
)
from nano_review.evaluation.metrics import coherence, copy_rates, distinct_n, perplexity, words

RATINGS = (1, 2, 3, 4, 5)


# Grade de λ (unigrama, bigrama, trigrama) em passos de 0,1, com no mínimo 0,1 para cada ordem:
# um peso zero no unigrama daria probabilidade zero a combinações nunca vistas.
def _lambda_grid() -> list[tuple[float, float, float]]:
    grid = []
    for bigram_tenths in range(1, 9):
        for trigram_tenths in range(1, 9):
            # Em décimos inteiros, para não acumular erro de ponto flutuante (0,1 + 0,2 != 0,3).
            unigram_tenths = 10 - bigram_tenths - trigram_tenths
            if unigram_tenths >= 1:
                grid.append((unigram_tenths / 10, bigram_tenths / 10, trigram_tenths / 10))
    return grid


LAMBDA_GRID = _lambda_grid()


def make_sequences(df: pd.DataFrame, vocabulary: set[str]) -> list[list[str]]:
    return [
        to_sequence(word_tokenize(text), int(rating), vocabulary)
        for text, rating in zip(df["review_text"], df["overall_rating"], strict=True)
    ]


def tune_lambdas(
    train: pd.DataFrame, vocab_min_count: int, tuning_fraction: float, seed: int
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Escolhe os λ com menor perplexity numa fatia do treino separada por produto."""
    rng = np.random.default_rng(seed)
    products = np.sort(train["product_id"].unique())
    held_out = set(rng.choice(products, size=int(len(products) * tuning_fraction), replace=False))
    is_held_out = train["product_id"].isin(held_out)
    if not is_held_out.any():
        raise ValueError("a fatia de ajuste ficou vazia; aumente tuning_fraction")

    # O vocabulário sai só da parte usada para ajustar, exatamente como no modelo final. Com o
    # vocabulário do treino inteiro, uma palavra presente só na fatia separada teria contagem zero
    # aqui, probabilidade zero e log(0) na perplexity (bug achado ao rodar com os dados reais).
    fitting = train[~is_held_out]
    vocabulary = build_vocabulary(map(word_tokenize, fitting["review_text"]), vocab_min_count)
    model = InterpolatedTrigram(LAMBDA_GRID[0])
    model.fit(make_sequences(fitting, vocabulary))
    held_out_sequences = make_sequences(train[is_held_out], vocabulary)

    scores: dict[str, float] = {}
    for lambdas in LAMBDA_GRID:
        # As contagens não dependem dos λ: treinamos uma vez e só trocamos os pesos.
        model.lambdas = lambdas
        scores[str(lambdas)] = round(perplexity(model, held_out_sequences), 2)
    best = min(LAMBDA_GRID, key=lambda lambdas: scores[str(lambdas)])
    return best, scores


def generate_samples(
    model: InterpolatedTrigram | RatingUnigram, per_rating: int, max_tokens: int, seed: int
) -> list[tuple[int, str]]:
    """``per_rating`` resenhas para cada nota, com a mesma semente para todos os modelos."""
    rng = np.random.default_rng(seed)
    return [
        (rating, detokenize(model.generate(rating, rng, max_tokens)))
        for rating in RATINGS
        for _ in range(per_rating)
    ]


def score_texts(
    requested: Sequence[int], texts: Sequence[str], judge: Pipeline, train_texts: Sequence[str]
) -> dict[str, object]:
    """As métricas de amostra do charter para um conjunto de textos."""
    judged = predict_sentiment(judge, texts)
    return {
        "coherence": coherence(requested, judged),
        # Amostras sem nenhuma palavra: contam como incoerentes, e a fração delas é reportada.
        "empty_rate": round(judged.count(EMPTY) / len(judged), 4),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "copy": {k: round(v, 4) for k, v in copy_rates(texts, train_texts).items()},
        "mean_words": round(float(np.mean([len(words(t)) for t in texts])), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--vocab-min-count", type=int, required=True)
    parser.add_argument("--tuning-fraction", type=float, required=True)
    parser.add_argument("--samples-per-rating", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    judge = joblib.load(args.judge)
    train_texts = train["review_text"].tolist()

    vocabulary = build_vocabulary(map(word_tokenize, train_texts), args.vocab_min_count)
    lambdas, tuning_scores = tune_lambdas(
        train, args.vocab_min_count, args.tuning_fraction, args.seed
    )

    train_sequences = make_sequences(train, vocabulary)
    validation_sequences = make_sequences(validation, vocabulary)
    unigram = RatingUnigram()
    unigram.fit(train_sequences)
    trigram = InterpolatedTrigram(lambdas)
    trigram.fit(train_sequences)

    # Referência: resenhas reais da validação, o mesmo número por nota que as amostras geradas.
    reference = validation.groupby("overall_rating").sample(
        n=args.samples_per_rating, random_state=args.seed
    )

    metrics: dict[str, object] = {
        "vocabulary_size": len(vocabulary),
        "lambdas": {"unigram": lambdas[0], "bigram": lambdas[1], "trigram": lambdas[2]},
        "lambda_tuning_perplexity": tuning_scores,
    }
    sample_rows = []
    models: list[tuple[str, InterpolatedTrigram | RatingUnigram]] = [
        ("rating_unigram", unigram),
        ("trigram", trigram),
    ]
    for name, model in models:
        samples = generate_samples(model, args.samples_per_rating, args.max_tokens, args.seed)
        requested = [rating for rating, _ in samples]
        texts = [text for _, text in samples]
        metrics[name] = {
            "validation_perplexity": round(perplexity(model, validation_sequences), 2),
            **score_texts(requested, texts, judge, train_texts),
        }
        sample_rows += [{"model": name, "requested_rating": r, "text": t} for r, t in samples]
    metrics["real_validation_reference"] = score_texts(
        reference["overall_rating"].tolist(), reference["review_text"].tolist(), judge, train_texts
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in metrics.items() if k != "lambda_tuning_perplexity"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
