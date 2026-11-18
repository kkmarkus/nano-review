"""Split por produto: de ``data/processed`` para ``data/curated/{train,validation,test}.parquet``.

Por que por produto, e não por linha: resenhas do mesmo produto repetem nome, defeito e
vocabulário. Se o mesmo produto estivesse no treino e no teste, o teste mediria memória do
produto, e não generalização.

**O ``test.parquet`` é gravado aqui, mas nada no projeto o lê até a Etapa 4.** Ele é aberto uma
única vez, no fim; qualquer decisão tomada olhando para ele invalida a avaliação.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.data.split --input data/processed/reviews.parquet \\
        --output-dir data/curated --metrics reports/metrics/split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")


def assign_splits(
    product_ids: pd.Series[str], validation_fraction: float, test_fraction: float, seed: int
) -> pd.Series[str]:
    """Devolve o nome do conjunto de cada linha, com todas as linhas de um produto juntas.

    Os produtos são embaralhados e vão enchendo treino, validação e teste até a contagem de
    **resenhas** chegar a cada fração. Sortear uma fração de *produtos* não bastaria: com a cauda
    longa do B2W (um produto tem quase 800 resenhas e 71% têm uma só), 10% dos produtos pode
    significar bem mais ou bem menos que 10% das resenhas.
    """
    train_fraction = 1.0 - validation_fraction - test_fraction
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and train_fraction > 0):
        raise ValueError(
            "as frações de validação e teste precisam ser positivas e somar menos de 1"
        )

    # Uma linha sem produto não teria conjunto e sumiria em silêncio dos três arquivos.
    if product_ids.isna().any():
        raise ValueError("há linhas sem product_id; o split por produto não sabe onde colocá-las")

    # sort_index antes de embaralhar: a ordem de entrada do value_counts entre produtos com a
    # mesma contagem não é garantida, e a mesma semente precisa dar o mesmo split sempre.
    reviews_per_product = product_ids.value_counts().sort_index()
    rng = np.random.default_rng(seed)
    shuffled = reviews_per_product.iloc[rng.permutation(len(reviews_per_product))]

    # Fração acumulada de resenhas *depois* de incluir cada produto. Um produto que cruza a
    # fronteira vai inteiro para o conjunto seguinte, por isso as frações ficam aproximadas.
    cumulative = shuffled.cumsum() / shuffled.sum()
    split_names = np.where(
        cumulative <= train_fraction,
        "train",
        np.where(cumulative <= train_fraction + validation_fraction, "validation", "test"),
    )
    split_of_product = pd.Series(split_names, index=shuffled.index)
    return product_ids.map(split_of_product)


def describe_split(df: pd.DataFrame, split: pd.Series[str]) -> dict[str, dict[str, object]]:
    """Resenhas, produtos e distribuição de notas de cada conjunto.

    Serve para conferir que os três conjuntos se parecem: se a validação tivesse muito mais nota 5
    que o treino, as métricas dela não diriam nada sobre o modelo.
    """
    summary: dict[str, dict[str, object]] = {}
    for name in SPLITS:
        part = df[split == name]
        ratings = part["overall_rating"].value_counts(normalize=True).sort_index()
        summary[name] = {
            "rows": len(part),
            "row_fraction": round(len(part) / len(df), 4),
            "products": int(part["product_id"].nunique()),
            "rating_fraction": {str(k): round(float(v), 4) for k, v in ratings.items()},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    # Sem defaults: os valores vêm só do params.yaml, para não existirem dois lugares para mudar.
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--test-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    split = assign_splits(df["product_id"], args.validation_fraction, args.test_fraction, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        df[split == name].to_parquet(args.output_dir / f"{name}.parquet", index=False)

    metrics = describe_split(df, split)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
