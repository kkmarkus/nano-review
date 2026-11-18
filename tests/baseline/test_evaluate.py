"""A escolha dos pesos λ do trigrama."""

from __future__ import annotations

import pandas as pd

from nano_review.baseline.evaluate import LAMBDA_GRID, tune_lambdas


def test_tune_lambdas_survives_words_only_in_held_out_products() -> None:
    # Regressão de um bug achado ao rodar com os dados reais. Cada produto tem uma palavra só dele,
    # repetida duas vezes: ela entra no vocabulário do treino inteiro, mas some do modelo de ajuste
    # quando o produto cai na fatia separada. Com o vocabulário errado, a probabilidade dela era 0,
    # e a perplexity quebrava com log(0).
    rows = []
    for i in range(10):
        rows += [(f"p{i}", 5, f"bom produto palavra{i}")] * 2
    train = pd.DataFrame(rows, columns=["product_id", "overall_rating", "review_text"])
    lambdas, scores = tune_lambdas(train, vocab_min_count=2, tuning_fraction=0.3, seed=0)

    assert lambdas in LAMBDA_GRID
    assert len(scores) == len(LAMBDA_GRID)
