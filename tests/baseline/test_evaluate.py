"""A escolha dos pesos λ do trigrama e a referência humana de cópia antes do dedupe."""

from __future__ import annotations

import pandas as pd

from nano_review.baseline.evaluate import LAMBDA_GRID, copy_before_dedupe, tune_lambdas


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


def test_copy_before_dedupe_uses_only_train_and_validation_products() -> None:
    # p1 é do treino, p2 da validação, p3 do teste. A resenha repetida de p2 conta como cópia do
    # treino. A de p3 é idêntica a uma de p2, mas o teste não pode virar "treino" da conta: se
    # entrasse, a segunda resenha de p2 também contaria como cópia.
    text = "chegou rapido e bem embalado recomendo"
    other = "nao gostei da cor do produto"
    raw = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p2", "p3", "p2"],
            "review_text": [text, text.upper() + "!!", other, other, None],
        }
    )
    train = pd.DataFrame({"product_id": ["p1"]})
    validation = pd.DataFrame({"product_id": ["p2"]})
    rates = copy_before_dedupe(raw, train, validation, max_repeat=3)
    assert rates["exact"] == 0.5  # 1 das 2 resenhas da validação com texto
    assert rates["ngram"] == 0.0  # 6 palavras: não há trecho de 8
    assert rates["any"] == 0.5
