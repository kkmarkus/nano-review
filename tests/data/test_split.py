"""O split por produto precisa ser completo, sem vazamento e repetível.

O dado de teste imita a cauda longa do B2W: muitos produtos com uma resenha e alguns gigantes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nano_review.data.split import SPLITS, assign_splits, describe_split


@pytest.fixture
def product_ids() -> pd.Series[str]:
    giants = [f"g{i}" for i in range(5) for _ in range(40)]  # 5 produtos x 40 resenhas
    singles = [f"s{i}" for i in range(800)]  # 800 produtos x 1 resenha
    return pd.Series(giants + singles, dtype=str)


def test_every_row_gets_a_split(product_ids: pd.Series[str]) -> None:
    split = assign_splits(product_ids, 0.1, 0.1, seed=42)
    assert split.notna().all()
    assert set(split) == set(SPLITS)


def test_no_product_in_two_splits(product_ids: pd.Series[str]) -> None:
    # A propriedade que justifica o split por produto: se falhar, há vazamento.
    split = assign_splits(product_ids, 0.1, 0.1, seed=42)
    splits_per_product = split.groupby(product_ids).nunique()
    assert (splits_per_product == 1).all()


def test_row_fractions_close_to_target(product_ids: pd.Series[str]) -> None:
    split = assign_splits(product_ids, 0.1, 0.1, seed=42)
    fractions = split.value_counts(normalize=True)
    # Tolerância de um produto gigante (40 de 1000 linhas): é o máximo que a fronteira desloca.
    assert fractions["train"] == pytest.approx(0.8, abs=0.04)
    assert fractions["validation"] == pytest.approx(0.1, abs=0.04)
    assert fractions["test"] == pytest.approx(0.1, abs=0.04)


def test_same_seed_same_split(product_ids: pd.Series[str]) -> None:
    first = assign_splits(product_ids, 0.1, 0.1, seed=7)
    # A ordem das linhas muda, mas o split de cada produto não pode mudar.
    shuffled = product_ids.sample(frac=1, random_state=0)
    second = assign_splits(shuffled, 0.1, 0.1, seed=7)
    assert first.groupby(product_ids).first().equals(second.groupby(shuffled).first())


def test_different_seed_different_split(product_ids: pd.Series[str]) -> None:
    # Garante que o teste acima não passa "de graça" por uma semente ignorada.
    first = assign_splits(product_ids, 0.1, 0.1, seed=1)
    second = assign_splits(product_ids, 0.1, 0.1, seed=2)
    assert not first.equals(second)


@pytest.mark.parametrize(("validation", "test"), [(0.0, 0.1), (0.5, 0.5), (-0.1, 0.1)])
def test_rejects_invalid_fractions(
    product_ids: pd.Series[str], validation: float, test: float
) -> None:
    with pytest.raises(ValueError):
        assign_splits(product_ids, validation, test, seed=42)


def test_rejects_missing_product_id() -> None:
    # Sem o erro, a linha ficaria sem conjunto e sumiria em silêncio dos três arquivos.
    with pytest.raises(ValueError):
        assign_splits(pd.Series(["a", "b", None] * 10), 0.1, 0.1, seed=1)


def test_describe_split_counts_rows_and_products() -> None:
    df = pd.DataFrame(
        {"product_id": ["a", "a", "b", "c"], "overall_rating": [5, 5, 1, 3]},
    )
    split = pd.Series(["train", "train", "validation", "test"])
    summary = describe_split(df, split)
    assert summary["train"]["rows"] == 2
    assert summary["train"]["products"] == 1
    assert summary["train"]["rating_fraction"] == {"5": 1.0}
    assert summary["validation"]["row_fraction"] == 0.25
