"""A mesma semente precisa gerar os mesmos números.

Se este teste quebrar, nenhum experimento futuro é comparável, então ele é o primeiro
alarme de que o ambiente ou o código perdeu a reprodutibilidade.
"""

import random

import numpy as np
import torch

from nano_review.reproducibility import set_seed


def _draw_samples() -> tuple[float, float, list[float]]:
    """Sorteia um valor de cada fonte de aleatoriedade controlada por ``set_seed``."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
    )


def test_same_seed_gives_same_numbers() -> None:
    set_seed(42)
    first = _draw_samples()
    set_seed(42)
    second = _draw_samples()
    assert first == second


def test_different_seeds_give_different_numbers() -> None:
    # Garante que o teste acima não passa "de graça" por uma semente ignorada.
    set_seed(1)
    first = _draw_samples()
    set_seed(2)
    second = _draw_samples()
    assert first != second
