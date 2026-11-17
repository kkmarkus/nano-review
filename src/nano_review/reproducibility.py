"""Controle de aleatoriedade para que cada treino possa ser repetido."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa a semente de todas as fontes de aleatoriedade usadas no projeto.

    Por que as três: ``random`` embaralha dados em Python puro, ``numpy`` (estado global,
    que o scikit-learn também usa) cuida do pré-processamento e o ``torch`` define a
    inicialização dos pesos, o dropout e a amostragem de texto. Se uma delas ficar solta,
    dois treinos com a mesma configuração divergem e os experimentos deixam de ser
    comparáveis no MLflow.

    Não ativamos ``torch.use_deterministic_algorithms`` de propósito: ele restringe
    operações e deixa o treino mais lento, então só vale a pena se aparecer divergência
    entre execuções com a mesma semente.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
