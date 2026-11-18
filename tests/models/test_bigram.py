"""Forward, loss inicial, backward conferido à mão e descida do gradiente no bigrama."""

from __future__ import annotations

import math

import pytest
import torch

from nano_review.data.dataset import IGNORE_INDEX
from nano_review.models.bigram import BigramLanguageModel
from nano_review.models.loss import language_model_loss, manual_loss_gradient

VOCAB = 50


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (4, 7), generator=generator)
    y = torch.randint(0, VOCAB, (4, 7), generator=generator)
    y[:, 0] = IGNORE_INDEX  # como no collate: a posição da nota não conta
    y[3, 4:] = IGNORE_INDEX  # como no collate: padding
    return x, y


def test_forward_shape() -> None:
    x, _ = _batch()
    assert BigramLanguageModel(VOCAB)(x).shape == (4, 7, VOCAB)


def test_initial_loss_is_close_to_log_vocab() -> None:
    # Sem treino, o modelo não tem opinião: ~1/V para cada token, e a loss é ln(V).
    torch.manual_seed(0)
    x, y = _batch()
    loss = language_model_loss(BigramLanguageModel(VOCAB)(x), y)
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.05)


def test_manual_gradient_matches_autograd() -> None:
    x, y = _batch()
    logits = torch.randn(4, 7, VOCAB, requires_grad=True)
    language_model_loss(logits, y).backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, manual_loss_gradient(logits, y), atol=1e-7)
    # Posições ignoradas não recebem gradiente nenhum.
    assert torch.all(logits.grad[y == IGNORE_INDEX] == 0)


def test_gradient_reaches_only_the_table_rows_that_were_used() -> None:
    # A regra da cadeia através da busca na tabela: a linha i recebe a soma dos gradientes de
    # todas as posições em que o token i apareceu na entrada. Linhas não usadas recebem zero.
    torch.manual_seed(0)
    x, y = _batch()
    model = BigramLanguageModel(VOCAB)
    logits = model(x)
    language_model_loss(logits, y).backward()

    expected = torch.zeros(VOCAB, VOCAB)
    expected.index_add_(0, x.reshape(-1), manual_loss_gradient(logits, y).reshape(-1, VOCAB))
    assert model.table.weight.grad is not None
    assert torch.allclose(model.table.weight.grad, expected, atol=1e-7)


def test_manual_gradient_descent_lowers_the_loss() -> None:
    # O passo de atualização escrito à mão, sem otimizador: peso -= taxa * gradiente.
    torch.manual_seed(0)
    x, y = _batch()
    model = BigramLanguageModel(VOCAB)
    losses = []
    for _ in range(20):
        loss = language_model_loss(model(x), y)
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                assert parameter.grad is not None
                parameter.sub_(5.0 * parameter.grad)
    assert losses[-1] < losses[0] - 0.5
