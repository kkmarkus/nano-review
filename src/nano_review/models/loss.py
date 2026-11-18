"""A loss de modelo de linguagem e o gradiente dela calculado à mão.

A loss é a entropia cruzada média sobre as posições que contam (texto e ``<eos>``):

    loss = média sobre as posições válidas de  -log softmax(logits)[alvo]

É a mesma quantidade da perplexity do baseline: ``perplexity = exp(loss)``.

O gradiente dela em relação aos logits tem uma forma fechada simples, e é o que o ``backward()``
calcula na primeira etapa da volta:

    d loss / d logits = (softmax(logits) - one_hot(alvo)) / N

em que N é o número de posições válidas. Intuição: o gradiente **empurra para cima** o logit do
token certo (o termo -1 do one-hot) e **empurra para baixo** todos os outros, cada um na proporção
da probabilidade que o modelo deu a ele. Uma posição ignorada (``-100``) não contribui com nada.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nano_review.data.dataset import IGNORE_INDEX


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada média por token previsto.

    ``logits``: ``(B, T, V)``; ``targets``: ``(B, T)``, com ``IGNORE_INDEX`` onde não conta.
    O ``cross_entropy`` do PyTorch espera ``(N, V)`` e ``(N,)``, por isso o achatamento.
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )


def manual_loss_gradient(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """O gradiente de ``language_model_loss`` em relação aos logits, sem autograd.

    Serve para conferir, no notebook e nos testes, que o ``backward()`` calcula exatamente a
    fórmula do docstring do módulo.
    """
    valid = targets != IGNORE_INDEX
    gradient = torch.softmax(logits.detach(), dim=-1)
    # Para tirar o one-hot, o alvo ignorado (-100) precisa virar um índice válido qualquer; a
    # linha dele é zerada logo depois, então o valor não importa.
    safe_targets = targets.clamp(min=0)
    gradient.scatter_add_(-1, safe_targets.unsqueeze(-1), -torch.ones_like(gradient[..., :1]))
    gradient = gradient * valid.unsqueeze(-1)
    return gradient / valid.sum()
