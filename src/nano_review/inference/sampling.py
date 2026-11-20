"""Amostragem com temperatura e top-k: o "botão" de qualidade da geração.

O modelo devolve, a cada passo, uma nota (logit) para cada token do vocabulário. Sortear direto
dessa distribuição (temperatura 1, sem top-k) é a medida honesta do modelo, usada na Etapa 3. Para
o produto, dá para trocar um pouco de variedade por texto mais seguro:

- **temperatura** divide os logits antes do softmax. Abaixo de 1, os tokens prováveis ficam ainda
  mais prováveis, e os raros quase somem;
- **top-k** guarda só os ``k`` tokens de maior logit e zera a chance de todos os outros. Isso corta
  de uma vez a cauda de tokens improváveis, de onde vêm as palavras inventadas
  (``docs/error_analysis.md``, causa A).

A função vive aqui, e não em ``models/transformer.py``, para não mudar o código do treino: o DVC
pediria para retreinar o modelo por causa de uma função que o treino nem usa.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nano_review.models.transformer import TransformerLanguageModel


def filter_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Mantém os ``k`` maiores logits e põe ``-inf`` nos outros; ``None`` não filtra nada.

    Depois do softmax, ``-inf`` vira probabilidade 0: o token simplesmente não pode ser sorteado.
    Em empate na fronteira, ficam todos os que empatam com o k-ésimo (nenhum é sorteado à toa).
    """
    if k is None or k >= logits.size(-1):
        return logits
    if k < 1:
        raise ValueError("o top-k precisa ser >= 1 (ou None, para não filtrar)")
    kth = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def sample(
    model: TransformerLanguageModel,
    prefix: list[int],
    eos_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
    max_new_tokens: int | None = None,
) -> list[int]:
    """Gera token a token a partir de ``prefix`` até ``<eos>``, até encher o contexto ou até sortear
    ``max_new_tokens`` tokens (contando o ``<eos>``).

    É o mesmo laço de ``models.transformer.generate``, com o top-k e o limite a mais. Com
    temperatura 1, sem top-k, sem limite e com o mesmo ``generator``, as duas funções sorteiam
    exatamente os mesmos tokens (há um teste para isso): a régua da Etapa 3 continua valendo.
    """
    if temperature <= 0:
        raise ValueError("a temperatura precisa ser > 0")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens precisa ser >= 1 (ou None, para não limitar)")
    limit = len(prefix) + max_new_tokens if max_new_tokens is not None else None
    was_training = model.training
    model.eval()  # sem dropout na geração
    ids = list(prefix)
    with torch.no_grad():
        while len(ids) <= model.config.block_size and (limit is None or len(ids) < limit):
            logits = model(torch.tensor([ids]))[0, -1] / temperature
            logits = filter_top_k(logits, top_k)
            next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator))
            ids.append(next_id)
            if next_id == eos_id:
                break
    model.train(was_training)
    return ids
