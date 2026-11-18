"""Bigrama neural: a rede mais simples que aprende a prever o próximo token.

Ela é uma tabela ``V x V`` de números livres: a linha ``i`` guarda os logits do token que vem depois
do token ``i``. É o mesmo modelo do bigrama de contagem, mas os números são **aprendidos por
gradiente** em vez de contados. Por isso ele é a ponte entre o baseline de n-gramas e o
Transformer: o ciclo forward -> loss -> backward -> atualização aparece aqui sem nada mais em volta.

Ele só olha **um** token para trás. Logo, a nota (``<nota_k>``) só influencia a primeira palavra
da resenha; depois disso, a rede já não sabe qual nota foi pedida.
"""

from __future__ import annotations

import torch
from torch import nn


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
        # Começo pequeno e quase igual para todos: com logits iguais, o softmax dá 1/V a cada
        # token, e a loss inicial esperada é ln(V). Com o padrão do nn.Embedding (desvio 1), a
        # rede começaria com opiniões fortes e aleatórias, e a loss inicial seria maior.
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: ``(B, T)`` -> logits ``(B, T, V)``.

        Cada posição consulta a linha da tabela correspondente ao seu token: o forward inteiro é
        uma busca na tabela, sem nenhuma conta entre posições.
        """
        logits: torch.Tensor = self.table(tokens)
        return logits
