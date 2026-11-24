"""Transformer só-decoder pequeno, com a atenção escrita à mão (sem ``nn.MultiheadAttention``).

Visão geral do forward, com os shapes (B = lote, T = posições, C = ``d_model``, V = vocabulário,
H = cabeças, D = C / H)::

    tokens (B, T)
      -> embedding de token + embedding de posição          (B, T, C)
      -> N blocos, cada um:
           x = x + Atenção(LayerNorm(x))                   (B, T, C)
           x = x + MLP(LayerNorm(x))                       (B, T, C)
      -> LayerNorm final                                   (B, T, C)
      -> camada de saída (pesos do embedding de token)     (B, T, V)  = logits

A diferença em relação ao bigrama e ao trigrama está na atenção: em cada posição, o modelo consulta
**todas as anteriores**, inclusive o token ``<nota_k>`` do começo, e decide quanto peso dar a cada
uma. A janela deixa de ser de 1 ou 2 tokens e passa a ser o contexto inteiro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float


class CausalSelfAttention(nn.Module):
    """Atenção multi-cabeça causal.

    Em cada posição, três projeções do mesmo vetor:

    - **query** (Q): "o que eu procuro";
    - **key** (K): "o que eu ofereço";
    - **value** (V): "o que eu passo adiante, se for escolhido".

    O peso que a posição ``i`` dá à posição ``j`` é ``softmax(Q_i · K_j / sqrt(D))``, e a saída de
    ``i`` é a média dos values ponderada por esses pesos. Com várias cabeças, cada uma faz isso num
    pedaço menor do vetor (D = C / H) e pode aprender a olhar para coisas diferentes: uma para a
    nota, outra para a palavra anterior, por exemplo.
    """

    causal_mask: torch.Tensor  # registrado como buffer no __init__

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model precisa ser divisível por n_heads")
        self.n_heads = config.n_heads
        # Q, K e V numa multiplicação só (mais rápido), separados logo depois.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        # Máscara causal: a posição i só vê as posições j <= i. Sem ela, ao prever o próximo
        # token, o modelo poderia simplesmente olhar para ele, e o treino não ensinaria nada.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        # Os pesos da última chamada, guardados só para inspeção (testes e visualização).
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, C)`` -> ``(B, T, C)``."""
        batch, length, channels = x.shape
        head_size = channels // self.n_heads

        q, k, v = self.qkv(x).split(channels, dim=2)  # 3 x (B, T, C)
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D): cada cabeça vira uma "fila" independente.
        q = q.view(batch, length, self.n_heads, head_size).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, head_size).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, head_size).transpose(1, 2)

        # Afinidade de cada posição com cada outra: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T).
        # Dividir por sqrt(D) mantém os scores numa escala estável: sem isso, com D grande, o
        # softmax satura (um peso ~1, o resto ~0) e o gradiente quase some.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        mask = self.causal_mask[:length, :length]
        # -inf vira peso 0 depois do softmax: o futuro fica invisível.
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, H, T, T); cada linha soma 1
        self.last_attention = weights.detach()

        out = self.attention_dropout(weights) @ v  # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        # Junta as cabeças de volta: (B, H, T, D) -> (B, T, H, D) -> (B, T, C).
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        result: torch.Tensor = self.residual_dropout(self.output(out))
        return result


class MLP(nn.Module):
    """Rede de duas camadas aplicada a cada posição separadamente.

    A atenção **mistura** informação entre posições; o MLP **processa** o que cada posição juntou.
    A camada do meio é 4 vezes mais larga, o padrão do Transformer original.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.layers(x)
        return result


class Block(nn.Module):
    """Um bloco do Transformer: atenção e MLP, cada um com LayerNorm antes e conexão residual.

    - **Residual** (``x = x + ...``): cada bloco só soma uma correção ao que já existe. O gradiente
      tem um caminho direto da loss até as primeiras camadas, o que torna redes fundas treináveis.
    - **LayerNorm antes** (pré-norma): normaliza a entrada de cada sub-camada, o que deixa o treino
      mais estável que a versão original, que normalizava depois.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # A atenção, sozinha, não sabe a ordem das posições (ela compara todas com todas). O
        # embedding de posição soma a cada token um vetor que diz "estou na posição t".
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Pesos amarrados: a camada de saída reutiliza a tabela de embedding. Ler o token e
        # escrevê-lo usam a mesma representação, e o modelo economiza V x C parâmetros (no modelo
        # final, 8.000 x 256 = 2.048.000, quase 40% dos 5,2 milhões), o que pesa num modelo pequeno.
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Pesos pequenos no começo: logits perto de zero, softmax quase uniforme, loss inicial
        # perto de ln(V). Valores grandes dariam opiniões fortes e aleatórias logo de saída.
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: ``(B, T)`` com ``T <= block_size`` -> logits ``(B, T, V)``.

        Padding não precisa de máscara própria: ele fica sempre **depois** do texto (à direita), e
        a máscara causal já impede que uma posição de texto olhe para frente. As saídas nas
        posições de padding são lixo, mas o alvo delas é ``-100``, então não entram na loss.
        """
        length = tokens.size(1)
        if length > self.config.block_size:
            raise ValueError(f"sequência de {length} tokens > block_size {self.config.block_size}")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)  # (B, T, C)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits: torch.Tensor = self.head(self.final_norm(x))  # (B, T, V)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Parâmetros treináveis, contando uma vez só os que são compartilhados (pesos amarrados)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def generate(
    model: TransformerLanguageModel,
    prefix: list[int],
    eos_id: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Gera token a token a partir de ``prefix`` (por exemplo ``<bos> <nota_5>``) até ``<eos>``.

    A cada passo: forward na sequência inteira, softmax dos logits da **última** posição e sorteio
    do próximo token. Gerar é repetir a previsão do treino, só que realimentando o modelo com o
    que ele mesmo escreveu. A temperatura divide os logits: abaixo de 1 o texto fica mais
    previsível, acima de 1, mais variado. Top-k e ajuste fino de temperatura ficam para a Etapa 4.
    """
    if temperature <= 0:
        raise ValueError("a temperatura precisa ser > 0")
    was_training = model.training
    model.eval()  # sem dropout na geração
    ids = list(prefix)
    # Igual ao treino: a entrada tem no máximo block_size tokens, e a saída prevê mais um. O
    # modelo não tem embedding de posição além do block_size.
    while len(ids) <= model.config.block_size:
        # Refaz o forward da sequência inteira a cada token: simples de ler, e barato com
        # resenhas curtas. (Modelos grandes guardam as keys e values já calculados.)
        logits = model(torch.tensor([ids]))[0, -1] / temperature
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator))
        ids.append(next_id)
        if next_id == eos_id:
            break
    model.train(was_training)
    return ids
