"""O Transformer: shapes, causalidade, atenção, loss inicial e a capacidade de aprender um lote."""

from __future__ import annotations

import math

import pytest
import torch

from nano_review.models.loss import language_model_loss
from nano_review.models.transformer import (
    CausalSelfAttention,
    TransformerConfig,
    TransformerLanguageModel,
    count_parameters,
)
from nano_review.reproducibility import set_seed

CONFIG = TransformerConfig(
    vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0
)


def _model(config: TransformerConfig = CONFIG) -> TransformerLanguageModel:
    set_seed(0)
    return TransformerLanguageModel(config).eval()


def _tokens(batch: int = 3, length: int = 10) -> torch.Tensor:
    return torch.randint(
        0, CONFIG.vocab_size, (batch, length), generator=torch.Generator().manual_seed(1)
    )


def test_forward_shape() -> None:
    assert _model()(_tokens()).shape == (3, 10, CONFIG.vocab_size)


def test_rejects_sequence_longer_than_block() -> None:
    with pytest.raises(ValueError):
        _model()(_tokens(length=CONFIG.block_size + 1))


def test_future_tokens_do_not_change_past_predictions() -> None:
    # A propriedade que torna o treino honesto: a previsão da posição t só pode depender dos
    # tokens até t. Se a máscara causal falhasse, mudar o token 6 mudaria os logits de 0 a 5.
    model = _model()
    tokens = _tokens()
    changed = tokens.clone()
    changed[:, 6] = (changed[:, 6] + 1) % CONFIG.vocab_size
    with torch.no_grad():
        before, after = model(tokens), model(changed)
    assert torch.allclose(before[:, :6], after[:, :6], atol=1e-6)
    assert not torch.allclose(before[:, 6:], after[:, 6:])


def test_padding_on_the_right_does_not_change_real_positions() -> None:
    # No lote, resenhas curtas são completadas com <pad> à direita. As posições de texto não podem
    # mudar por causa disso: a máscara causal impede que elas olhem para o padding, que vem depois.
    model = _model()
    alone = _tokens(batch=1, length=6)
    padded = torch.cat([alone, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(alone), model(padded)[:, :6], atol=1e-6)


def test_attention_rows_sum_to_one_and_ignore_the_future() -> None:
    model = _model()
    with torch.no_grad():
        model(_tokens())
    attention = model.blocks[0].attention
    assert isinstance(attention, CausalSelfAttention)
    weights = attention.last_attention
    assert weights is not None
    assert weights.shape == (3, CONFIG.n_heads, 10, 10)  # (B, H, T, T)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, CONFIG.n_heads, 10))
    future = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[..., future] == 0)


def test_initial_loss_is_close_to_log_vocab() -> None:
    tokens = _tokens()
    targets = torch.roll(tokens, -1, dims=1)
    loss = language_model_loss(_model()(tokens), targets)
    assert loss.item() == pytest.approx(math.log(CONFIG.vocab_size), abs=0.1)


def test_output_layer_shares_weights_with_token_embedding() -> None:
    model = _model()
    assert model.head.weight is model.token_embedding.weight
    # Compartilhado = contado uma vez só.
    unique = {id(p): p for p in model.parameters()}
    assert count_parameters(model) == sum(p.numel() for p in unique.values())


def test_rejects_heads_that_do_not_divide_d_model() -> None:
    with pytest.raises(ValueError):
        TransformerLanguageModel(
            TransformerConfig(
                vocab_size=50, block_size=8, d_model=30, n_layers=1, n_heads=4, dropout=0.0
            )
        )


def test_overfits_a_single_batch() -> None:
    # O teste clássico de que forward, backward e otimizador estão ligados certo: um modelo com
    # capacidade de sobra precisa conseguir decorar um lote só. Se não decora, há bug no caminho.
    model = _model().train()
    tokens = _tokens(batch=4, length=12)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        loss = language_model_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1
