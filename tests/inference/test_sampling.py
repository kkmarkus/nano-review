"""Amostragem da inferência: top-k, temperatura e equivalência com a régua da Etapa 3."""

from __future__ import annotations

import pytest
import torch

from nano_review.inference.sampling import filter_top_k, sample
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel, generate

CONFIG = TransformerConfig(
    vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0
)
EOS = 1


def _model() -> TransformerLanguageModel:
    torch.manual_seed(0)
    return TransformerLanguageModel(CONFIG)


def test_top_k_keeps_only_the_k_largest_logits() -> None:
    logits = torch.tensor([0.5, 3.0, -1.0, 2.0, 1.0])
    kept = filter_top_k(logits, 2)
    assert torch.isfinite(kept).tolist() == [False, True, False, True, False]
    assert kept[1] == 3.0 and kept[3] == 2.0  # os que ficam não mudam
    # Depois do softmax, os cortados têm probabilidade exatamente 0.
    assert torch.softmax(kept, dim=-1)[[0, 2, 4]].sum() == 0.0


def test_top_k_keeps_every_logit_tied_with_the_kth() -> None:
    logits = torch.tensor([2.0, 2.0, 1.0])
    assert torch.isfinite(filter_top_k(logits, 1)).tolist() == [True, True, False]


def test_top_k_none_or_larger_than_the_vocabulary_changes_nothing() -> None:
    logits = torch.tensor([0.5, 3.0, -1.0])
    assert torch.equal(filter_top_k(logits, None), logits)
    assert torch.equal(filter_top_k(logits, 3), logits)
    assert torch.equal(filter_top_k(logits, 10), logits)
    with pytest.raises(ValueError, match="top-k"):
        filter_top_k(logits, 0)


def test_without_top_k_at_temperature_1_it_matches_the_stage_3_generator() -> None:
    # A régua da Etapa 3 (models.transformer.generate) e a inferência sorteiam os mesmos tokens.
    model = _model()
    ours = sample(model, [2, 3], EOS, generator=torch.Generator().manual_seed(7))
    theirs = generate(model, [2, 3], EOS, generator=torch.Generator().manual_seed(7))
    assert ours == theirs


def test_top_k_1_is_greedy_and_ignores_the_seed() -> None:
    # Com k = 1 só sobra o token mais provável: o sorteio deixa de depender da semente.
    model = _model()
    first = sample(model, [2, 3], EOS, top_k=1, generator=torch.Generator().manual_seed(1))
    second = sample(model, [2, 3], EOS, top_k=1, generator=torch.Generator().manual_seed(99))
    assert first == second
    with torch.no_grad():
        best = int(model(torch.tensor([[2, 3]]))[0, -1].argmax())
    assert first[2] == best


def test_generation_stops_at_eos_or_when_the_context_is_full() -> None:
    ids = sample(_model(), [2, 3], EOS, generator=torch.Generator().manual_seed(3))
    assert len(ids) <= CONFIG.block_size + 1
    assert ids[-1] == EOS or len(ids) == CONFIG.block_size + 1
    assert EOS not in ids[2:-1]  # nada depois de um <eos>


def test_max_new_tokens_stops_early_and_counts_the_eos() -> None:
    ids = sample(
        _model(), [2, 3], EOS, generator=torch.Generator().manual_seed(3), max_new_tokens=4
    )
    assert 3 <= len(ids) <= 2 + 4
    # Com o mesmo gerador, o começo é o mesmo da geração sem limite: o limite só corta.
    full = sample(_model(), [2, 3], EOS, generator=torch.Generator().manual_seed(3))
    assert ids == full[: len(ids)]
    with pytest.raises(ValueError, match="max_new_tokens"):
        sample(_model(), [2, 3], EOS, max_new_tokens=0)


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperatura"):
        sample(_model(), [2, 3], EOS, temperature=0.0)
