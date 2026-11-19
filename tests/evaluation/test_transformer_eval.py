"""A avaliação do Transformer: janelas deslizantes, perplexity por palavra, amostras e limiares."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from nano_review.evaluation.transformer_eval import (
    check_hypotheses,
    generate_samples,
    sliding_windows,
    total_log_prob,
    word_perplexity,
)
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.reproducibility import set_seed
from nano_review.tokenization.bpe import train_tokenizer

TINY = TransformerConfig(
    vocab_size=30, block_size=8, d_model=16, n_layers=1, n_heads=2, dropout=0.0
)


def _model() -> TransformerLanguageModel:
    set_seed(0)
    return TransformerLanguageModel(TINY).eval()


def test_short_sequence_is_one_window_like_training() -> None:
    ids = [1, 4, 10, 11, 12, 2]  # <bos> <nota> 3 tokens <eos>
    (window,) = sliding_windows(ids, block_size=8, stride=4)
    assert window.tokens == ids[:-1]  # a entrada do treino: tudo menos o último
    assert window.offset == 1  # a posição da nota prevê o primeiro token de texto
    assert window.targets == ids[2:]


@pytest.mark.parametrize("length", [9, 10, 17, 30])
@pytest.mark.parametrize("stride", [1, 3, 6])
def test_long_sequence_scores_every_target_once_with_the_right_context(
    length: int, stride: int
) -> None:
    ids = [1, 4, *range(10, 10 + length - 3), 2]
    windows = sliding_windows(ids, block_size=8, stride=stride)

    # Todo alvo (texto e <eos>) é pontuado uma vez só, em ordem.
    assert [t for w in windows for t in w.targets] == ids[2:]
    j = 2  # índice, em ids, do próximo alvo
    for window in windows:
        assert len(window.tokens) <= 8  # cabe no modelo
        assert window.tokens[:2] == [1, 4]  # a nota continua à vista
        for k, target in enumerate(window.targets):
            assert target == ids[j]
            # A posição que prevê ids[j] precisa estar lendo ids[j - 1].
            assert window.tokens[window.offset + k] == ids[j - 1]
            j += 1


def test_stride_must_leave_room_for_the_prefix() -> None:
    with pytest.raises(ValueError):
        sliding_windows([1, 4, 10, 2], block_size=8, stride=7)


def test_window_log_prob_matches_a_direct_forward() -> None:
    # Numa sequência que cabe, a conta por janelas é a mesma do forward do treino.
    model = _model()
    ids = [1, 4, 10, 11, 12, 13, 2]
    with torch.no_grad():
        log_probs = torch.log_softmax(model(torch.tensor([ids[:-1]])), dim=-1)[0]
    expected = sum(float(log_probs[i - 1, ids[i]]) for i in range(2, len(ids)))
    result = total_log_prob(model, sliding_windows(ids, 8, 4), pad_id=0)
    assert result == pytest.approx(expected, abs=1e-5)


def test_batching_and_padding_do_not_change_the_result() -> None:
    # Janelas de tamanhos diferentes no mesmo lote: o padding à direita não pode vazar.
    model = _model()
    sequences = [[1, 4, *range(10, 10 + n), 2] for n in (1, 3, 6, 12, 20)]
    windows = [w for ids in sequences for w in sliding_windows(ids, 8, 3)]
    together = total_log_prob(model, windows, pad_id=0, batch_size=64)
    alone = total_log_prob(model, windows, pad_id=0, batch_size=1)
    assert together == pytest.approx(alone, abs=1e-4)


class _UniformModel(TransformerLanguageModel):
    """Logits todos iguais: cada token tem probabilidade 1/V, uma conta que se faz à mão."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*tokens.shape, self.config.vocab_size)


def test_word_perplexity_normalizes_by_baseline_words() -> None:
    texts = ["Produto muito bom, recomendo.", "Não gostei, veio quebrado e atrasado."]
    df = pd.DataFrame({"review_text": texts, "overall_rating": [5, 1]})
    tokenizer = train_tokenizer(texts * 5, vocab_size=60)
    vocab = tokenizer.get_vocab_size()
    config = TransformerConfig(vocab, block_size=6, d_model=8, n_layers=1, n_heads=2, dropout=0.0)
    result = word_perplexity(_UniformModel(config), tokenizer, df, stride=2)

    # log P(textos) = -(tokens BPE) * ln V. Palavras do baseline: 6 + 1 e 8 + 1 (<eos>) = 16.
    # Então a perplexity por palavra é V ** (tokens / 16), e a por token é exatamente V.
    assert result["token_perplexity"] == pytest.approx(vocab, rel=1e-3)
    tokens = result["tokens_per_word"] * 16
    assert result["word_perplexity"] == pytest.approx(vocab ** (tokens / 16), rel=1e-3)
    assert result["reviews"] == 2
    assert result["reviews_over_block_size"] >= 1  # o block_size 6 força janelas


def test_samples_are_repeatable_and_cover_every_rating() -> None:
    texts = ["Produto muito bom.", "Não gostei."]
    tokenizer = train_tokenizer(texts * 5, vocab_size=40)
    config = TransformerConfig(
        tokenizer.get_vocab_size(), block_size=10, d_model=8, n_layers=1, n_heads=2, dropout=0.0
    )
    set_seed(0)
    model = TransformerLanguageModel(config).eval()
    first = generate_samples(model, tokenizer, per_rating=2, seed=7)
    again = generate_samples(model, tokenizer, per_rating=2, seed=7)
    assert first == again
    assert [rating for rating, _ in first] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_hypotheses_use_the_charter_thresholds() -> None:
    scores: dict[str, object] = {
        "coherence": {"macro": 0.57},
        "copy": {"exact": 0.02, "ngram": 0.07},
        "distinct_2": 0.58,
    }
    result = check_hypotheses(72.0, scores)
    passed = {name: value["passed"] for name, value in result.items()}  # type: ignore[index]
    # Nos limites exatos, passa; um pouco além, não.
    assert passed == {
        "H1": True,
        "H2": True,
        "H4": False,
        "H5": True,
        "H6": False,
        "H3": "pendente",
    }
