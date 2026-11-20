"""Escolha da decodificação: a regra de docs/inference.md, a taxa de inventadas e as amostras."""

from __future__ import annotations

import torch

from nano_review.baseline.judge import build_judge
from nano_review.evaluation.decoding import choose, generate_with, invented_rate, label, measure
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.tokenization.bpe import train_tokenizer


def _result(name: str, passes: bool, invented: float, distinct: float) -> dict[str, object]:
    return {
        "label": name,
        "passes_all": passes,
        "invented_per_100_words": invented,
        "distinct_2": distinct,
    }


def test_choice_needs_all_four_hypotheses_then_fewest_invented_words() -> None:
    results = [
        _result("reprova", False, 0.1, 0.9),  # a menor taxa, mas não passa: fica de fora
        _result("muitas", True, 2.0, 0.8),
        _result("poucas", True, 1.0, 0.6),
    ]
    assert choose(results) == results[2]


def test_tie_on_invented_words_goes_to_the_higher_distinct_2() -> None:
    results = [_result("a", True, 1.0, 0.60), _result("b", True, 1.0, 0.70)]
    assert choose(results) == results[1]


def test_no_eligible_combination_returns_none() -> None:
    assert choose([_result("a", False, 1.0, 0.7)]) is None


def test_invented_rate_by_hand() -> None:
    vocabulary = {"produto", "bom", "chegou", "dia"}
    # 6 palavras ("29" não conta), 2 inventadas ("xyzq" e "abcd"): 33,333 por 100 palavras.
    assert invented_rate(["Produto bom xyzq", "chegou dia 29 abcd"], vocabulary) == 33.333
    assert invented_rate([""], vocabulary) == 0.0


def test_labels_are_short_and_zero_means_no_top_k() -> None:
    assert label(0.8, 50) == "t0.8-k50"
    assert label(1.0, None) == "t1.0-k0"


def test_measure_reports_the_four_sample_hypotheses() -> None:
    texts = ["Produto ótimo, adorei", "Péssimo, não funciona", "Razoável, esperava mais"]
    judge = build_judge(ngram_max=1, min_df=1, c=1.0).fit(
        texts * 4, ["positive", "negative", "neutral"] * 4
    )
    samples = [(5, texts[0]), (1, texts[1]), (3, texts[2])]
    result = measure(samples, judge, train_texts=["outra coisa qualquer"], vocabulary={"produto"})
    passed = result["passed"]
    assert isinstance(passed, dict)
    assert set(passed) == {"H2", "H4", "H5", "H6"}
    assert passed["H2"] is True  # o juiz acerta as três
    assert result["passes_all"] == all(passed.values())
    # Só "produto" está no vocabulário: 8 das 9 palavras são "inventadas".
    assert result["invented_per_100_words"] == round(100 * 8 / 9, 3)


def test_generation_uses_every_rating_and_is_repeatable() -> None:
    tokenizer = train_tokenizer(["Produto muito bom.", "Não gostei."] * 5, vocab_size=40)
    config = TransformerConfig(
        tokenizer.get_vocab_size(), block_size=10, d_model=8, n_layers=1, n_heads=2, dropout=0.0
    )
    torch.manual_seed(0)
    model = TransformerLanguageModel(config).eval()
    first = generate_with(model, tokenizer, per_rating=2, seed=7, temperature=0.8, top_k=5)
    again = generate_with(model, tokenizer, per_rating=2, seed=7, temperature=0.8, top_k=5)
    assert first == again
    assert [rating for rating, _ in first] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
