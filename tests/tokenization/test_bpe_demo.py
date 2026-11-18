"""O BPE didático num corpus de brinquedo, com as fusões calculadas à mão.

Corpus: "bom" 3 vezes, "boa" 2 vezes, "bonito" 1 vez. Com o marcador de início:

- Pares iniciais: (▁,b) 6, (b,o) 6, (o,m) 3, (o,a) 2, e os de "bonito" com 1.
- Empate em 6: vence (b,o), porque "b" vem antes de "▁" em ordem de caracteres. Fusão 1: "bo".
- Agora (▁,bo) aparece 6 vezes. Fusão 2: "▁bo".
- (▁bo,m) 3 vezes. Fusão 3: "▁bom". Depois (▁bo,a) 2 vezes. Fusão 4: "▁boa".
"""

from __future__ import annotations

from collections import Counter

from nano_review.tokenization.bpe_demo import (
    apply_merges,
    count_pairs,
    merge_pair,
    split_words,
    train_bpe,
)

CORPUS = ["bom bom bom", "boa boa", "bonito"]


def test_split_words_counts_characters_with_word_start() -> None:
    words = split_words(CORPUS)
    assert words[("▁", "b", "o", "m")] == 3
    assert words[("▁", "b", "o", "a")] == 2


def test_count_pairs_weights_by_word_frequency() -> None:
    pairs = count_pairs(split_words(CORPUS))
    assert pairs[("▁", "b")] == 6
    assert pairs[("b", "o")] == 6
    assert pairs[("o", "m")] == 3
    assert pairs[("n", "i")] == 1


def test_merge_pair_joins_every_occurrence() -> None:
    words: Counter[tuple[str, ...]] = Counter({("▁", "b", "o", "b", "o"): 1})
    assert merge_pair(words, ("b", "o")) == Counter({("▁", "bo", "bo"): 1})


def test_train_bpe_matches_hand_computed_merges() -> None:
    assert train_bpe(CORPUS, num_merges=4) == [
        ("b", "o"),
        ("▁", "bo"),
        ("▁bo", "m"),
        ("▁bo", "a"),
    ]


def test_apply_merges_on_seen_and_new_words() -> None:
    merges = train_bpe(CORPUS, num_merges=4)
    assert apply_merges("bom", merges) == ["▁bom"]
    # Palavra nova: usa as fusões que servem e deixa o resto em caracteres; nunca é "desconhecida".
    assert apply_merges("bonito", merges) == ["▁bo", "n", "i", "t", "o"]
    assert apply_merges("bolo", merges) == ["▁bo", "l", "o"]


def test_train_bpe_stops_when_nothing_left_to_merge() -> None:
    # "▁aa" vira um símbolo só depois de 2 fusões; pedir 10 não pode quebrar. Na primeira, (▁,a) e
    # (a,a) empatam com 1; vence (a,a), porque "a" vem antes de "▁" em ordem de caracteres.
    assert train_bpe(["aa"], num_merges=10) == [("a", "a"), ("▁", "aa")]
