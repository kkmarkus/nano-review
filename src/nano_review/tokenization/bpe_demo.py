"""BPE escrito à mão, só para entender o algoritmo (o treino real usa a biblioteca ``tokenizers``).

A ideia do Byte Pair Encoding cabe em três linhas:

1. Comece com cada palavra quebrada em caracteres.
2. Conte todos os pares de símbolos vizinhos e **funda o par mais frequente** num símbolo novo.
3. Repita até ter o número de fusões desejado.

Pedaços frequentes ("▁bo", "ção", "▁recomendo") viram um símbolo só; palavras raras continuam
quebradas em pedaços menores, mas nunca ficam "desconhecidas". É o meio-termo entre tokenizar por
caractere (sequências longas demais) e por palavra (vocabulário enorme e muitas palavras novas).

O "▁" marca o começo de palavra, como no pré-tokenizador ``Metaspace`` que o projeto usa: assim
"bom" no início de uma palavra e "bom" dentro de "abombado" viram tokens diferentes.
"""

from __future__ import annotations

from collections import Counter

WORD_START = "▁"


def split_words(corpus: list[str]) -> Counter[tuple[str, ...]]:
    """Conta cada palavra do corpus já quebrada em caracteres, com o marcador de início."""
    return Counter(tuple(WORD_START + word) for text in corpus for word in text.split())


def count_pairs(words: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Quantas vezes cada par de símbolos vizinhos aparece, pesado pela frequência da palavra."""
    pairs: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in words.items():
        for pair in zip(symbols, symbols[1:], strict=False):
            pairs[pair] += frequency
    return pairs


def merge_pair(words: Counter[tuple[str, ...]], pair: tuple[str, str]) -> Counter[tuple[str, ...]]:
    """Substitui toda ocorrência do par pelo símbolo fundido."""
    merged: Counter[tuple[str, ...]] = Counter()
    for symbols, frequency in words.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        merged[tuple(new_symbols)] += frequency
    return merged


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Aprende a lista de fusões, na ordem em que aconteceram."""
    words = split_words(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(words)
        if not pairs:
            break
        # Empate de frequência: o par que vem primeiro em ordem alfabética, para o resultado não
        # depender da ordem de um dicionário.
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        words = merge_pair(words, best)
        merges.append(best)
    return merges


def apply_merges(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokeniza uma palavra repetindo as fusões aprendidas, na mesma ordem do treino."""
    words: Counter[tuple[str, ...]] = Counter({tuple(WORD_START + word): 1})
    for pair in merges:
        words = merge_pair(words, pair)
    return list(next(iter(words)))
