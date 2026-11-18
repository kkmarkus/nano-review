"""Métricas do charter (seção 5): perplexity, coerência com a nota, diversidade e cópia do treino.

Ficam fora do baseline porque o Transformer (Etapas 2 a 4) é medido com as **mesmas** funções:
dois modelos só se comparam se a régua for a mesma.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from nano_review.baseline.judge import CLASSES, rating_to_sentiment
from nano_review.data.clean import dedupe_key


class LanguageModel(Protocol):
    """Qualquer modelo que dá a probabilidade da próxima palavra dado o histórico."""

    def prob(self, word: str, history: Sequence[str]) -> float: ...


def perplexity(model: LanguageModel, sequences: Iterable[Sequence[str]]) -> float:
    """exp da perda média (entropia cruzada) por token previsto.

    Intuição: perplexity 50 = o modelo fica, em média, tão indeciso quanto se escolhesse entre 50
    palavras igualmente prováveis. Quanto menor, melhor. Os tokens previstos são as palavras e o
    ``<eos>`` (saber *quando parar* também faz parte do modelo); ``<bos>`` e ``<nota_k>`` são dados.
    """
    total_log_prob = 0.0
    n_tokens = 0
    for sequence in sequences:
        for i in range(2, len(sequence)):
            total_log_prob += math.log(model.prob(sequence[i], sequence[:i]))
            n_tokens += 1
    if n_tokens == 0:
        raise ValueError("nenhum token para avaliar")
    return math.exp(-total_log_prob / n_tokens)


def words(text: str) -> list[str]:
    """Palavras em caixa baixa, sem pontuação: "Muito bom!" e "muito bom" são o mesmo texto.

    É a contagem de palavras de todas as métricas de amostra, para elas falarem da mesma coisa.
    """
    return dedupe_key(text).split()


def _ngrams(words: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """n-gramas únicos / total de n-gramas, somando todas as amostras.

    Um modelo que repete sempre "muito bom" tem distinct baixo. Só se compara com o **mesmo número
    de amostras**: quanto mais texto, mais repetição natural, e menor o valor.
    """
    all_ngrams = [gram for text in texts for gram in _ngrams(words(text), n)]
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def copy_rates(
    samples: Sequence[str], train_texts: Iterable[str], n: int = 8, min_exact_words: int = 5
) -> dict[str, float]:
    """Quanto as amostras copiam o treino. As três frações são sobre **todas** as amostras.

    - ``exact``: amostras com pelo menos ``min_exact_words`` palavras idênticas a uma resenha de
      treino (mesma chave do dedupe). Texto curto e genérico ("ótimo", "boa qualidade") coincide
      com o treino sem que nada tenha sido decorado: as 3 "cópias" do trigrama eram assim (achado
      da revisão do fim da Etapa 2). Por isso ele não conta como cópia.
    - ``ngram``: amostras com algum trecho de ``n`` palavras seguidas que existe no treino. Pega a
      cópia parcial, que o ``exact`` deixa passar. Amostras com menos de ``n`` palavras não têm
      como ter esse trecho e contam como "não copiou".
    - ``any``: ``exact`` ou ``ngram``. Uma cópia exata de 5 a 7 palavras não tem trecho de 8, então
      ``exact`` pode ser maior que ``ngram``; ``any`` junta os dois casos.
    """
    if not samples:
        raise ValueError("nenhuma amostra para avaliar")
    train_keys: set[str] = set()
    # Guardamos o hash de cada n-grama, e não o n-grama: são milhões de trechos, e um inteiro
    # ocupa bem menos memória que uma tupla de 8 palavras. Colisão de hash é desprezível aqui.
    train_ngrams: set[int] = set()
    for text in train_texts:
        text_words = words(text)
        train_keys.add(" ".join(text_words))
        train_ngrams.update(hash(gram) for gram in _ngrams(text_words, n))

    exact = ngram = either = 0
    for sample in samples:
        sample_words = words(sample)
        is_exact = len(sample_words) >= min_exact_words and " ".join(sample_words) in train_keys
        has_ngram = any(hash(gram) in train_ngrams for gram in _ngrams(sample_words, n))
        exact += is_exact
        ngram += has_ngram
        either += is_exact or has_ngram
    total = len(samples)
    return {"exact": exact / total, "ngram": ngram / total, "any": either / total}


def coherence(requested_ratings: Sequence[int], judged: Sequence[str]) -> dict[str, object]:
    """Fração de amostras que o juiz classifica na classe da nota pedida.

    Média **macro** entre as 3 classes, como define o charter: a nota 5, a mais comum, não pode
    inflar o número. A fração por nota pedida também sai, para ver onde o gerador falha.
    """
    if len(requested_ratings) != len(judged):
        raise ValueError("cada nota pedida precisa de um julgamento")
    hits_by_class: dict[str, list[bool]] = {name: [] for name in CLASSES}
    hits_by_rating: dict[int, list[bool]] = {}
    for rating, label in zip(requested_ratings, judged, strict=True):
        expected = rating_to_sentiment(rating)
        hits_by_class[expected].append(label == expected)
        hits_by_rating.setdefault(rating, []).append(label == expected)

    by_class = {name: sum(h) / len(h) for name, h in hits_by_class.items() if h}
    return {
        "macro": round(sum(by_class.values()) / len(by_class), 4),
        "by_class": {name: round(value, 4) for name, value in by_class.items()},
        "by_rating": {str(r): round(sum(h) / len(h), 4) for r, h in sorted(hits_by_rating.items())},
    }
