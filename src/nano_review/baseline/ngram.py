"""Geradores de n-gramas de palavras: o piso (unigrama por nota) e o baseline (trigrama).

Os dois recebem a nota do mesmo jeito que o Transformer vai receber: como um token de prefixo,
na sequência ``<bos> <nota_k> palavra palavra ... <eos>``.

Por que o trigrama é o baseline certo: ele é o "modelo de linguagem mais simples que funciona".
Ele só olha as **duas** palavras anteriores para prever a próxima. Logo depois do início, o token
``<nota_k>`` ainda está nessa janela; da terceira palavra em diante, a nota sai da visão do
modelo, e o resto da resenha é escrito sem saber qual nota foi pedida. O Transformer vê o token de
nota em toda posição, pela atenção. Se ele não for mais coerente que o trigrama, a atenção não está
servindo para o que deveria.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

# Palavra (letras, dígitos) ou um sinal de pontuação isolado: "bom!!" vira "bom", "!", "!".
TOKEN = re.compile(r"\w+|[^\w\s]")
# Na volta para texto, pontuação de fechamento cola na palavra anterior: "bom ," -> "bom,".
SPACE_BEFORE_CLOSING = re.compile(r" ([.,!?;:%)\]}…])")
SPACE_AFTER_OPENING = re.compile(r"([(\[{]) ")


def rating_token(rating: int) -> str:
    """Token de prefixo que carrega a nota pedida, por exemplo ``<nota_5>``."""
    return f"<nota_{rating}>"


def word_tokenize(text: str) -> list[str]:
    """Quebra o texto em palavras e pontuação, mantendo a caixa.

    Tokenização provisória, só para o baseline. Na Etapa 2 o Transformer usa BPE, e por isso a
    perplexity daqui (por palavra) não se compara diretamente com a dele (por token BPE).
    """
    return TOKEN.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    """Junta os tokens de volta num texto legível."""
    text = " ".join(tokens)
    text = SPACE_BEFORE_CLOSING.sub(r"\1", text)
    return SPACE_AFTER_OPENING.sub(r"\1", text)


def build_vocabulary(tokenized_texts: Iterable[Sequence[str]], min_count: int) -> set[str]:
    """Palavras que aparecem pelo menos ``min_count`` vezes no treino.

    O resto vira ``<unk>``. Sem isso, toda palavra nova da validação teria probabilidade zero, e a
    perplexity seria infinita. Com ``<unk>``, o modelo aprende "quanto de palavra rara costuma
    aparecer" a partir das palavras raras do próprio treino.
    """
    counts = Counter(token for tokens in tokenized_texts for token in tokens)
    return {token for token, count in counts.items() if count >= min_count}


def to_sequence(tokens: Sequence[str], rating: int, vocabulary: set[str]) -> list[str]:
    """Monta a sequência de treino: ``<bos> <nota_k> palavras... <eos>``."""
    words = [token if token in vocabulary else UNK for token in tokens]
    return [BOS, rating_token(rating), *words, EOS]


class _Sampler:
    """Sorteia a próxima palavra a partir de contagens, guardando os vetores já montados.

    Montar o vetor de probabilidades de um contexto custa caro; durante a geração, os mesmos
    contextos ("<bos> <nota_5>", por exemplo) se repetem muito, então vale guardar.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, ...], tuple[list[str], np.ndarray]] = {}

    def sample(
        self, key: tuple[str, ...], get_counts: Callable[[], Counter[str]], rng: np.random.Generator
    ) -> str:
        """``get_counts`` só é chamada se o contexto ainda não está guardado."""
        if key not in self._cache:
            counts = get_counts()
            words = list(counts)
            weights = np.array([counts[word] for word in words], dtype=float)
            self._cache[key] = (words, weights / weights.sum())
        words, probabilities = self._cache[key]
        return words[int(rng.choice(len(words), p=probabilities))]


class InterpolatedTrigram:
    """Trigrama de palavras com suavização por interpolação linear.

    P(w | a, b) = λ3·P_tri(w | a, b) + λ2·P_bi(w | b) + λ1·P_uni(w)

    P_tri e P_bi são frequências relativas contadas no treino. A mistura resolve o problema do
    trigrama puro: a maioria das sequências de três palavras da validação nunca apareceu no treino.
    Quando o trigrama não sabe nada, o bigrama e o unigrama ainda dão uma probabilidade razoável.

    P_uni tem suavização de Laplace (+1 em toda contagem), para que **nenhum** token possível tenha
    probabilidade zero, nem um ``<unk>`` que por acaso não apareceu no treino. Uma probabilidade
    zero faria a perplexity ser infinita por causa de uma única palavra.
    """

    def __init__(self, lambdas: tuple[float, float, float]) -> None:
        """``lambdas`` = (λ1 unigrama, λ2 bigrama, λ3 trigrama); precisam somar 1."""
        if abs(sum(lambdas) - 1.0) > 1e-9 or min(lambdas) < 0:
            raise ValueError("os pesos da interpolação precisam ser >= 0 e somar 1")
        # O unigrama é a única ordem que existe para qualquer histórico. Com peso zero nele, um
        # histórico todo desconhecido não teria distribuição nenhuma (divisão por zero).
        if lambdas[0] <= 0:
            raise ValueError("o peso do unigrama precisa ser > 0")
        self.lambdas = lambdas
        # counts[n][contexto] = Counter da palavra seguinte, com contexto de n palavras (0, 1, 2).
        self.counts: list[dict[tuple[str, ...], Counter[str]]] = [{}, {}, {}]
        self.totals: list[dict[tuple[str, ...], int]] = [{}, {}, {}]
        # Tudo o que o modelo pode prever; o tamanho entra no denominador do Laplace.
        self.targets: set[str] = {UNK, EOS}
        self._sampler = _Sampler()

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        for sequence in sequences:
            # A partir da posição 2: <bos> e <nota_k> são dados, nunca previstos.
            for i in range(2, len(sequence)):
                word = sequence[i]
                self.targets.add(word)
                for order in range(3):
                    context = tuple(sequence[i - order : i])
                    self.counts[order].setdefault(context, Counter())[word] += 1
                    self.totals[order][context] = self.totals[order].get(context, 0) + 1

    def _available(self, history: Sequence[str]) -> list[tuple[float, int, tuple[str, ...]]]:
        """Pesos e contextos que o treino conhece para esta história.

        Um contexto nunca visto não tem distribuição. O peso dele é redistribuído para as ordens
        menores (dividindo pela soma dos pesos disponíveis), para que as probabilidades continuem
        somando 1.
        """
        available = []
        for order, weight in enumerate(self.lambdas):
            context = tuple(history[len(history) - order :]) if order else ()
            if weight > 0 and context in self.counts[order]:
                available.append((weight, order, context))
        return available

    def prob(self, word: str, history: Sequence[str]) -> float:
        available = self._available(history)
        total_weight = sum(weight for weight, _, _ in available)
        probability = 0.0
        for weight, order, context in available:
            count = self.counts[order][context][word]
            total = self.totals[order][context]
            if order == 0:
                probability += weight * (count + 1) / (total + len(self.targets))
            else:
                probability += weight * count / total
        return probability / total_weight

    def sample_next(self, history: Sequence[str], rng: np.random.Generator) -> str:
        """Sorteia da mistura sem montar a distribuição inteira.

        Truque: sortear "de qual ordem tirar" (com os pesos λ) e depois "qual palavra" (pelas
        contagens daquela ordem) dá exatamente a mesma distribuição da fórmula acima, mas custa
        só o tamanho de um contexto, e não do vocabulário inteiro.
        """
        available = self._available(history)
        weights = np.array([weight for weight, _, _ in available])
        _, order, context = available[int(rng.choice(len(available), p=weights / weights.sum()))]
        if order == 0:
            # O unigrama sorteia com o mesmo +1 do Laplace usado em prob().
            return self._sampler.sample(
                ("0",), lambda: _plus_one(self.counts[0][()], self.targets), rng
            )
        return self._sampler.sample(
            (str(order), *context), lambda: self.counts[order][context], rng
        )

    def generate(self, rating: int, rng: np.random.Generator, max_tokens: int) -> list[str]:
        return _generate(self, rating, rng, max_tokens)


class RatingUnigram:
    """Piso trivial: sorteia palavras pela frequência delas nas resenhas da nota pedida.

    Não olha o contexto: "bom" pode vir depois de "não", e o texto não tem ordem. Se o trigrama não
    superar isto com folga, a ordem das palavras não está sendo aprendida.
    """

    def __init__(self) -> None:
        self.counts: dict[str, Counter[str]] = {}
        self.vocabulary: set[str] = {UNK, EOS}
        self._sampler = _Sampler()

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        for sequence in sequences:
            rating = sequence[1]
            self.counts.setdefault(rating, Counter()).update(sequence[2:])
            self.vocabulary.update(sequence[2:])

    def prob(self, word: str, history: Sequence[str]) -> float:
        # Suavização de Laplace (+1 em toda contagem): uma palavra que existe no vocabulário mas
        # nunca apareceu numa nota não pode ter probabilidade zero.
        if history[1] not in self.counts:
            raise ValueError(f"nota sem nenhuma resenha no treino: {history[1]}")
        counts = self.counts[history[1]]
        return (counts[word] + 1) / (counts.total() + len(self.vocabulary))

    def sample_next(self, history: Sequence[str], rng: np.random.Generator) -> str:
        # Sorteia com o mesmo +1 do Laplace usado em prob(), para a amostra seguir o modelo.
        rating = history[1]
        return self._sampler.sample(
            (rating,), lambda: _plus_one(self.counts[rating], self.vocabulary), rng
        )

    def generate(self, rating: int, rng: np.random.Generator, max_tokens: int) -> list[str]:
        return _generate(self, rating, rng, max_tokens)


def _plus_one(counts: Counter[str], vocabulary: set[str]) -> Counter[str]:
    """Contagens com +1 em toda palavra do vocabulário (Laplace), para sortear."""
    return Counter({word: counts[word] + 1 for word in vocabulary})


def _generate(
    model: InterpolatedTrigram | RatingUnigram,
    rating: int,
    rng: np.random.Generator,
    max_tokens: int,
) -> list[str]:
    """Gera palavra por palavra até ``<eos>`` ou ``max_tokens`` passos; devolve só as palavras."""
    history = [BOS, rating_token(rating)]
    words: list[str] = []
    # Conta passos, e não palavras: assim o laço termina mesmo se só sair <unk>.
    for _ in range(max_tokens):
        word = model.sample_next(history, rng)
        if word == EOS:
            break
        # <unk> é um "balde" de palavras raras, e não uma palavra que se possa escrever. Ele entra
        # no histórico, para o contexto seguir correto, mas fica fora do texto gerado.
        history.append(word)
        if word != UNK:
            words.append(word)
    return words
