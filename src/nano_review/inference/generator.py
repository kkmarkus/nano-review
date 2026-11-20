"""O gerador do produto: nota de 1 a 5 entra, resenha curta sai, já filtrada.

É o que a API chama. Carrega o modelo e o tokenizador **uma vez** e, a cada pedido:

1. valida a entrada (nota, quantidade, temperatura, top-k, semente);
2. sorteia o texto com a decodificação escolhida na 4.1 (``docs/inference.md``);
3. reprova e sorteia de novo o texto que cai num dos filtros da 4.2, até ``max_attempts`` vezes:

   - ``too_long``: passou de ``max_text_tokens`` sem chegar ao ``<eos>``;
   - ``empty``: nenhuma palavra;
   - ``repetition``: a mesma palavra 3 vezes seguidas (o loop do "bla bla bla"). É a forma mais
     simples da repetição 1 da rubrica; o loop de uma frase inteira não é checado aqui;
   - ``personal_data``: e-mail, telefone, CPF ou URL, com os mesmos padrões da limpeza.

Se as tentativas acabam, sobe ``GenerationError``: o produto nunca entrega um texto que falhou num
filtro. Os padrões de cada limite estão justificados em ``docs/inference.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tokenizers import Tokenizer

from nano_review.baseline.ngram import BOS, EOS, rating_token
from nano_review.data.clean import URL, contains_personal_data
from nano_review.evaluation.metrics import words
from nano_review.inference.sampling import sample
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.tokenization.bpe import decode, special_id

RATINGS = (1, 2, 3, 4, 5)
REJECTIONS = ("too_long", "empty", "repetition", "personal_data")
LOOP_LENGTH = 3  # a mesma palavra 3 vezes seguidas: a forma mais simples da repetição 1 da rubrica
MAX_SEED = 2**63 - 1  # o maior inteiro que o gerador do torch aceita sem estourar


@dataclass(frozen=True)
class Settings:
    """Os padrões do produto, decididos na 4.1 e na 4.2 (``docs/inference.md``)."""

    temperature: float = 1.0
    top_k: int | None = None  # sem top-k
    max_text_tokens: int = 83  # percentil 95 do tamanho das resenhas do treino, em tokens
    max_attempts: int = 5
    max_reviews: int = 20  # resenhas por pedido
    max_temperature: float = 2.0


@dataclass(frozen=True)
class Review:
    """Uma resenha entregue, com quantas tentativas custou e por que as outras caíram."""

    rating: int
    text: str
    attempts: int
    rejected: tuple[str, ...] = field(default_factory=tuple)


class GenerationError(RuntimeError):
    """Todas as tentativas caíram num filtro. Guarda os motivos, na ordem."""

    def __init__(self, rating: int, reasons: list[str]) -> None:
        super().__init__(
            f"nota {rating}: {len(reasons)} tentativas reprovadas ({', '.join(reasons)})"
        )
        self.rating = rating
        self.reasons = reasons


def has_loop(text: str, length: int = LOOP_LENGTH) -> bool:
    """A mesma palavra ``length`` vezes seguidas. "muito muito bom" passa; "bla bla bla" não."""
    tokens = words(text)
    return any(len(set(tokens[i : i + length])) == 1 for i in range(len(tokens) - length + 1))


def rejection(text: str, finished: bool) -> str | None:
    """O motivo para descartar o texto, ou ``None`` se ele pode ser entregue."""
    if not finished:
        return "too_long"
    if not words(text):
        return "empty"
    if has_loop(text):
        return "repetition"
    # A URL é procurada no texto em minúsculas: o padrão da limpeza não pega "WWW." nem "HTTPS://".
    if contains_personal_data(text) or URL.search(text.lower()):
        return "personal_data"
    return None


def _is_int(value: object) -> bool:
    """Inteiro de verdade: ``True`` é ``int`` para o Python, mas não é uma nota nem uma semente."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_checkpoint(path: Path) -> TransformerLanguageModel:
    """O modelo a partir do checkpoint, que guarda a configuração junto com os pesos.

    Recusa o modelo da ablação E6, treinado sem a nota: com ele, a nota pedida seria ignorada.
    """
    checkpoint = torch.load(path, weights_only=True)
    if not checkpoint.get("conditioning", True):
        raise ValueError("este checkpoint foi treinado sem a nota (E6) e não serve ao produto")
    model = TransformerLanguageModel(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


class ReviewGenerator:
    """Gera resenhas para uma nota, com validação, filtros e semente opcional."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        tokenizer: Tokenizer,
        settings: Settings | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.settings = settings or Settings()
        self.bos = special_id(tokenizer, BOS)
        self.eos = special_id(tokenizer, EOS)
        self.rating_ids = {r: special_id(tokenizer, rating_token(r)) for r in RATINGS}

    @classmethod
    def from_files(
        cls, checkpoint: Path, tokenizer: Path, settings: Settings | None = None
    ) -> ReviewGenerator:
        return cls(load_checkpoint(checkpoint), Tokenizer.from_file(str(tokenizer)), settings)

    def check_request(
        self, rating: int, n: int, temperature: float, top_k: int | None, seed: int | None
    ) -> None:
        """Valida o pedido. A API valida de novo, mas o gerador não confia em quem o chama.

        Confere o tipo além do intervalo: um ``True``, um ``1.0`` ou um ``numpy.int64`` passariam
        pela comparação e quebrariam mais adiante com outro erro (um 500 na API).
        """
        if not _is_int(rating) or rating not in RATINGS:
            raise ValueError("a nota precisa ser um inteiro de 1 a 5")
        if not _is_int(n) or not 1 <= n <= self.settings.max_reviews:
            raise ValueError(
                f"o número de resenhas precisa ser um inteiro entre 1 e {self.settings.max_reviews}"
            )
        valid_number = isinstance(temperature, int | float) and not isinstance(temperature, bool)
        if not valid_number or not math.isfinite(temperature):
            raise ValueError("a temperatura precisa ser um número")
        if not 0 < temperature <= self.settings.max_temperature:
            raise ValueError(f"a temperatura precisa estar em (0, {self.settings.max_temperature}]")
        vocab = self.tokenizer.get_vocab_size()
        if top_k is not None and (not _is_int(top_k) or not 1 <= top_k <= vocab):
            raise ValueError(f"o top-k precisa ser um inteiro entre 1 e {vocab}, ou 0 (sem top-k)")
        if seed is not None and (not _is_int(seed) or not 0 <= seed <= MAX_SEED):
            raise ValueError(f"a semente precisa ser um inteiro entre 0 e {MAX_SEED}")

    def sample_review(
        self, rating: int, temperature: float, top_k: int | None, generator: torch.Generator
    ) -> Review:
        """Uma resenha: sorteia, filtra e sorteia de novo, até passar ou acabar as tentativas."""
        prefix = [self.bos, self.rating_ids[rating]]
        reasons: list[str] = []
        for attempt in range(1, self.settings.max_attempts + 1):
            ids = sample(
                self.model,
                prefix,
                self.eos,
                temperature,
                top_k,
                generator,
                # O texto pode ter até max_text_tokens tokens, e o <eos> vem depois deles.
                max_new_tokens=self.settings.max_text_tokens + 1,
            )
            text = " ".join(decode(self.tokenizer, ids).split())
            reason = rejection(text, finished=ids[-1] == self.eos)
            if reason is None:
                return Review(rating, text, attempt, tuple(reasons))
            reasons.append(reason)
        raise GenerationError(rating, reasons)

    def generate(
        self,
        rating: int,
        n: int = 1,
        temperature: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> list[Review]:
        """``n`` resenhas para a nota pedida.

        ``temperature`` e ``top_k`` omitidos usam os padrões do produto; ``top_k=0`` pede
        explicitamente "sem top-k". Com ``seed``, o mesmo pedido devolve os mesmos textos no mesmo
        ambiente (a mesma versão do torch e o mesmo número de threads).
        """
        temperature = self.settings.temperature if temperature is None else temperature
        if top_k is None:
            top_k = self.settings.top_k
        elif _is_int(top_k) and top_k == 0:  # um False não pode virar "sem top-k" sem querer
            top_k = None
        self.check_request(rating, n, temperature, top_k, seed)
        generator = torch.Generator()
        if seed is None:
            generator.seed()  # semente aleatória: cada pedido sem semente sai diferente
        else:
            generator.manual_seed(seed)
        return [self.sample_review(rating, temperature, top_k, generator) for _ in range(n)]
