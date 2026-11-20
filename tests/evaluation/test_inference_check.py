"""Medida do pipeline completo: contagem de tentativas, motivos, falhas e resenhas entregues."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer

from nano_review.evaluation.inference_check import run_config
from nano_review.inference import generator as generator_module
from nano_review.inference.generator import ReviewGenerator, Settings
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel
from nano_review.tokenization.bpe import train_tokenizer


def _gen(settings: Settings) -> tuple[ReviewGenerator, Tokenizer]:
    tokenizer = train_tokenizer(["Produto bom, chegou rápido.", "bla bla bla"] * 5, vocab_size=60)
    config = TransformerConfig(
        tokenizer.get_vocab_size(), block_size=24, d_model=16, n_layers=1, n_heads=2, dropout=0.0
    )
    torch.manual_seed(0)
    return ReviewGenerator(TransformerLanguageModel(config), tokenizer, settings), tokenizer


def test_counts_attempts_reasons_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    gen, tokenizer = _gen(Settings(max_attempts=2))
    # Por resenha, a amostragem falsa devolve: loop, depois bom (2 tentativas); bom (1); loop, loop
    # (esgota). São 5 notas x 1 resenha, e a fila se repete.
    script = ["bla bla bla", "Produto bom", "Produto bom", "bla bla bla", "bla bla bla"]
    queue = script * 3

    def fake(
        model: object, prefix: list[int], eos: int, *args: object, **kwargs: object
    ) -> list[int]:
        return [*prefix, *tokenizer.encode(queue.pop(0)).ids, eos]

    monkeypatch.setattr(generator_module, "sample", fake)
    rows, stats = run_config(gen, per_rating=1, seed=0, temperature=1.0, top_k=None)
    # Notas 1 a 5: (loop, bom) / (bom) / (loop, loop: falha) / (loop, bom) / (bom).
    assert [row["requested_rating"] for row in rows] == [1, 2, 4, 5]
    assert [row["attempts"] for row in rows] == [2, 1, 2, 1]
    assert stats["delivered"] == 4 and stats["failures"] == 1
    assert stats["rejected"] == {"too_long": 0, "empty": 0, "repetition": 4, "personal_data": 0}
    assert stats["attempts"] == {"mean": 1.5, "max": 2, "more_than_one": 0.5}
    ms = stats["ms_per_review"]
    assert isinstance(ms, dict) and set(ms) == {"mean", "median", "p95"}


def test_every_review_failing_does_not_break_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    gen, tokenizer = _gen(Settings(max_attempts=1))

    def fake(
        model: object, prefix: list[int], eos: int, *args: object, **kwargs: object
    ) -> list[int]:
        return [*prefix, *tokenizer.encode("bla bla bla").ids, eos]

    monkeypatch.setattr(generator_module, "sample", fake)
    rows, stats = run_config(gen, per_rating=1, seed=0, temperature=1.0, top_k=None)
    assert rows == [] and stats["failures"] == 5 and stats["attempts"] is None
    ms = stats["ms_per_review"]
    assert isinstance(ms, dict) and ms["mean"] >= 0  # as falhas também entram no tempo
