"""Agenda do LR, grupos do otimizador, avaliação por token, geração e um treino de ponta a ponta."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer

from nano_review.data.dataset import IGNORE_INDEX, ReviewDataset, make_loader
from nano_review.models.transformer import (
    TransformerConfig,
    TransformerLanguageModel,
    generate,
)
from nano_review.reproducibility import set_seed
from nano_review.tokenization.bpe import train_tokenizer
from nano_review.training import train
from nano_review.training.train import evaluate, learning_rate_at, make_optimizer

TINY = TransformerConfig(
    vocab_size=30, block_size=12, d_model=16, n_layers=1, n_heads=2, dropout=0.0
)


def test_learning_rate_schedule() -> None:
    lr = [learning_rate_at(s, max_lr=1.0, warmup_steps=10, max_steps=100) for s in range(100)]
    assert lr[0] == pytest.approx(0.1)  # warmup começa pequeno
    assert lr[9] == pytest.approx(1.0)  # e chega ao máximo no fim do warmup
    assert all(a >= b for a, b in zip(lr[10:], lr[11:], strict=False))  # depois só desce
    assert lr[99] == pytest.approx(0.1)  # e termina exatamente em 10% do máximo


def test_weight_decay_only_on_matrices() -> None:
    model = TransformerLanguageModel(TINY)
    decayed, not_decayed = make_optimizer(model, 1e-3, 0.1).param_groups
    assert decayed["weight_decay"] == 0.1 and all(p.dim() >= 2 for p in decayed["params"])
    assert not_decayed["weight_decay"] == 0.0 and all(p.dim() < 2 for p in not_decayed["params"])


class _ConstantLossModel(TransformerLanguageModel):
    """Logits fixos: a loss de cada token conhecida de antemão."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*tokens.shape, self.config.vocab_size)
        logits[..., 0] = 10.0  # quase certeza no token 0
        return logits


def test_evaluate_averages_per_token_not_per_batch() -> None:
    # Lote 1: 1 token certo (loss ~0). Lote 2: 3 tokens errados (loss alta, L). Por token, a
    # média é 3L/4; a média dos lotes daria L/2.
    model = _ConstantLossModel(TINY)
    first = (torch.zeros(1, 2, dtype=torch.long), torch.tensor([[IGNORE_INDEX, 0]]))
    second = (torch.zeros(1, 4, dtype=torch.long), torch.tensor([[IGNORE_INDEX, 1, 1, 1]]))
    wrong = torch.nn.functional.cross_entropy(model(first[0])[0, :1], torch.tensor([1])).item()
    right = torch.nn.functional.cross_entropy(model(first[0])[0, :1], torch.tensor([0])).item()
    result = evaluate(model, [first, second])  # type: ignore[arg-type]
    assert result == pytest.approx((right + 3 * wrong) / 4)


def test_generate_stops_at_eos_and_respects_block_size() -> None:
    set_seed(0)
    model = TransformerLanguageModel(TINY)
    ids = generate(model, [1, 4], eos_id=2, generator=torch.Generator().manual_seed(0))
    assert ids[:2] == [1, 4]
    assert len(ids) <= TINY.block_size + 1
    assert 2 not in ids[:-1]  # se <eos> saiu, foi o último
    same = generate(model, [1, 4], eos_id=2, generator=torch.Generator().manual_seed(0))
    assert ids == same


def test_generate_runs_to_block_size_plus_one_without_eos() -> None:
    # Com um <eos> impossível, a geração só para pelo limite: block_size tokens de entrada e mais
    # um previsto, igual ao treino. Assim o teste não passa se a geração parar cedo demais.
    set_seed(0)
    model = TransformerLanguageModel(TINY)
    ids = generate(
        model, [1, 4], eos_id=TINY.vocab_size, generator=torch.Generator().manual_seed(0)
    )
    assert len(ids) == TINY.block_size + 1


def test_training_script_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Um treino minúsculo de ponta a ponta: confere que o script grava tudo o que o DVC espera e
    # que a loss de validação cai.
    texts = ["Produto muito bom, recomendo.", "Não gostei, veio quebrado.", "Chegou rápido."]
    df = pd.DataFrame({"review_text": texts * 40, "overall_rating": [5, 1, 4] * 40})
    df.to_parquet(tmp_path / "train.parquet")
    df.to_parquet(tmp_path / "validation.parquet")
    train_tokenizer(df["review_text"], vocab_size=80).save(str(tmp_path / "tokenizer.json"))

    paths = {
        "train": "train.parquet",
        "validation": "validation.parquet",
        "tokenizer": "tokenizer.json",
        "checkpoint": "model.pt",
        "metrics": "metrics.json",
        "log": "log.csv",
        "figure": "curve.png",
        "samples": "samples.jsonl",
    }
    numbers = {
        "block-size": 32,
        "d-model": 16,
        "n-layers": 1,
        "n-heads": 2,
        "batch-size": 8,
        "warmup-steps": 5,
        "max-steps": 60,
        "eval-every": 20,
        "eval-reviews": 30,
        "samples-per-rating": 1,
        "seed": 0,
        "dropout": 0.0,
        "learning-rate": 0.01,
        "weight-decay": 0.1,
    }
    argv = ["train"]
    for key, value in paths.items():
        argv += [f"--{key}", str(tmp_path / value)]
    for key, number in numbers.items():
        argv += [f"--{key}", str(number)]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    for value in paths.values():
        assert (tmp_path / value).exists(), value
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["step"]) == [20, 40, 60]
    assert log["val_loss"].iloc[-1] < log["val_loss"].iloc[0]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["steps"] == 60

    # O checkpoint recarregado num modelo novo reproduz a loss de validação reportada.
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    model = TransformerLanguageModel(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    assert model.head.weight is model.token_embedding.weight  # a amarração sobrevive ao reload
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    loader = make_loader(ReviewDataset(df, tokenizer, 32), tokenizer, 128, False, 0)
    assert round(evaluate(model, loader), 4) == metrics["val_loss"]
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 5
