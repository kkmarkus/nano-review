"""Do DataFrame ao lote: formato, deslocamento, máscaras e ordem reproduzível."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer

from nano_review.data.dataset import (
    IGNORE_INDEX,
    LengthBucketSampler,
    ReviewDataset,
    collate,
    make_loader,
)
from nano_review.tokenization.bpe import train_tokenizer

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    texts = ["Produto muito bom, recomendo.", "Não gostei, veio quebrado.", "Ok."] * 20
    return train_tokenizer(texts, vocab_size=100)


def _df(texts: list[str], ratings: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"review_text": texts, "overall_rating": ratings})


def test_dataset_encodes_with_bos_rating_and_eos(tokenizer: Tokenizer) -> None:
    dataset = ReviewDataset(_df(["Ok."], [3]), tokenizer, block_size=32)
    ids = dataset[0]
    assert ids[0] == BOS_ID
    assert ids[1] == tokenizer.token_to_id("<nota_3>")
    assert ids[-1] == EOS_ID


def test_dataset_drops_reviews_longer_than_block(tokenizer: Tokenizer) -> None:
    long_text = " ".join(["Produto muito bom"] * 20)
    dataset = ReviewDataset(_df(["Ok.", long_text], [5, 5]), tokenizer, block_size=16)
    assert len(dataset) == 1
    assert dataset.n_dropped == 1
    # Cabe exatamente quem tem block_size + 1 IDs (a entrada perde o último).
    assert all(len(ids) - 1 <= 16 for ids in dataset.sequences)


def test_collate_shifts_pads_and_masks() -> None:
    batch = [[1, 8, 10, 11, 2], [1, 4, 12, 2]]
    x, y = collate(batch, pad_id=PAD_ID)

    assert x.shape == y.shape == (2, 4)  # T = maior resenha - 1
    assert x.tolist() == [[1, 8, 10, 11], [1, 4, 12, PAD_ID]]
    # Alvo = próximo token. A posição 0 (que "preveria" a nota) e o padding são ignorados.
    assert y.tolist() == [
        [IGNORE_INDEX, 10, 11, 2],
        [IGNORE_INDEX, 12, 2, IGNORE_INDEX],
    ]


def test_collate_counts_only_text_and_eos_as_targets() -> None:
    batch = [[1, 8, 10, 11, 2], [1, 4, 12, 2]]
    _, y = collate(batch, pad_id=PAD_ID)
    # Cada resenha prevê texto + <eos> = len - 2 tokens (sem <bos> e sem <nota_k>).
    assert int((y != IGNORE_INDEX).sum()) == sum(len(ids) - 2 for ids in batch)


def test_loader_order_is_reproducible(tokenizer: Tokenizer) -> None:
    texts = [f"Produto {'muito ' * i}bom." for i in range(12)]
    dataset = ReviewDataset(_df(texts, [5] * 12), tokenizer, block_size=64)

    def first_batch(seed: int) -> torch.Tensor:
        loader = make_loader(dataset, tokenizer, batch_size=4, shuffle=True, seed=seed)
        batch: tuple[torch.Tensor, torch.Tensor] = next(iter(loader))
        return batch[0]

    assert torch.equal(first_batch(7), first_batch(7))
    # Garante que o teste acima não passa "de graça" por uma semente ignorada.
    assert not torch.equal(first_batch(7), first_batch(8))


# Comprimentos com cauda longa, como os reais: muitas curtas e algumas longas.
SKEWED = [10] * 90 + [100] * 10


@pytest.mark.parametrize("shuffle", [True, False])
def test_bucket_sampler_yields_every_index_once(shuffle: bool) -> None:
    sampler = LengthBucketSampler(SKEWED, batch_size=8, shuffle=shuffle, seed=0, bucket_factor=3)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(SKEWED)))
    assert len(batches) == len(sampler)


def test_bucket_sampler_reduces_padding() -> None:
    def padding_fraction(batches: list[list[int]]) -> float:
        total = sum(max(SKEWED[i] for i in batch) * len(batch) for batch in batches)
        return 1 - sum(SKEWED) / total

    # Lotes aleatórios: as 10 longas espalhadas contaminam quase todos os lotes.
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(100, generator=generator).tolist()
    random_batches = [order[i : i + 8] for i in range(0, 100, 8)]
    bucketed = list(LengthBucketSampler(SKEWED, batch_size=8, shuffle=True, seed=0))
    assert padding_fraction(bucketed) < padding_fraction(random_batches) / 2


def test_bucket_sampler_changes_order_between_epochs_but_repeats_with_seed() -> None:
    first = LengthBucketSampler(SKEWED, batch_size=8, shuffle=True, seed=0, bucket_factor=3)
    second = LengthBucketSampler(SKEWED, batch_size=8, shuffle=True, seed=0, bucket_factor=3)
    epoch_1 = list(first)
    assert epoch_1 == list(second)  # mesma semente, mesma primeira época
    assert list(first) != epoch_1  # a segunda época sorteia outra ordem
