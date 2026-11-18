"""Dataset e DataLoader: como as resenhas viram os tensores que entram no modelo.

O caminho completo, para uma resenha de nota 5 "Chegou rápido.":

1. Tokenização: ``<bos> <nota_5> ▁Chegou ▁rápido . <eos>`` -> IDs ``[1, 8, 713, 1199, 9, 2]``.
2. Entrada e alvo deslocados de uma posição: o modelo lê ``x`` e, em cada posição, tenta prever o
   token seguinte, que está em ``y`` na mesma posição::

       x = [1, 8, 713, 1199, 9]        (tudo menos o último)
       y = [8, 713, 1199, 9, 2]        (tudo menos o primeiro)

3. Lote: várias resenhas de tamanhos diferentes viram uma matriz ``(B, T)``, completada com
   ``<pad>``. As posições de ``y`` que não devem contar na loss viram ``-100``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import partial

import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, Sampler

from nano_review.tokenization.bpe import PAD, encode_reviews, special_id

# Valor que a ``cross_entropy`` do PyTorch ignora por padrão (``ignore_index=-100``).
IGNORE_INDEX = -100


class ReviewDataset(Dataset[list[int]]):
    """Resenhas já codificadas em IDs, no formato ``<bos> <nota_k> ... <eos>``.

    Codifica tudo uma vez, na criação: tokenizar a cada época repetiria o mesmo trabalho.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, block_size: int) -> None:
        encoded = encode_reviews(
            tokenizer, df["review_text"].tolist(), df["overall_rating"].tolist()
        )
        # A entrada do modelo é a sequência sem o último token, então cabe quem tem até
        # block_size + 1 IDs. As mais longas saem inteiras, e não cortadas: cortada, a resenha
        # perderia o <eos>, e o modelo aprenderia que resenha às vezes termina sem fim.
        self.sequences = [ids for ids in encoded if len(ids) - 1 <= block_size]
        self.n_dropped = len(encoded) - len(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


def collate(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Junta resenhas de tamanhos diferentes num lote ``(B, T)``.

    ``T`` é o tamanho da **maior resenha do lote**, e não o ``block_size``: com a mediana em ~26
    tokens, completar tudo até 128 faria a CPU calcular atenção sobre padding quase o tempo todo.

    Viram ``IGNORE_INDEX`` em ``y``:

    - o padding, que não é texto;
    - a primeira posição, em que o modelo leria ``<bos>`` e "preveria" ``<nota_k>``. A nota é
      **dada** pelo usuário, e não prevista. Assim a loss mede só o texto e o ``<eos>``, as mesmas
      posições que a perplexity do baseline de n-gramas mede.
    """
    length = max(len(ids) for ids in batch) - 1
    x = torch.full((len(batch), length), pad_id, dtype=torch.long)
    y = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    for row, ids in enumerate(batch):
        sequence = torch.tensor(ids, dtype=torch.long)
        x[row, : len(ids) - 1] = sequence[:-1]
        y[row, : len(ids) - 1] = sequence[1:]
    y[:, 0] = IGNORE_INDEX
    return x, y


class LengthBucketSampler(Sampler[list[int]]):
    """Monta lotes com resenhas de tamanho parecido, sem perder a aleatoriedade.

    Problema: com lotes sorteados ao acaso, basta uma resenha longa no lote para todas as outras
    serem completadas com ``<pad>`` até o tamanho dela. Medido nos dados reais, 75% das posições de
    um lote eram padding, e a CPU calculava atenção sobre elas à toa.

    Solução (agrupamento por comprimento):

    1. embaralha a ordem das resenhas;
    2. pega blocos de ``batch_size * bucket_factor`` resenhas e ordena cada bloco por tamanho;
    3. corta cada bloco em lotes, que agora têm resenhas de tamanho parecido;
    4. embaralha a ordem dos lotes, para o modelo não ver todas as curtas e depois todas as longas.

    Cada época sorteia uma ordem nova, derivada de ``seed + época``: o treino é aleatório, mas
    repetível.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_factor: int = 50,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_factor = bucket_factor
        self.epoch = 0

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        n = len(self.lengths)
        if self.shuffle:
            order = torch.randperm(n, generator=generator).tolist()
            chunk = self.batch_size * self.bucket_factor
        else:
            # Sem embaralhar (validação): ordena tudo de uma vez, o que dá o mínimo de padding.
            order = list(range(n))
            chunk = max(n, 1)
        batches: list[list[int]] = []
        for start in range(0, n, chunk):
            bucket = sorted(order[start : start + chunk], key=lambda i: self.lengths[i])
            batches += [
                bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)
            ]
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.lengths)
        chunk = self.batch_size * self.bucket_factor if self.shuffle else max(n, 1)
        full_chunks, rest = divmod(n, chunk)
        return full_chunks * math.ceil(chunk / self.batch_size) + math.ceil(rest / self.batch_size)


def make_loader(
    dataset: ReviewDataset, tokenizer: Tokenizer, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[list[int]]:
    """DataLoader com lotes agrupados por comprimento e ordem reproduzível."""
    sampler = LengthBucketSampler(
        [len(ids) for ids in dataset.sequences], batch_size, shuffle=shuffle, seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate, pad_id=special_id(tokenizer, PAD)),
    )
