"""Avaliação do Transformer com a régua do charter: H1 a H6 na validação.

Mede o Transformer com as **mesmas** funções e na **mesma** base que mediram o trigrama
(``nano_review.baseline.evaluate``), para os números serem comparáveis:

1. **Perplexity por palavra (H1)** na validação inteira, inclusive nas resenhas maiores que o
   ``block_size`` (o treino descarta essas; a avaliação não pode, porque são as mais difíceis).
2. **Amostras** (H2, H4, H5, H6): ``N`` resenhas por nota, sorteadas da distribuição do modelo
   (temperatura 1, sem top-k, como o trigrama), julgadas pelo mesmo juiz e pelas mesmas métricas.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.evaluation.transformer_eval --checkpoint models/transformer.pt \\
        --tokenizer models/tokenizer.json --judge models/judge.joblib \\
        --train data/gold/train.parquet --validation data/gold/validation.parquet \\
        --baseline reports/metrics/baseline.json --train-metrics reports/metrics/train.json \\
        --metrics reports/metrics/transformer_eval.json \\
        --samples reports/samples/transformer_eval.jsonl --samples-per-rating 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from nano_review.baseline.evaluate import RATINGS, score_texts
from nano_review.baseline.ngram import BOS, EOS, rating_token, word_tokenize
from nano_review.data.dataset import drop_rating, given_tokens
from nano_review.models.transformer import TransformerConfig, TransformerLanguageModel, generate
from nano_review.reproducibility import set_seed
from nano_review.tokenization.bpe import PAD, decode, encode_reviews, special_id
from nano_review.training.train import configure_mlflow

# Limiares do charter (seção 7), fixados antes do Transformer: (medida, "<=" ou ">=", limiar).
# Repetidos aqui para o relatório dizer "passou" ou "não passou" sozinho; a fonte da verdade
# continua sendo o charter.
THRESHOLDS = {
    "H1": ("perplexity por palavra", "<=", 72.0),
    "H2": ("coerência macro", ">=", 0.57),
    "H4": ("cópia exata", "<=", 0.01),
    "H5": ("cópia de 8 palavras", "<=", 0.07),
    "H6": ("distinct-2", ">=", 0.59),
}


@dataclass(frozen=True)
class Window:
    """Um pedaço de sequência que cabe no modelo e os alvos que ele pontua.

    ``tokens`` é a entrada do modelo. Os alvos ``targets`` são previstos, em ordem, pelas posições
    ``offset``, ``offset + 1``, ... da entrada.
    """

    tokens: list[int]
    offset: int
    targets: list[int]


def sliding_windows(
    ids: Sequence[int], block_size: int, stride: int, given: int = 2
) -> list[Window]:
    """Quebra uma sequência ``<bos> <nota_k> texto <eos>`` em janelas que cabem no modelo.

    Os alvos são as posições ``given`` em diante (texto e ``<eos>``): o prefixo é dado, e não
    previsto. ``given`` é 2 com a nota e 1 sem ela (``<bos> texto <eos>``, a ablação E6). Cada
    alvo é pontuado **uma vez só**, com o máximo de contexto que cabe.

    - Sequência curta (o caso de 97,5% das resenhas): uma janela só, igual ao treino.
    - Sequência longa: depois da primeira janela, cada nova janela pontua os próximos ``stride``
      alvos. A entrada é ``<bos> <nota_k>`` mais o texto imediatamente anterior que couber. Repetir
      o prefixo mantém a nota à vista do modelo, como ela estava no treino; sem ele, a segunda
      metade de uma resenha longa seria prevista sem saber a nota.

    Para prever ``ids[j]``, a entrada precisa terminar em ``ids[j - 1]``. Com a entrada
    ``prefixo + ids[start:end]``, o token ``ids[j - 1]`` fica na posição ``given + j - 1 - start``.
    """
    if not 1 <= stride <= block_size - given:
        raise ValueError("o stride precisa estar entre 1 e block_size - given")
    prefix = list(ids[:given])
    windows = []
    first, last = given, len(ids)  # alvos: ids[given:]
    a = first
    while a < last:
        # A primeira janela pontua tudo o que cabe; as seguintes, ``stride`` alvos por vez.
        b = min(last, block_size + 1) if a == first else min(last, a + stride)
        # O máximo de contexto que cabe junto com o prefixo: given + (b - 1 - start) <= block_size.
        start = max(given, b - 1 + given - block_size)
        windows.append(
            Window(
                tokens=prefix + list(ids[start : b - 1]),
                offset=given + a - 1 - start,
                targets=list(ids[a:b]),
            )
        )
        a = b
    return windows


@torch.no_grad()
def total_log_prob(
    model: TransformerLanguageModel,
    windows: Sequence[Window],
    pad_id: int,
    batch_size: int = 64,
) -> float:
    """Soma do log P(alvo) de todas as janelas, em lotes de janelas de tamanho parecido."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(windows)), key=lambda i: len(windows[i].tokens))
    total = 0.0
    for start in range(0, len(order), batch_size):
        batch = [windows[i] for i in order[start : start + batch_size]]
        length = max(len(w.tokens) for w in batch)
        # Padding à direita: a máscara causal impede que as posições pontuadas o vejam.
        x = torch.full((len(batch), length), pad_id, dtype=torch.long)
        for row, window in enumerate(batch):
            x[row, : len(window.tokens)] = torch.tensor(window.tokens)
        log_probs = F.log_softmax(model(x), dim=-1)  # (B, T, V)
        for row, window in enumerate(batch):
            positions = torch.arange(window.offset, window.offset + len(window.targets))
            targets = torch.tensor(window.targets)
            total += float(log_probs[row, positions, targets].sum())
    model.train(was_training)
    return total


def word_perplexity(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    df: pd.DataFrame,
    stride: int,
    conditioning: bool = True,
) -> dict[str, float]:
    """Perplexity por **palavra**, na unidade do trigrama (cautela 2 do charter).

    exp(−log P(textos) / número de palavras), com as palavras contadas pela tokenização do baseline
    (palavras e pontuação) mais um ``<eos>`` por resenha. log P(texto) é a soma dos log-probs dos
    tokens BPE: a probabilidade de um texto não depende de como ele foi quebrado, então dividir pelo
    número de palavras põe os dois modelos na mesma régua.

    Sem a nota (``conditioning=False``), os alvos são os mesmos (texto e ``<eos>``); só o prefixo
    muda. Assim a perplexity com e sem a nota mede exatamente o mesmo texto.
    """
    sequences = encode_reviews(tokenizer, df["review_text"].tolist(), df["overall_rating"].tolist())
    if not conditioning:
        sequences = [drop_rating(ids) for ids in sequences]
    given = given_tokens(conditioning)
    block_size = model.config.block_size
    windows = [w for ids in sequences for w in sliding_windows(ids, block_size, stride, given)]
    log_prob = total_log_prob(model, windows, special_id(tokenizer, PAD))
    n_tokens = sum(len(ids) - given for ids in sequences)  # texto + <eos>
    n_words = sum(len(word_tokenize(text)) + 1 for text in df["review_text"])
    return {
        "word_perplexity": round(math.exp(-log_prob / n_words), 2),
        "token_perplexity": round(math.exp(-log_prob / n_tokens), 2),
        "tokens_per_word": round(n_tokens / n_words, 4),
        "reviews": len(sequences),
        "reviews_over_block_size": sum(len(ids) - 1 > block_size for ids in sequences),
    }


def generate_samples(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    per_rating: int,
    seed: int,
    conditioning: bool = True,
) -> list[tuple[int, str]]:
    """``per_rating`` resenhas por nota, sorteadas com temperatura 1 e sem top-k.

    Temperatura 1 é a distribuição que o modelo aprendeu, a mesma régua do trigrama. Baixar a
    temperatura deixaria o texto mais "seguro" e poderia inflar a coerência; isso é ajuste de
    produto (Etapa 4), e não medida do modelo.

    Sem a nota, o modelo não a recebe, mas cada amostra continua registrada com a nota "pedida":
    é isso que permite medir a coerência da ablação.
    """
    generator = torch.Generator().manual_seed(seed)
    bos, eos = special_id(tokenizer, BOS), special_id(tokenizer, EOS)
    samples = []
    for rating in RATINGS:
        prefix = [bos, special_id(tokenizer, rating_token(rating))][: given_tokens(conditioning)]
        for _ in range(per_rating):
            ids = generate(model, prefix, eos, generator=generator)
            samples.append((rating, decode(tokenizer, ids)))
    return samples


def check_hypotheses(perplexity: float, scores: dict[str, object]) -> dict[str, object]:
    """Compara cada medida com o limiar do charter. O H3 depende da rubrica humana."""
    coherence, copy = scores["coherence"], scores["copy"]
    assert isinstance(coherence, dict) and isinstance(copy, dict)
    values = {
        "H1": perplexity,
        "H2": coherence["macro"],
        "H4": copy["exact"],
        "H5": copy["ngram"],
        "H6": scores["distinct_2"],
    }
    result: dict[str, object] = {}
    for name, (measure, direction, threshold) in THRESHOLDS.items():
        value = float(values[name])
        passed = value <= threshold if direction == "<=" else value >= threshold
        result[name] = {
            "measure": measure,
            "value": value,
            "threshold": f"{direction} {threshold}",
            "passed": passed,
        }
    result["H3"] = {"measure": "fluência (rubrica, às cegas)", "passed": "pendente"}
    return result


def flat_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """As medidas numéricas num dicionário plano, no formato que o MLflow aceita.

    ``{"samples": {"copy": {"exact": 0.01}}}`` vira ``{"samples.copy.exact": 0.01}``; um "passou"
    vira 1 ou 0, para o painel poder ordenar as runs por ele.
    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in flat_metrics(value).items()})
        elif isinstance(value, bool | int | float):
            flat[key] = float(value)
    return flat


def log_to_mlflow(metrics: dict[str, object], run_id: str | None, artifacts: list[Path]) -> None:
    """Grava a avaliação no run do treino que gerou o checkpoint (ou num run novo, sem ele)."""
    configure_mlflow()
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "evaluate"):
        mlflow.log_metrics(flat_metrics(metrics))
        for path in artifacts:
            mlflow.log_artifact(str(path))


def load_model(path: Path) -> tuple[TransformerLanguageModel, bool]:
    """Recria o modelo a partir do checkpoint: a configuração viaja junto com os pesos.

    Devolve também se o modelo foi treinado com o token da nota. Checkpoints anteriores ao E6 não
    guardam isso, e todos eles usavam a nota.
    """
    checkpoint = torch.load(path, weights_only=True)
    model = TransformerLanguageModel(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), bool(checkpoint.get("conditioning", True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in (
        "checkpoint",
        "tokenizer",
        "judge",
        "train",
        "validation",
        "baseline",
        "train-metrics",
        "metrics",
        "samples",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("samples-per-rating", "stride", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    model, conditioning = load_model(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    judge = joblib.load(args.judge)
    train_texts = pd.read_parquet(args.train, columns=["review_text"])["review_text"].tolist()
    validation = pd.read_parquet(args.validation)

    perplexity = word_perplexity(model, tokenizer, validation, args.stride, conditioning)
    samples = generate_samples(model, tokenizer, args.samples_per_rating, args.seed, conditioning)
    scores = score_texts(
        [rating for rating, _ in samples], [text for _, text in samples], judge, train_texts
    )
    baseline = json.loads(args.baseline.read_text())
    metrics = {
        "conditioning": conditioning,
        "validation": perplexity,
        "samples": scores,
        "hypotheses": check_hypotheses(perplexity["word_perplexity"], scores),
        # Lado a lado com o trigrama e com as resenhas reais, como na tabela do charter.
        "reference": {
            "trigram_word_perplexity": baseline["trigram"]["validation_perplexity"],
            "trigram_coherence_macro": baseline["trigram"]["coherence"]["macro"],
            "real_coherence_macro": baseline["real_validation_reference"]["coherence"]["macro"],
            "real_distinct_2": baseline["real_validation_reference"]["distinct_2"],
        },
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as f:
        for rating, text in samples:
            f.write(json.dumps({"requested_rating": rating, "text": text}, ensure_ascii=False))
            f.write("\n")
    run_id = json.loads(args.train_metrics.read_text()).get("mlflow_run_id")
    log_to_mlflow(metrics, run_id, [args.metrics, args.samples])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
