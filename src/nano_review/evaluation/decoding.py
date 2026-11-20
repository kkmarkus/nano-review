"""Escolha da decodificação (Etapa 4.1): temperatura e top-k, sem tocar no teste.

Para cada combinação de temperatura e top-k, sorteia ``N`` resenhas por nota e mede o que muda com a
decodificação. A perplexity (H1) não entra: ela mede o modelo, e não a forma de sortear.

- H2 (coerência), H4 (cópia exata), H5 (cópia de 8 palavras) e H6 (distinct-2), com os limiares do
  charter;
- palavras inventadas por 100 palavras, a medida da análise de erros (``docs/error_analysis.md``).

A regra de escolha foi escrita em ``docs/inference.md`` **antes** da varredura rodar. Entre as
combinações que passam H2, H4, H5 e H6, vence a de menos palavras inventadas por 100 palavras; no
empate, a de maior distinct-2.

Nada aqui lê resenhas da validação ou do teste. As amostras são geradas pelo modelo, a cópia e o
vocabulário vêm do treino, e o juiz foi treinado no treino.

Uso (normalmente pelo ``dvc repro decoding``)::

    python -m nano_review.evaluation.decoding --checkpoint models/transformer.pt \\
        --tokenizer models/tokenizer.json --judge models/judge.joblib \\
        --train data/curated/train.parquet --metrics reports/metrics/decoding.json \\
        --samples reports/samples/decoding.jsonl --temperatures 1.0 0.8 --top-ks 0 50 \\
        --samples-per-rating 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from tokenizers import Tokenizer

from nano_review.baseline.evaluate import RATINGS, score_texts
from nano_review.baseline.ngram import BOS, EOS, rating_token
from nano_review.evaluation.error_analysis import alpha_words, train_vocabulary
from nano_review.evaluation.transformer_eval import THRESHOLDS, flat_metrics, load_model
from nano_review.inference.sampling import sample
from nano_review.models.transformer import TransformerLanguageModel
from nano_review.tokenization.bpe import decode, special_id
from nano_review.training.train import configure_mlflow

# As hipóteses que dependem das amostras. O H1 é do modelo, e o H3 é da rubrica humana.
SAMPLE_HYPOTHESES = ("H2", "H4", "H5", "H6")


def _number(value: object) -> float:
    """As métricas chegam como ``object`` dos dicionários; aqui elas voltam a ser número."""
    if not isinstance(value, int | float):
        raise TypeError(f"esperava um número, e veio {value!r}")
    return float(value)


def generate_with(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    per_rating: int,
    seed: int,
    temperature: float,
    top_k: int | None,
) -> list[tuple[int, str]]:
    """``per_rating`` resenhas por nota, com a mesma semente em todas as combinações.

    A mesma semente não garante os mesmos textos (o top-k muda o sorteio), mas tira uma fonte de
    diferença entre as combinações: o ponto de partida do gerador.
    """
    generator = torch.Generator().manual_seed(seed)
    bos, eos = special_id(tokenizer, BOS), special_id(tokenizer, EOS)
    samples = []
    for rating in RATINGS:
        prefix = [bos, special_id(tokenizer, rating_token(rating))]
        for _ in range(per_rating):
            ids = sample(model, prefix, eos, temperature, top_k, generator)
            samples.append((rating, decode(tokenizer, ids)))
    return samples


def invented_rate(texts: Sequence[str], vocabulary: set[str]) -> float:
    """Palavras que não existem no treino, por 100 palavras (números e datas não contam)."""
    tokens = [word for text in texts for word in alpha_words(text)]
    invented = sum(word not in vocabulary for word in tokens)
    return round(100 * invented / max(len(tokens), 1), 3)


def measure(
    samples: Sequence[tuple[int, str]],
    judge: Pipeline,
    train_texts: Sequence[str],
    vocabulary: set[str],
) -> dict[str, object]:
    """As métricas de amostra do charter, a taxa de palavras inventadas e quais hipóteses passam."""
    requested = [rating for rating, _ in samples]
    texts = [text for _, text in samples]
    scores = score_texts(requested, texts, judge, train_texts)
    coherence, copy = scores["coherence"], scores["copy"]
    assert isinstance(coherence, dict) and isinstance(copy, dict)
    values = {
        "H2": _number(coherence["macro"]),
        "H4": _number(copy["exact"]),
        "H5": _number(copy["ngram"]),
        "H6": _number(scores["distinct_2"]),
    }
    passed = {}
    for name in SAMPLE_HYPOTHESES:
        _, direction, threshold = THRESHOLDS[name]
        value = values[name]
        passed[name] = value <= threshold if direction == "<=" else value >= threshold
    return {
        **scores,
        "invented_per_100_words": invented_rate(texts, vocabulary),
        "passed": passed,
        "passes_all": all(passed.values()),
    }


def choose(results: Sequence[dict[str, object]]) -> dict[str, object] | None:
    """A regra de ``docs/inference.md``: passa H2, H4, H5 e H6; menos inventadas; maior distinct-2.

    Devolve ``None`` se nenhuma combinação passa nas quatro, e nesse caso a decisão volta para o
    autor, com os números na mão.
    """
    eligible = [result for result in results if result["passes_all"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda r: (_number(r["invented_per_100_words"]), -_number(r["distinct_2"])),
    )


def label(temperature: float, top_k: int | None) -> str:
    """Nome curto da combinação, o mesmo no JSON e no MLflow: ``t0.8-k50`` ou ``t1.0-k0``."""
    return f"t{temperature}-k{top_k or 0}"


def sweep(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    judge: Pipeline,
    train_texts: Sequence[str],
    temperatures: Sequence[float],
    top_ks: Sequence[int | None],
    per_rating: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Todas as combinações: as métricas de cada uma e as amostras, com a combinação de origem."""
    vocabulary = train_vocabulary(train_texts)
    results: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for temperature in temperatures:
        for top_k in top_ks:
            samples = generate_with(model, tokenizer, per_rating, seed, temperature, top_k)
            result: dict[str, object] = {
                "label": label(temperature, top_k),
                "temperature": temperature,
                "top_k": top_k,
                **measure(samples, judge, train_texts, vocabulary),
            }
            results.append(result)
            rows.extend(
                {"label": result["label"], "requested_rating": rating, "text": text}
                for rating, text in samples
            )
            print(
                f"{result['label']}: inventadas {result['invented_per_100_words']}/100, "
                f"passa H2/H4/H5/H6: {result['passes_all']}",
                flush=True,
            )
    return results, rows


def log_to_mlflow(results: Sequence[dict[str, object]], prefix: str = "decoding") -> None:
    """Um run por combinação (``decoding-t1.0-k0`` etc.), para comparar no painel do MLflow."""
    configure_mlflow()
    for result in results:
        with mlflow.start_run(run_name=f"{prefix}-{result['label']}"):
            mlflow.log_params({"temperature": result["temperature"], "top_k": result["top_k"]})
            mlflow.log_metrics(flat_metrics({k: v for k, v in result.items() if k != "label"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    # 0 quer dizer "sem top-k": o YAML do params.yaml não tem um jeito simples de passar None.
    parser.add_argument("--top-ks", type=int, nargs="+", required=True)
    parser.add_argument("--samples-per-rating", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    model, conditioning = load_model(args.checkpoint)
    if not conditioning:
        raise ValueError("a decodificação é escolhida para o modelo com a nota, e não para o E6")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    judge = joblib.load(args.judge)
    train_texts = pd.read_parquet(args.train)["review_text"].tolist()
    top_ks = [k or None for k in args.top_ks]

    results, rows = sweep(
        model,
        tokenizer,
        judge,
        train_texts,
        args.temperatures,
        top_ks,
        args.samples_per_rating,
        args.seed,
    )
    chosen = choose(results)
    report = {
        "rule": "passa H2, H4, H5 e H6; menos inventadas por palavra; empate: maior distinct-2",
        "chosen": chosen["label"] if chosen else None,
        "results": results,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_to_mlflow(results)
    print(f"escolhida: {report['chosen']}")


if __name__ == "__main__":
    main()
