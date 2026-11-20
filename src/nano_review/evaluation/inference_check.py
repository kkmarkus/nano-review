"""Medida do pipeline completo do produto (4.2): decodificação e filtros juntos, sem tocar no teste.

A 4.1 escolheu a decodificação com o texto cru. A 4.2 acrescentou os filtros do gerador (tamanho,
repetição, dado pessoal), que mudam o que o produto entrega. Este estágio gera ``N`` resenhas por
nota com o ``ReviewGenerator``, com a mesma decodificação e os mesmos filtros da API (chama direto
``sample_review``, com um gerador só para todas as resenhas), e mede:

- se H2, H4, H5 e H6 continuam passando com os filtros;
- quantas tentativas cada resenha custou e quantas caíram em cada filtro;
- quantas esgotaram as tentativas (o erro 503 da API);
- o tempo por resenha nesta máquina, contando também as que esgotaram as tentativas.

As resenhas geradas aqui são o conjunto de onde sai o lote humano v2 (4.3): o que vai ser lido é o
que a API entrega, e ele foi gerado antes de o teste ser aberto. A semente é a da avaliação da
Etapa 3, e por isso muitas resenhas coincidem com as dela; o lote v2 exclui os textos já lidos no
lote v1 (``docs/inference.md``).

Uso (normalmente pelo ``dvc repro inference_check``)::

    python -m nano_review.evaluation.inference_check --checkpoint models/transformer.pt \\
        --tokenizer models/tokenizer.json --judge models/judge.joblib \\
        --train data/curated/train.parquet --metrics reports/metrics/inference.json \\
        --samples reports/samples/inference.jsonl --top-ks 0 50 --samples-per-rating 200 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from nano_review.evaluation.decoding import label, log_to_mlflow, measure
from nano_review.evaluation.error_analysis import train_vocabulary
from nano_review.inference.generator import RATINGS, REJECTIONS, GenerationError, ReviewGenerator


def run_config(
    gen: ReviewGenerator, per_rating: int, seed: int, temperature: float, top_k: int | None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Gera ``per_rating`` resenhas por nota e conta tentativas, motivos, falhas e tempo."""
    gen.check_request(RATINGS[0], 1, temperature, top_k, seed)
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    attempts: list[int] = []
    seconds: list[float] = []
    failures = 0
    for rating in RATINGS:
        for _ in range(per_rating):
            start = time.perf_counter()
            try:
                review = gen.sample_review(rating, temperature, top_k, generator)
            except GenerationError as error:
                failures += 1
                rejected.update(error.reasons)
                continue
            finally:
                # As que esgotam as tentativas são as mais lentas: sem elas, o tempo fica baixo.
                seconds.append(time.perf_counter() - start)
            attempts.append(review.attempts)
            rejected.update(review.rejected)
            rows.append(
                {
                    "label": label(temperature, top_k),
                    "requested_rating": rating,
                    "text": review.text,
                    "attempts": review.attempts,
                }
            )
    tries = np.asarray(attempts)
    stats: dict[str, object] = {
        "delivered": len(rows),
        "failures": failures,
        # Sem nenhuma resenha entregue, não há tentativas para resumir.
        "attempts": {
            "mean": round(float(tries.mean()), 3),
            "max": int(tries.max()),
            "more_than_one": round(float((tries > 1).mean()), 4),
        }
        if len(tries)
        else None,
        # Cada tentativa descartada conta uma vez, pelo motivo dela.
        "rejected": {reason: int(rejected[reason]) for reason in REJECTIONS},
        # A média entra porque o tempo de um pedido de n resenhas é a soma delas, e a soma depende
        # da média (puxada pela cauda), e não da mediana.
        "ms_per_review": {
            "mean": round(1000 * float(np.mean(seconds)), 1),
            "median": round(1000 * float(np.median(seconds)), 1),
            "p95": round(1000 * float(np.percentile(seconds, 95)), 1),
        },
    }
    return rows, stats


def check(
    gen: ReviewGenerator,
    judge: Pipeline,
    train_texts: list[str],
    top_ks: list[int | None],
    per_rating: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Cada top-k com a temperatura padrão do produto: métricas e as resenhas entregues."""
    vocabulary = train_vocabulary(train_texts)
    temperature = gen.settings.temperature
    results: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for top_k in top_ks:
        rows, stats = run_config(gen, per_rating, seed, temperature, top_k)
        samples = [(int(str(r["requested_rating"])), str(r["text"])) for r in rows]
        results.append(
            {
                "label": label(temperature, top_k),
                "temperature": temperature,
                "top_k": top_k,
                **measure(samples, judge, train_texts, vocabulary),
                **stats,
            }
        )
        all_rows.extend(rows)
        print(f"{label(temperature, top_k)}: {stats}", flush=True)
    return results, all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--top-ks", type=int, nargs="+", required=True)  # 0 = sem top-k
    parser.add_argument("--samples-per-rating", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    gen = ReviewGenerator.from_files(args.checkpoint, args.tokenizer)
    judge = joblib.load(args.judge)
    train_texts = pd.read_parquet(args.train)["review_text"].tolist()
    top_ks = [k or None for k in args.top_ks]
    results, rows = check(gen, judge, train_texts, top_ks, args.samples_per_rating, args.seed)

    report = {"settings": asdict(gen.settings), "results": results}
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_to_mlflow(results, prefix="inference")


if __name__ == "__main__":
    main()
