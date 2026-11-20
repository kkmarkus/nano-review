"""Teste final (Etapa 4.3): o único código do projeto que lê ``data/curated/test.parquet``.

Segue o pré-registro de ``docs/final_test.md``, escrito e commitado antes da execução:

1. **Sanidade, na validação:** recalcula a perplexity por palavra do Transformer e a do trigrama e
   confere com os números já conhecidos. Se não bater, para com erro **antes** de ler o teste.
2. **H1 no teste:** perplexity por palavra do Transformer e do trigrama nas mesmas resenhas.
3. **H2, H4, H5 e H6:** 1.000 resenhas novas, geradas pelo produto (decodificação padrão e filtros)
   com uma semente nunca usada.
4. **Referências humanas do teste:** as resenhas reais medidas com a mesma régua (teto do juiz,
   distinct-2, cópia do treino depois e antes do dedupe).
5. **Controle do lote v2:** sorteia as resenhas reais do teste (até 83 tokens) que entram no lote
   humano v2, para nenhum outro código precisar ler o teste.

Roda uma vez. Depois, o estágio é congelado com ``dvc freeze final_eval``.

Uso (normalmente pelo ``dvc repro final_eval``, uma única vez)::

    python -m nano_review.evaluation.final_eval --checkpoint models/transformer.pt \\
        --tokenizer models/tokenizer.json --judge models/judge.joblib \\
        --train data/curated/train.parquet --validation data/curated/validation.parquet \\
        --test data/curated/test.parquet --raw data/raw/B2W-Reviews01.csv \\
        --baseline reports/metrics/baseline.json \\
        --transformer-eval reports/metrics/transformer_eval.json \\
        --metrics reports/metrics/final_eval.json --samples reports/samples/final_eval.jsonl \\
        --control reports/samples/test_real_control.jsonl --stride 64 --max-repeat 3 \\
        --vocab-min-count 2 --samples-per-rating 200 --seed 2027 --control-per-rating 10 \\
        --control-seed 2028
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from tokenizers import Tokenizer

from nano_review.baseline.evaluate import copy_before_dedupe, make_sequences, score_texts
from nano_review.baseline.ngram import InterpolatedTrigram, build_vocabulary, word_tokenize
from nano_review.evaluation.decoding import measure
from nano_review.evaluation.error_analysis import train_vocabulary
from nano_review.evaluation.inference_check import run_config
from nano_review.evaluation.metrics import perplexity
from nano_review.evaluation.transformer_eval import check_hypotheses, flat_metrics, word_perplexity
from nano_review.inference.generator import ReviewGenerator
from nano_review.training.train import configure_mlflow

# Os números da validação que o estágio precisa reproduzir antes de abrir o teste. Eles estão
# gravados com 2 casas; a tolerância cobre só o arredondamento.
SANITY_TOLERANCE = 0.006


def fit_trigram(
    train: pd.DataFrame, lambdas: Sequence[float], vocab_min_count: int
) -> tuple[InterpolatedTrigram, set[str]]:
    """O trigrama do baseline, refeito no treino com os pesos λ já escolhidos na Etapa 1."""
    vocabulary = build_vocabulary(map(word_tokenize, train["review_text"]), vocab_min_count)
    trigram = InterpolatedTrigram((lambdas[0], lambdas[1], lambdas[2]))
    trigram.fit(make_sequences(train, vocabulary))
    return trigram, vocabulary


def check_sanity(measured: dict[str, float], expected: dict[str, float]) -> dict[str, object]:
    """Compara cada número recalculado com o conhecido. Para tudo se algum não bater."""
    report: dict[str, object] = {}
    wrong = []
    for name, value in measured.items():
        ok = abs(value - expected[name]) <= SANITY_TOLERANCE
        report[name] = {"measured": value, "expected": expected[name], "ok": ok}
        if not ok:
            wrong.append(f"{name}: medido {value}, esperado {expected[name]}")
    if wrong:
        raise RuntimeError("sanidade falhou, o teste não foi aberto -> " + "; ".join(wrong))
    return report


def control_sample(
    test: pd.DataFrame, tokenizer: Tokenizer, max_tokens: int, per_rating: int, seed: int
) -> pd.DataFrame:
    """Resenhas reais do teste para o lote v2: até ``max_tokens`` tokens, ``per_rating`` por nota.

    O limite é o mesmo do produto: uma resenha real muito longa denunciaria a origem no lote cego.
    """
    lengths = [len(e.ids) for e in tokenizer.encode_batch(test["review_text"].tolist())]
    short = test[[n <= max_tokens for n in lengths]]
    picked = short.groupby("overall_rating").sample(n=per_rating, random_state=seed)
    picked = picked.rename(columns={"overall_rating": "requested_rating", "review_text": "text"})
    return picked[["requested_rating", "text"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in (
        "checkpoint",
        "tokenizer",
        "judge",
        "train",
        "validation",
        "test",
        "raw",
        "baseline",
        "transformer-eval",
        "metrics",
        "samples",
        "control",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "stride",
        "max-repeat",
        "vocab-min-count",
        "samples-per-rating",
        "seed",
        "control-per-rating",
        "control-seed",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()

    gen = ReviewGenerator.from_files(args.checkpoint, args.tokenizer)
    judge = joblib.load(args.judge)
    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    baseline = json.loads(args.baseline.read_text())
    known = json.loads(args.transformer_eval.read_text())
    lambdas = [baseline["lambdas"][k] for k in ("unigram", "bigram", "trigram")]
    trigram, vocabulary = fit_trigram(train, lambdas, args.vocab_min_count)

    # 1. Sanidade, na validação. Só depois dela o teste é lido.
    sanity = check_sanity(
        {
            "transformer_validation": word_perplexity(
                gen.model, gen.tokenizer, validation, args.stride
            )["word_perplexity"],
            "trigram_validation": round(
                perplexity(trigram, make_sequences(validation, vocabulary)), 2
            ),
        },
        {
            "transformer_validation": known["validation"]["word_perplexity"],
            "trigram_validation": baseline["trigram"]["validation_perplexity"],
        },
    )
    print(f"sanidade ok: {sanity}", flush=True)

    # 2. O teste, aberto uma única vez.
    test = pd.read_parquet(args.test)
    transformer_test = word_perplexity(gen.model, gen.tokenizer, test, args.stride)
    trigram_test = round(perplexity(trigram, make_sequences(test, vocabulary)), 2)

    # 3. Resenhas novas do produto, com uma semente nunca usada.
    train_texts = train["review_text"].tolist()
    rows, stats = run_config(
        gen, args.samples_per_rating, args.seed, gen.settings.temperature, gen.settings.top_k
    )
    samples = [(int(str(r["requested_rating"])), str(r["text"])) for r in rows]
    sample_scores = {**measure(samples, judge, train_texts, train_vocabulary(train_texts)), **stats}

    # 4. Referências humanas: resenhas reais do teste, no mesmo número por nota que as geradas.
    reference = test.groupby("overall_rating").sample(
        n=args.samples_per_rating, random_state=args.seed
    )
    raw = pd.read_csv(
        args.raw, dtype={"product_id": str}, usecols=["product_id", "review_text"], low_memory=False
    )
    real_reference = {
        **score_texts(
            reference["overall_rating"].tolist(),
            reference["review_text"].tolist(),
            judge,
            train_texts,
        ),
        "copy_before_dedupe": copy_before_dedupe(raw, train, test, args.max_repeat),
    }

    # 5. O controle do lote v2.
    control = control_sample(
        test,
        gen.tokenizer,
        gen.settings.max_text_tokens,
        args.control_per_rating,
        args.control_seed,
    )

    report: dict[str, object] = {
        "sanity": sanity,
        "test": {
            "reviews": len(test),
            "transformer": transformer_test,
            "trigram_word_perplexity": trigram_test,
        },
        "samples": sample_scores,
        "real_test_reference": real_reference,
        "hypotheses": check_hypotheses(transformer_test["word_perplexity"], sample_scores),
        "control": {
            "per_rating": args.control_per_rating,
            "max_tokens": gen.settings.max_text_tokens,
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for path, records in (
        (args.samples, rows),
        (args.control, control.to_dict("records")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=int) + "\n")

    configure_mlflow()
    with mlflow.start_run(run_name="final-test"):
        mlflow.log_metrics(flat_metrics({k: v for k, v in report.items() if k != "control"}))
    print(json.dumps(report["hypotheses"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
