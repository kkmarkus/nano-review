"""Rubrica humana às cegas (H3): monta o lote para pontuar e, depois, resume as notas.

São duas etapas, com uma pessoa no meio:

1. ``batch``: sorteia N amostras por nota de cada fonte e embaralha tudo. As fontes são o trigrama,
   o Transformer e resenhas **reais** da validação (o controle). A planilha é gravada **sem** a
   origem, que fica num arquivo separado (a chave), aberto só depois da pontuação.
2. ``summarize``: com a planilha pontuada, confere as notas, abre a chave e calcula as médias, a
   fração de amostras ruins e o H3.

Não é estágio do DVC: a planilha pontuada é trabalho humano, e um ``dvc repro`` não pode regerá-la.

Uso::

    python -m nano_review.evaluation.rubric batch --trigram reports/samples/baseline.jsonl \\
        --transformer reports/samples/transformer_eval.jsonl \\
        --validation data/curated/validation.parquet --tokenizer models/tokenizer.json \\
        --block-size 128 --out-dir reports/rubric --per-rating 20 --seed 2026
    python -m nano_review.evaluation.rubric summarize --sheet reports/rubric/sheet_v1.csv \\
        --key reports/rubric/key_v1.csv --metrics reports/metrics/rubric.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from nano_review.baseline.evaluate import RATINGS
from nano_review.data.dataset import fits_in_context
from nano_review.tokenization.bpe import encode_reviews

RUBRIC_VERSION = "1.0"  # a versão de docs/rubric.md usada nesta planilha
BATCH_SUFFIX = "_v1"
CRITERIA = ("fluency", "plausibility", "coherence", "repetition")
SCORES = (1, 2, 3)
SHEET_COLUMNS = ("sample_id", "requested_rating", "text", *CRITERIA, "notes")
# Nomes das fontes na chave. "real" é o controle: quantas resenhas humanas a mesma pessoa, com a
# mesma rubrica, marca como ruins. É o piso contra o qual os modelos são lidos.
TRIGRAM, TRANSFORMER, REAL = "trigram", "transformer", "real"


def load_samples(path: Path, model: str | None = None) -> pd.DataFrame:
    """Amostras de um ``.jsonl`` como ``requested_rating, text``. ``model`` filtra o do baseline."""
    df = pd.read_json(path, lines=True)
    if model is not None:
        df = df[df["model"] == model]
    return df[["requested_rating", "text"]].reset_index(drop=True)


def real_reviews(validation: pd.DataFrame, tokenizer: Tokenizer, block_size: int) -> pd.DataFrame:
    """Resenhas reais que o modelo conseguiria escrever: as que cabem no contexto, como no treino.

    Sem esse filtro, uma resenha de 500 palavras denunciaria que é humana (o Transformer para em
    ``block_size`` tokens), e a avaliação deixaria de ser cega.
    """
    encoded = encode_reviews(
        tokenizer, validation["review_text"].tolist(), validation["overall_rating"].tolist()
    )
    keep = [fits_in_context(ids, block_size) for ids in encoded]
    real = validation.loc[keep, ["overall_rating", "review_text"]]
    real = real.rename(columns={"overall_rating": "requested_rating", "review_text": "text"})
    return real.reset_index(drop=True)


def normalize_spaces(text: str) -> str:
    """Quebras de linha e espaços repetidos viram um espaço: a forma não pode denunciar a origem."""
    return " ".join(text.split())


def build_batch(
    sources: Mapping[str, pd.DataFrame], per_rating: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorteia ``per_rating`` amostras por nota de cada fonte e embaralha: (planilha, chave)."""
    rng = np.random.default_rng(seed)
    picked = []
    for name in sorted(sources):  # ordem fixa: a mesma semente sempre sorteia as mesmas amostras
        pool = sources[name]
        for rating in RATINGS:
            group = pool[pool["requested_rating"] == rating]
            if len(group) < per_rating:
                raise ValueError(
                    f"{name}: só {len(group)} amostras de nota {rating}, menos que {per_rating}"
                )
            picked.append(group.sample(per_rating, random_state=rng).assign(model=name))
    batch = pd.concat(picked).sample(frac=1, random_state=rng).reset_index(drop=True)
    width = len(str(len(batch)))
    batch["sample_id"] = [f"s{i:0{width}d}" for i in range(1, len(batch) + 1)]
    batch["text"] = batch["text"].map(normalize_spaces)

    sheet = batch[["sample_id", "requested_rating", "text"]].copy()
    for column in (*CRITERIA, "notes"):
        sheet[column] = ""
    key = batch[["sample_id", "model"]].copy()
    return sheet, key


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def batch_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    """Planilha, chave e metadados do lote."""
    return (
        out_dir / f"sheet{BATCH_SUFFIX}.csv",
        out_dir / f"key{BATCH_SUFFIX}.csv",
        out_dir / f"batch{BATCH_SUFFIX}.json",
    )


def write_batch(
    out_dir: Path, sheet: pd.DataFrame, key: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    """Grava o lote. Recusa sobrescrever: a planilha pode já estar pontuada."""
    paths = batch_paths(out_dir)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"já existe: {', '.join(existing)}. Sobrescrever apagaria a pontuação; "
            "se for mesmo para refazer o lote, apague os arquivos à mão."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path, metadata_path = paths
    # utf-8-sig (com BOM): sem ele, planilhas abrem os acentos como "Ã©".
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def read_sheet(path: Path) -> pd.DataFrame:
    """Lê a planilha pontuada, salva com vírgula ou com ponto e vírgula (Excel em português)."""
    sheet = pd.read_csv(path, encoding="utf-8-sig", dtype={"notes": str})
    if len(sheet.columns) == 1:
        sheet = pd.read_csv(path, encoding="utf-8-sig", sep=";", dtype={"notes": str})
    return sheet


def check_scores(sheet: pd.DataFrame) -> None:
    """Toda amostra precisa das quatro notas, de 1 a 3. Diz quais ids estão faltando."""
    problems = []
    for column in CRITERIA:
        values = pd.to_numeric(sheet[column], errors="coerce")
        wrong = sheet.loc[~values.isin(SCORES), "sample_id"].astype(str).tolist()
        if wrong:
            more = f" e mais {len(wrong) - 10}" if len(wrong) > 10 else ""
            problems.append(f"{column}: {', '.join(wrong[:10])}{more}")
    if problems:
        raise ValueError("notas faltando ou fora de 1 a 3 -> " + "; ".join(problems))


def _summary(group: pd.DataFrame) -> dict[str, object]:
    return {
        "samples": len(group),
        "bad_fraction": round(float(group["bad"].mean()), 4),
        "means": {column: round(float(group[column].mean()), 3) for column in CRITERIA},
    }


def summarize(sheet: pd.DataFrame, key: pd.DataFrame) -> dict[str, object]:
    """Médias por fonte e por nota pedida, fração de amostras ruins e o H3."""
    check_scores(sheet)
    if sheet["sample_id"].duplicated().any():
        raise ValueError("a planilha tem sample_id repetido")
    if set(sheet["sample_id"]) != set(key["sample_id"]):
        raise ValueError("a planilha e a chave não têm os mesmos sample_id")

    df = sheet.merge(key, on="sample_id")
    for column in CRITERIA:
        df[column] = pd.to_numeric(df[column]).astype(int)
    # Amostra ruim: nota 1 em qualquer critério (docs/rubric.md, "Como resumir").
    df["bad"] = (df[list(CRITERIA)] == 1).any(axis=1)

    sources: dict[str, object] = {}
    bad_fraction: dict[str, float] = {}
    for name, group in df.groupby("model"):
        summary = _summary(group)
        summary["by_rating"] = {
            str(rating): _summary(part) for rating, part in group.groupby("requested_rating")
        }
        summary["with_notes"] = int(group["notes"].fillna("").str.strip().ne("").sum())
        sources[str(name)] = summary
        bad_fraction[str(name)] = float(group["bad"].mean())

    hypothesis: dict[str, object] | None = None
    if TRIGRAM in bad_fraction and TRANSFORMER in bad_fraction:
        hypothesis = {
            "rule": "fração de amostras ruins do Transformer menor que a do trigrama",
            "transformer": round(bad_fraction[TRANSFORMER], 4),
            "trigram": round(bad_fraction[TRIGRAM], 4),
            "passed": bad_fraction[TRANSFORMER] < bad_fraction[TRIGRAM],
        }
    return {"rubric_version": RUBRIC_VERSION, "sources": sources, "H3": hypothesis}


def _batch(args: argparse.Namespace) -> None:
    validation = pd.read_parquet(args.validation)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sources = {
        TRIGRAM: load_samples(args.trigram, model=TRIGRAM),
        TRANSFORMER: load_samples(args.transformer),
        REAL: real_reviews(validation, tokenizer, args.block_size),
    }
    sheet, key = build_batch(sources, args.per_rating, args.seed)
    metadata = {
        "rubric_version": RUBRIC_VERSION,
        "seed": args.seed,
        "per_rating": args.per_rating,
        "rows": len(sheet),
        "sampling": "amostras da avaliação: temperatura 1, sem top-k, semente 42",
        "real_filter": f"resenhas da validação que cabem em block_size={args.block_size}",
        "sources": {
            TRIGRAM: {"path": str(args.trigram), "md5": _md5(args.trigram)},
            TRANSFORMER: {"path": str(args.transformer), "md5": _md5(args.transformer)},
            REAL: {"path": str(args.validation), "md5": _md5(args.validation)},
        },
        "pool_sizes": {name: len(pool) for name, pool in sources.items()},
    }
    write_batch(args.out_dir, sheet, key, metadata)
    print(f"lote gravado em {args.out_dir}: {len(sheet)} amostras, {len(sources)} fontes")


def _summarize(args: argparse.Namespace) -> None:
    result = summarize(read_sheet(args.sheet), pd.read_csv(args.key, encoding="utf-8-sig"))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["H3"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)

    batch = commands.add_parser("batch", help="monta a planilha cega e a chave")
    batch.add_argument("--trigram", type=Path, required=True)
    batch.add_argument("--transformer", type=Path, required=True)
    batch.add_argument("--validation", type=Path, required=True)
    batch.add_argument("--tokenizer", type=Path, required=True)
    batch.add_argument("--block-size", type=int, required=True)
    batch.add_argument("--out-dir", type=Path, required=True)
    batch.add_argument("--per-rating", type=int, required=True)
    batch.add_argument("--seed", type=int, required=True)
    batch.set_defaults(run=_batch)

    summary = commands.add_parser("summarize", help="resume a planilha já pontuada")
    summary.add_argument("--sheet", type=Path, required=True)
    summary.add_argument("--key", type=Path, required=True)
    summary.add_argument("--metrics", type=Path, required=True)
    summary.set_defaults(run=_summarize)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
