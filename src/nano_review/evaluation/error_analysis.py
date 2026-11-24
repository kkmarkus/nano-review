"""Análise de erros do Transformer (sub-etapa 3.5): mede as hipóteses de causa das amostras ruins.

Parte da rubrica já pontuada (planilha e chave) e da classificação dos erros feita à mão em
``error_types_v1.csv``, com uma linha por amostra ruim do Transformer. Mede três coisas:

- **palavras inventadas:** palavras que não aparecem em nenhuma resenha do treino. Resenha real
  também tem, por erro de digitação ou nome raro, e por isso o controle de resenhas reais dá a
  referência;
- **onde o texto quebra:** a posição da primeira palavra inventada, relativa (0 = início, 1 = fim) e
  absoluta (o número da palavra). A relativa sozinha engana: com a mesma taxa de erro por palavra, o
  texto longo tem a primeira inventada mais cedo, em proporção;
- **nota rara:** a fração de amostras ruins de cada nota, ao lado da fração dessa nota no treino.

Uso::

    python -m nano_review.evaluation.error_analysis --sheet reports/rubric/sheet_v1.csv \\
        --key reports/rubric/key_v1.csv --types reports/rubric/error_types_v1.csv \\
        --train data/gold/train.parquet --metrics reports/metrics/error_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nano_review.evaluation.metrics import words
from nano_review.evaluation.rubric import TRANSFORMER, read_sheet, scored_table

# Tipos de erro de docs/error_analysis.md. Cada amostra ruim recebe um só, o que mais pesa nela.
ERROR_TYPES = ("drift", "garbled", "mixed_topics", "wrong_tone", "loop")


def alpha_words(text: str) -> list[str]:
    """Só as palavras com letras: número, data e medida ("29", "2x4gb") não são inventados."""
    return [word for word in words(text) if word.isalpha()]


def train_vocabulary(texts: Iterable[str]) -> set[str]:
    """Todas as palavras que aparecem em pelo menos uma resenha do treino."""
    return {word for text in texts for word in alpha_words(text)}


def text_profile(text: str, vocabulary: set[str]) -> tuple[int, int, float, float]:
    """Palavras, palavras inventadas e a posição da primeira inventada, relativa e absoluta.

    A relativa vai de 0 (primeira palavra) até perto de 1 (última). A absoluta é o número da palavra
    (1 = primeira). Sem palavra inventada, as duas são NaN.
    """
    tokens = alpha_words(text)
    invented = [i for i, word in enumerate(tokens) if word not in vocabulary]
    if not invented:
        return len(tokens), 0, float("nan"), float("nan")
    return len(tokens), len(invented), invented[0] / len(tokens), float(invented[0] + 1)


def profile_table(scored: pd.DataFrame, vocabulary: set[str]) -> pd.DataFrame:
    """Acrescenta ``n_words``, ``n_invented``, ``first_invented`` e ``first_invented_word``."""
    profiles = [text_profile(text, vocabulary) for text in scored["text"]]
    return scored.assign(
        n_words=[p[0] for p in profiles],
        n_invented=[p[1] for p in profiles],
        first_invented=[p[2] for p in profiles],
        first_invented_word=[p[3] for p in profiles],
    )


def _median_or_none(values: pd.Series) -> float | None:
    return round(float(values.median()), 3) if len(values) else None


def _group_summary(group: pd.DataFrame) -> dict[str, object]:
    with_invented = group["n_invented"] > 0
    return {
        "samples": len(group),
        "median_words": float(group["n_words"].median()),
        "with_invented_word": round(float(with_invented.mean()), 3),
        "invented_per_100_words": round(
            100 * float(group["n_invented"].sum()) / max(int(group["n_words"].sum()), 1), 2
        ),
        # As posições só fazem sentido entre as amostras que têm palavra inventada.
        "median_first_invented": _median_or_none(group.loc[with_invented, "first_invented"]),
        "median_first_invented_word": _median_or_none(
            group.loc[with_invented, "first_invented_word"]
        ),
    }


def read_types(path: Path) -> pd.DataFrame:
    """Lê a classificação, salva com vírgula ou ponto e vírgula, com ou sem BOM (Excel)."""
    types = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if len(types.columns) == 1:
        types = pd.read_csv(path, encoding="utf-8-sig", sep=";", dtype=str)
    return types


def check_types(types: pd.DataFrame, scored: pd.DataFrame) -> None:
    """A classificação cobre exatamente as amostras ruins do Transformer, uma vez cada."""
    # Célula vazia vira "", e não NaN, para cair na mensagem de tipo desconhecido.
    unknown = sorted(set(types["error_type"].fillna("").astype(str)) - set(ERROR_TYPES))
    if unknown:
        raise ValueError(f"tipos de erro desconhecidos: {unknown}")
    if types["sample_id"].duplicated().any():
        raise ValueError("a classificação tem sample_id repetido")
    bad = scored.loc[(scored["model"] == TRANSFORMER) & scored["bad"], "sample_id"]
    if set(types["sample_id"]) != set(bad):
        raise ValueError("a classificação não cobre exatamente as amostras ruins do Transformer")


def analyze(scored: pd.DataFrame, types: pd.DataFrame, train: pd.DataFrame) -> dict[str, object]:
    """As três medidas por fonte (todas, ruins e ok), mais a contagem dos tipos de erro."""
    check_types(types, scored)
    table = profile_table(scored, train_vocabulary(train["review_text"]))

    by_source: dict[str, object] = {}
    for source, group in table.groupby("model"):
        summary = {"all": _group_summary(group)}
        summary.update(
            {("bad" if bad else "ok"): _group_summary(part) for bad, part in group.groupby("bad")}
        )
        by_source[str(source)] = summary

    transformer = table[table["model"] == TRANSFORMER]
    train_share = train["overall_rating"].value_counts(normalize=True)
    by_rating = {
        str(rating): {
            # Indexação direta: uma nota ausente do treino (ou de outro tipo) dá erro, e não 0.
            "train_share": round(float(train_share.loc[rating]), 3),
            "bad_fraction": round(float(part["bad"].mean()), 3),
        }
        for rating, part in transformer.groupby("requested_rating")
    }

    merged = types.merge(table, on="sample_id")
    error_types = {
        name: {
            "samples": int((merged["error_type"] == name).sum()),
            "by_rating": {
                str(rating): int(count)
                for rating, count in merged.loc[merged["error_type"] == name, "requested_rating"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        for name in ERROR_TYPES
    }
    return {"by_source": by_source, "transformer_by_rating": by_rating, "error_types": error_types}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--types", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    scored = scored_table(read_sheet(args.sheet), pd.read_csv(args.key, encoding="utf-8-sig"))
    result = analyze(scored, read_types(args.types), pd.read_parquet(args.train))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.metrics.write_text(text, encoding="utf-8")
    print(json.dumps(result["by_source"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
